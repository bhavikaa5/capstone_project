"""Invariants the cleaned dataset must satisfy before step 2 may consume it."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data.loader import assert_no_leakage, load_clean


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {s: load_clean(s) for s in ("train", "test")}


@pytest.mark.parametrize("split", ["train", "test"])
def test_timestamps_unique_and_sorted(splits, split):
    ts = splits[split]["timestamp"]
    assert ts.is_monotonic_increasing
    assert not ts.duplicated().any()


@pytest.mark.parametrize("split", ["train", "test"])
def test_timestamps_are_tz_aware_ist(splits, split):
    assert str(splits[split]["timestamp"].dt.tz) == config.TIMEZONE


@pytest.mark.parametrize("split", ["train", "test"])
def test_ohlc_consistent_and_positive(splits, split):
    df = splits[split]
    o, h, l, c = (df[k] for k in config.OHLC_COLS)
    assert (h >= l).all()
    assert (h >= o).all() and (h >= c).all()
    assert (l <= o).all() and (l <= c).all()
    assert (df[config.OHLC_COLS] > 0).all().all()


@pytest.mark.parametrize("split", ["train", "test"])
def test_no_nulls(splits, split):
    assert splits[split].isna().sum().sum() == 0


@pytest.mark.parametrize("split", ["train", "test"])
def test_bars_within_regular_session(splits, split):
    tod = splits[split]["timestamp"].dt.strftime("%H:%M")
    assert (tod >= config.SESSION_START).all()
    assert (tod <= config.SESSION_END).all()
    # every bar sits on the 15-minute grid
    assert splits[split]["timestamp"].dt.minute.isin({0, 15, 30, 45}).all()


@pytest.mark.parametrize("split", ["train", "test"])
def test_session_flags_agree_with_session_count(splits, split):
    df = splits[split]
    n = df["session_date"].nunique()
    assert int(df["is_session_open"].sum()) == n
    assert int(df["is_session_close"].sum()) == n
    assert (df.groupby("session_date").size() <= config.BARS_PER_SESSION).all()


@pytest.mark.parametrize("split", ["train", "test"])
def test_bar_of_day_restarts_each_session(splits, split):
    df = splits[split]
    first = df.groupby("session_date")["bar_of_day"].first()
    assert (first == 0).all()
    assert df.groupby("session_date")["bar_of_day"].apply(
        lambda s: s.is_monotonic_increasing
    ).all()


def test_chronological_split_no_leakage(splits):
    assert_no_leakage(splits["train"], splits["test"])


def test_no_extreme_price_jumps(splits):
    """A >10% move on a single 15-min index bar would signal a corrupt print."""
    for df in splits.values():
        assert df["close"].pct_change().abs().max() < 0.10
