"""Correctness of the RSI feature block."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_clean
from src.features.rsi import (
    OVERBOUGHT,
    OVERSOLD,
    PRIMARY_PERIOD,
    RSI_PERIODS,
    attach_rsi,
    rsi,
    wilder_rma,
)

# Wilder, "New Concepts in Technical Trading Systems", the RSI worked example.
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
    43.42, 42.66, 43.13,
]
WILDER_EXPECTED = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26,
                   56.06, 62.38, 54.71, 50.42, 39.99, 41.46, 41.87, 45.46,
                   37.30, 33.08, 37.77]


def test_matches_wilders_published_example():
    """The published table rounds intermediates, so 0.1 is the achievable bound."""
    got = rsi(pd.Series(WILDER_CLOSES), 14).dropna().to_numpy()
    assert len(got) == len(WILDER_EXPECTED)
    np.testing.assert_allclose(got, WILDER_EXPECTED, atol=0.1)


def test_first_value_lands_at_period_index():
    """RSI(n) is undefined until n price *changes* exist — index n, not n-1."""
    r = rsi(pd.Series(np.random.default_rng(0).normal(100, 1, 200)), PRIMARY_PERIOD)
    assert r.first_valid_index() == PRIMARY_PERIOD
    assert r.iloc[:PRIMARY_PERIOD].isna().all()


def test_wilder_rma_seed_is_simple_mean():
    s = pd.Series([np.nan] + [2.0] * 20)
    out = wilder_rma(s, 5)
    assert out.first_valid_index() == 5      # leading NaN skipped
    assert out.iloc[5] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "series,expected",
    [
        (np.arange(100, 200, dtype=float), 100.0),        # unbroken advance
        (np.arange(200, 100, -1, dtype=float), 0.0),      # unbroken decline
        (np.full(100, 150.0), 50.0),                      # perfectly flat
    ],
)
def test_degenerate_series(series, expected):
    assert rsi(pd.Series(series), 14).iloc[-1] == pytest.approx(expected)


def test_rsi_is_bounded():
    r = rsi(load_clean("train")["close"], PRIMARY_PERIOD).dropna()
    assert r.min() >= 0.0 and r.max() <= 100.0


def test_rsi_is_scale_invariant():
    """RSI depends on relative moves, so doubling every price must not change it."""
    close = load_clean("test")["close"].head(2000)
    a = rsi(close, PRIMARY_PERIOD)
    b = rsi(close * 2.0, PRIMARY_PERIOD)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("split", ["train", "test"])
def test_attach_preserves_row_alignment(split):
    df = load_clean(split)
    out = attach_rsi(df, seed=load_clean("train") if split == "test" else None)
    assert len(out) == len(df)
    pd.testing.assert_series_equal(
        out["timestamp"], df.reset_index(drop=True)["timestamp"]
    )
    for p in RSI_PERIODS:
        assert f"rsi_{p}" in out.columns


def test_test_split_is_warm_started_with_no_nans():
    """Seeding from the train tail must leave the test split fully defined."""
    out = attach_rsi(load_clean("test"), seed=load_clean("train"))
    for p in RSI_PERIODS:
        assert out[f"rsi_{p}"].isna().sum() == 0
    assert not out["rsi_is_warmup"].any()


def test_warm_start_uses_only_past_data():
    """Seeding must be rejected if the seed overlaps the target period."""
    test = load_clean("test")
    with pytest.raises(ValueError, match="strictly before"):
        attach_rsi(test, seed=test)


def test_warm_start_converges_to_the_unseeded_result():
    """Late bars must not care how the smoothing was seeded."""
    test = load_clean("test")
    seeded = attach_rsi(test, seed=load_clean("train"))["rsi_14"]
    cold = attach_rsi(test)["rsi_14"]
    tail = slice(1000, None)
    np.testing.assert_allclose(seeded[tail], cold[tail], atol=1e-6)


def test_derived_columns_are_consistent():
    out = attach_rsi(load_clean("test"), seed=load_clean("train"))
    key = f"rsi_{PRIMARY_PERIOD}"
    np.testing.assert_allclose(out[f"{key}_norm"], (out[key] - 50.0) / 50.0)
    assert out[f"{key}_overbought"].equals(out[key] > OVERBOUGHT)
    assert out[f"{key}_oversold"].equals(out[key] < OVERSOLD)
    assert set(out[f"{key}_zone"].unique()) <= {-1, 0, 1}
    # overbought and oversold are mutually exclusive by construction
    assert not (out[f"{key}_overbought"] & out[f"{key}_oversold"]).any()


def test_shorter_period_is_more_volatile():
    """RSI(7) must swing wider than RSI(21) — otherwise the periods are swapped."""
    out = attach_rsi(load_clean("test"), seed=load_clean("train"))
    assert out["rsi_7"].std() > out["rsi_14"].std() > out["rsi_21"].std()


def test_no_lookahead_in_rsi():
    """Perturbing a future bar must not change any earlier RSI value."""
    close = load_clean("test")["close"].head(500).copy()
    base = rsi(close, PRIMARY_PERIOD)
    close.iloc[400] *= 1.05
    bumped = rsi(close, PRIMARY_PERIOD)
    pd.testing.assert_series_equal(base.iloc[:400], bumped.iloc[:400])
    assert not np.allclose(base.iloc[400:], bumped.iloc[400:], equal_nan=True)
