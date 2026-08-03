"""Correctness of the MACD, ATR, ADX and Bollinger feature blocks.

RSI has its own module (`test_rsi.py`); it was built first and carries the
Wilder reference-example test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_clean
from src.features import adx as adx_mod
from src.features import atr as atr_mod
from src.features import bollinger as bb_mod
from src.features import macd as macd_mod
from src.features.base import crossover, true_range


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return load_clean("train")


@pytest.fixture(scope="module")
def test_split() -> pd.DataFrame:
    return load_clean("test")


@pytest.fixture(scope="module")
def blocks(train, test_split) -> dict:
    """Every block attached to the test split, warm-started from train."""
    return {
        "macd": macd_mod.attach_macd(test_split, seed=train),
        "atr": atr_mod.attach_atr(test_split, seed=train),
        "adx": adx_mod.attach_adx(test_split, seed=train),
        "bb": bb_mod.attach_bollinger(test_split, seed=train),
    }


# --------------------------------------------------------------------------- #
# base helpers
# --------------------------------------------------------------------------- #
def test_true_range_is_the_widest_of_three_spans(train):
    tr = true_range(train)
    i = 500
    row, prev_close = train.iloc[i], train["close"].iloc[i - 1]
    expected = max(row.high - row.low,
                   abs(row.high - prev_close),
                   abs(row.low - prev_close))
    assert tr.iloc[i] == pytest.approx(expected)
    assert (tr.dropna() >= 0).all()


def test_true_range_first_bar_falls_back_to_range(train):
    """No previous close on bar 0, so TR is simply high - low."""
    tr = true_range(train)
    assert tr.iloc[0] == pytest.approx(train.high.iloc[0] - train.low.iloc[0])


def test_crossover_flags_transitions_only():
    fast = pd.Series([np.nan, np.nan, 1.0, 3.0, 3.0, 1.0, 1.0, 5.0])
    slow = pd.Series([np.nan, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert crossover(fast, slow).tolist() == [0, 0, 0, 1, 0, -1, 0, 1]


# --------------------------------------------------------------------------- #
# MACD
# --------------------------------------------------------------------------- #
def test_macd_identity(train):
    m = macd_mod.macd(train["close"])
    np.testing.assert_allclose(m["macd_hist"], m["macd"] - m["macd_signal"], atol=1e-12)


def test_macd_equals_difference_of_emas(train):
    close = train["close"]
    expected = macd_mod.ema(close, 12) - macd_mod.ema(close, 26)
    np.testing.assert_allclose(macd_mod.macd(close)["macd"], expected, atol=1e-12)


def test_macd_pct_is_scale_free(train):
    """Doubling every price must double raw MACD but leave macd_pct unchanged."""
    doubled = train.copy()
    doubled[["open", "high", "low", "close"]] *= 2.0
    a = macd_mod.build_macd_features(train)
    b = macd_mod.build_macd_features(doubled)
    np.testing.assert_allclose(b["macd"], a["macd"] * 2.0, atol=1e-9)
    np.testing.assert_allclose(b["macd_pct"], a["macd_pct"], atol=1e-12)


def test_macd_cross_agrees_with_state(blocks):
    f = blocks["macd"]
    flips = f.index[f["macd_cross"] != 0]
    for i in flips[:50]:
        if i == 0:
            continue
        assert f["macd_above_signal"].iloc[i] != f["macd_above_signal"].iloc[i - 1]
    assert f["macd_cross"].eq(1).sum() > 0 and f["macd_cross"].eq(-1).sum() > 0


# --------------------------------------------------------------------------- #
# ATR
# --------------------------------------------------------------------------- #
def test_atr_first_value_at_period_minus_one(train):
    """TR is defined on bar 0, so 14 true ranges exist by index 13.

    This differs from RSI, whose first *change* only appears at index 1.
    """
    assert atr_mod.atr(train, 14).first_valid_index() == 13


def test_atr_is_positive_and_bounded_by_max_tr(train):
    a = atr_mod.atr(train, 14).dropna()
    assert (a > 0).all()
    assert a.max() <= true_range(train).max()


def test_atr_pct_is_scale_free(train):
    doubled = train.copy()
    doubled[["open", "high", "low", "close"]] *= 2.0
    a = atr_mod.build_atr_features(train)
    b = atr_mod.build_atr_features(doubled)
    np.testing.assert_allclose(b["atr_14"], a["atr_14"] * 2.0, atol=1e-9)
    np.testing.assert_allclose(b["atr_pct"].dropna(), a["atr_pct"].dropna(), atol=1e-12)


def test_atr_percentile_is_a_probability(blocks):
    p = blocks["atr"]["atr_pctile_1000"].dropna()
    assert p.between(0.0, 1.0).all()


def test_atr_ranks_the_covid_crash_above_a_calm_month(train):
    """A volatility measure that cannot separate these is not measuring volatility."""
    f = atr_mod.build_atr_features(train)
    month = train["timestamp"].dt.strftime("%Y-%m").to_numpy()
    crash = f["atr_pct"][month == "2020-03"].mean()
    calm = f["atr_pct"][month == "2017-06"].mean()
    assert crash > 3 * calm


# --------------------------------------------------------------------------- #
# ADX
# --------------------------------------------------------------------------- #
def test_directional_movement_is_mutually_exclusive(train):
    plus, minus = adx_mod.directional_movement(train)
    assert not ((plus > 0) & (minus > 0)).any()
    assert (plus.dropna() >= 0).all() and (minus.dropna() >= 0).all()


def test_adx_components_are_bounded(train):
    a = adx_mod.adx(train, 14)
    for col in ("di_plus", "di_minus", "dx", "adx_14"):
        s = a[col].dropna()
        assert s.min() >= 0.0 and s.max() <= 100.0


def test_adx_needs_double_the_period_to_start(train):
    """DI takes `period` bars, then ADX averages `period` DX values on top."""
    a = adx_mod.adx(train, 14)
    assert a["di_plus"].first_valid_index() == 14
    assert a["adx_14"].first_valid_index() == 27


def test_adx_is_already_scale_free(train):
    doubled = train.copy()
    doubled[["open", "high", "low", "close"]] *= 2.0
    a = adx_mod.adx(train, 14)
    b = adx_mod.adx(doubled, 14)
    for col in ("di_plus", "di_minus", "adx_14"):
        np.testing.assert_allclose(b[col].dropna(), a[col].dropna(), atol=1e-9)


def test_di_spread_carries_direction_that_adx_does_not(train):
    """ADX is unsigned: the crash and the rally must differ in di_spread only."""
    f = adx_mod.build_adx_features(train)
    month = train["timestamp"].dt.strftime("%Y-%m").to_numpy()
    assert f["di_spread"][month == "2020-03"].mean() < 0     # crash
    assert f["di_spread"][month == "2020-11"].mean() > 0     # rally


def test_adx_regime_encoding(blocks):
    f = blocks["adx"]
    assert set(f["adx_regime"].unique()) <= {-1, 0, 1}
    # ranging bars must be coded 0 regardless of direction
    assert (f.loc[~f["adx_trending"], "adx_regime"] == 0).all()
    trending_up = f["adx_trending"] & (f["di_spread"] > 0)
    assert (f.loc[trending_up, "adx_regime"] == 1).all()


# --------------------------------------------------------------------------- #
# Bollinger
# --------------------------------------------------------------------------- #
def test_bollinger_uses_population_std(train):
    """ddof=1 would widen every band by ~2.6% — a silent mismatch with charts."""
    close = train["close"].head(5000)
    bands = bb_mod.bollinger(close, 20, 2.0)
    expected_sd = close.rolling(20, min_periods=20).std(ddof=0)
    np.testing.assert_allclose(bands["bb_std"].dropna(), expected_sd.dropna(), atol=1e-12)
    wrong = close.rolling(20, min_periods=20).std(ddof=1)
    assert not np.allclose(bands["bb_std"].dropna(), wrong.dropna())


def test_bollinger_band_ordering(train):
    b = bb_mod.bollinger(train["close"], 20, 2.0).dropna()
    assert (b["bb_upper"] >= b["bb_mid"]).all()
    assert (b["bb_mid"] >= b["bb_lower"]).all()


def test_pct_b_endpoints():
    """%B is 0 at the lower band and 1 at the upper, by construction."""
    close = pd.Series(np.random.default_rng(1).normal(100, 2, 500))
    f = bb_mod.build_bollinger_features(pd.DataFrame({"close": close}))
    at_upper = (close - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"])
    np.testing.assert_allclose(f["bb_pct_b"].dropna(), at_upper.dropna(), atol=1e-12)
    assert f["bb_pct_b"].dropna().between(-2, 3).all()


def test_bb_break_agrees_with_pct_b(blocks):
    f = blocks["bb"]
    assert (f.loc[f["bb_break"] == 1, "bb_pct_b"] > 1.0).all()
    assert (f.loc[f["bb_break"] == -1, "bb_pct_b"] < 0.0).all()


def test_bandwidth_is_scale_free(train):
    doubled = train.copy()
    doubled[["open", "high", "low", "close"]] *= 2.0
    a = bb_mod.build_bollinger_features(train)
    b = bb_mod.build_bollinger_features(doubled)
    np.testing.assert_allclose(b["bb_bandwidth"].dropna(),
                               a["bb_bandwidth"].dropna(), atol=1e-12)
    np.testing.assert_allclose(b["bb_pct_b"].dropna(),
                               a["bb_pct_b"].dropna(), atol=1e-12)


def test_squeeze_fires_near_its_target_rate(blocks):
    """Bottom-decile threshold, so materially more than ~10% means a bug."""
    rate = blocks["bb"]["bb_squeeze"].mean()
    assert 0.03 <= rate <= 0.20


# --------------------------------------------------------------------------- #
# cross-block guarantees
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["macd", "atr", "adx", "bb"])
def test_block_preserves_row_alignment(blocks, test_split, name):
    out = blocks[name]
    assert len(out) == len(test_split)
    pd.testing.assert_series_equal(
        out["timestamp"], test_split.reset_index(drop=True)["timestamp"]
    )


@pytest.mark.parametrize("name", ["macd", "atr", "adx", "bb"])
def test_warm_started_split_has_no_nans(blocks, name):
    out = blocks[name]
    added = [c for c in out.columns if c not in load_clean("test").columns]
    assert out[added].isna().sum().sum() == 0


@pytest.mark.parametrize(
    "attach",
    [macd_mod.attach_macd, atr_mod.attach_atr,
     adx_mod.attach_adx, bb_mod.attach_bollinger],
)
def test_seed_must_precede_target(attach, test_split):
    with pytest.raises(ValueError, match="strictly before"):
        attach(test_split, seed=test_split)


@pytest.mark.parametrize(
    "attach,col",
    [
        (macd_mod.attach_macd, "macd"),
        (atr_mod.attach_atr, "atr_14"),
        (adx_mod.attach_adx, "adx_14"),
        (bb_mod.attach_bollinger, "bb_mid"),
    ],
)
def test_warm_start_converges_to_the_cold_result(attach, col, train, test_split):
    """Late bars must not remember how the smoothing was seeded."""
    seeded = attach(test_split, seed=train)[col]
    cold = attach(test_split)[col]
    np.testing.assert_allclose(seeded[2000:], cold[2000:], atol=1e-6)


@pytest.mark.parametrize(
    "builder,col",
    [
        (macd_mod.build_macd_features, "macd"),
        (atr_mod.build_atr_features, "atr_14"),
        (adx_mod.build_adx_features, "adx_14"),
        (bb_mod.build_bollinger_features, "bb_mid"),
    ],
)
def test_no_lookahead(builder, col, test_split):
    """Perturbing a future bar must leave every earlier value untouched."""
    df = test_split.head(2000).copy()
    base = builder(df)[col]
    df.loc[1500, ["open", "high", "low", "close"]] *= 1.05
    bumped = builder(df)[col]
    pd.testing.assert_series_equal(base.iloc[:1500], bumped.iloc[:1500])
    assert not np.allclose(base.iloc[1500:], bumped.iloc[1500:], equal_nan=True)
