"""Step 2 (block 5 of 5) — Bollinger Bands.

A 20-bar simple moving average with bands at +/-2 standard deviations, and the
two derived quantities that are actually worth modelling: %B (where price sits
within the bands) and bandwidth (how wide they are).

Design notes
------------
* **Population standard deviation (ddof=0).** Bollinger's definition uses the
  population form. pandas' `.std()` defaults to `ddof=1`, which on a 20-bar
  window inflates the deviation by ~2.6% and widens every band accordingly —
  small, but it is a silent mismatch against every charting package.
* **Feed %B and bandwidth, not the bands.** The three band columns are in index
  points and drift with the price level exactly as MACD and ATR do. %B is
  unitless and bandwidth is a percentage, so both are comparable across 2015 and
  2024.
* **Bandwidth is the squeeze detector**, and it is the single most useful column
  here for regime work: a bandwidth percentile near zero is the classic
  low-volatility coil that precedes a breakout, and it gives the HMM a
  volatility signal that is constructed differently from ATR's.
"""

from __future__ import annotations

import pandas as pd

from src.features.base import WARMUP_BARS, attach_block, warmup_flag

PERIOD: int = 20
NUM_STD: float = 2.0
# Trailing window for the bandwidth percentile — ~40 sessions, matching ATR's.
SQUEEZE_WINDOW: int = 1000
# A bandwidth in the bottom decile of its trailing window is a "squeeze".
SQUEEZE_QUANTILE: float = 0.10

__all__ = [
    "NUM_STD",
    "PERIOD",
    "SQUEEZE_QUANTILE",
    "SQUEEZE_WINDOW",
    "attach_bollinger",
    "bollinger",
    "build_bollinger_features",
]


def bollinger(
    close: pd.Series,
    period: int = PERIOD,
    num_std: float = NUM_STD,
) -> pd.DataFrame:
    """Return the middle, upper and lower bands, in index points."""
    close = close.astype("float64")
    mid = close.rolling(period, min_periods=period).mean()
    # ddof=0: Bollinger's definition is the population standard deviation.
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + num_std * sd, "bb_lower": mid - num_std * sd,
         "bb_std": sd},
        index=close.index,
    )


def build_bollinger_features(
    df: pd.DataFrame,
    period: int = PERIOD,
    num_std: float = NUM_STD,
    squeeze_window: int = SQUEEZE_WINDOW,
    squeeze_quantile: float = SQUEEZE_QUANTILE,
) -> pd.DataFrame:
    """Return the Bollinger feature block, indexed to match `df`.

    Columns
    -------
    bb_mid, bb_upper, bb_lower, bb_std  Bands in index points (reference only —
                                        they drift with the price level).
    bb_pct_b        %B: 0 at the lower band, 1 at the upper, outside [0,1] on a
                    band break. **A model input.**
    bb_bandwidth    (upper - lower) / mid x 100 — band width as a percentage of
                    price. **A model input.**
    bb_dist_mid_pct Close vs the moving average, in percent: mean reversion.
    bb_bandwidth_pctile  Rank of bandwidth in a trailing 1000-bar window, 0-1.
    bb_squeeze      True when bandwidth is in the bottom decile of that window.
    bb_break        +1 close above the upper band, -1 below the lower, else 0.
    bb_is_warmup    True while the rolling window is not yet full.
    """
    close = df["close"]
    feats = bollinger(close, period=period, num_std=num_std)

    width = feats["bb_upper"] - feats["bb_lower"]
    # A zero-width band means 20 identical closes — %B is undefined, not 0.5.
    safe_width = width.replace(0.0, pd.NA).astype("float64")
    safe_mid = feats["bb_mid"].replace(0.0, pd.NA).astype("float64")

    feats["bb_pct_b"] = (close - feats["bb_lower"]) / safe_width
    feats["bb_bandwidth"] = width / safe_mid * 100.0
    feats["bb_dist_mid_pct"] = (close - feats["bb_mid"]) / safe_mid * 100.0

    feats["bb_bandwidth_pctile"] = (
        feats["bb_bandwidth"]
        .rolling(squeeze_window, min_periods=squeeze_window // 4)
        .rank(pct=True)
    )
    feats["bb_squeeze"] = feats["bb_bandwidth_pctile"].lt(squeeze_quantile)

    feats["bb_break"] = (
        close.gt(feats["bb_upper"]).astype("int8")
        - close.lt(feats["bb_lower"]).astype("int8")
    ).astype("int8")

    # A simple rolling window is exact once full — no seed to decay, so the
    # bands' own warm-up is just the window itself. The bandwidth percentile
    # takes far longer to become defined, so it has to be in the flag too.
    feats["bb_is_warmup"] = warmup_flag(
        feats,
        ["bb_mid", "bb_pct_b", "bb_bandwidth", "bb_bandwidth_pctile"],
        bars=period,
    )
    return feats


def attach_bollinger(
    df: pd.DataFrame,
    seed: pd.DataFrame | None = None,
    period: int = PERIOD,
    num_std: float = NUM_STD,
    squeeze_window: int = SQUEEZE_WINDOW,
    squeeze_quantile: float = SQUEEZE_QUANTILE,
    warmup_bars: int | None = None,
) -> pd.DataFrame:
    """Attach the Bollinger block to `df`, optionally warm-started from `seed`.

    Like ATR, this block sizes its own seed: the bandwidth percentile needs a
    full `squeeze_window` of history behind the first target bar, or the test
    split's opening squeeze flags are ranked against a shorter window than the
    training split's were.
    """
    if warmup_bars is None:
        warmup_bars = max(WARMUP_BARS, squeeze_window + period)
    return attach_block(
        df,
        lambda d: build_bollinger_features(
            d,
            period=period,
            num_std=num_std,
            squeeze_window=squeeze_window,
            squeeze_quantile=squeeze_quantile,
        ),
        seed=seed,
        warmup_bars=warmup_bars,
    )
