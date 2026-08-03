"""Step 2 (block 1 of 5) — RSI.

Wilder's Relative Strength Index and the derived features the sequence models
consume. Shared smoothing and warm-start machinery lives in `base.py`.

Design notes
------------
* **Wilder smoothing, not a simple `ewm`.** RSI is defined with Wilder's RMA
  (alpha = 1/period), seeded by the simple mean of the first `period` changes.
  `ewm(alpha=1/period)` alone adapts faster and will not match TradingView or
  the vendor's own indicator columns.
* **Computed on the continuous series, not per session.** Overnight gaps are
  genuine price moves for an index, and resetting RSI every morning would throw
  away 14 bars of state each day (56% of a 25-bar session). The session flags
  from step 1 stay available for any feature that *does* need to reset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import WARMUP_BARS, attach_block, warmup_flag, wilder_rma

# Periods carried forward. 14 is the canonical Wilder setting and the one the
# report should quote; 7 and 21 give the model a short/long momentum contrast.
RSI_PERIODS: tuple[int, ...] = (7, 14, 21)
PRIMARY_PERIOD: int = 14

OVERBOUGHT: float = 70.0
OVERSOLD: float = 30.0

__all__ = [
    "OVERBOUGHT",
    "OVERSOLD",
    "PRIMARY_PERIOD",
    "RSI_PERIODS",
    "attach_rsi",
    "build_rsi_features",
    "rsi",
    "wilder_rma",
]


def rsi(close: pd.Series, period: int = PRIMARY_PERIOD) -> pd.Series:
    """Wilder's RSI on `close`, returned on 0-100.

    The first `period` values are NaN by construction — RSI is undefined until
    `period` price changes exist.
    """
    delta = close.astype("float64").diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)

    # avg_loss == 0 means an unbroken run of up-bars: RS is infinite, RSI is 100.
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~(avg_gain.eq(0) & avg_loss.eq(0)), 50.0)  # flat market
    return out.where(avg_gain.notna())


def build_rsi_features(
    df: pd.DataFrame,
    periods: tuple[int, ...] = RSI_PERIODS,
    primary: int = PRIMARY_PERIOD,
) -> pd.DataFrame:
    """Return the RSI feature block, indexed to match `df`.

    Columns
    -------
    rsi_{p}            Wilder RSI, 0-100, for each period.
    rsi_{primary}_norm RSI centred and scaled to roughly [-1, 1] — models train
                       better on a zero-centred input than on 0-100.
    rsi_{primary}_chg  One-bar change; the *direction* of RSI carries momentum
                       information the level alone does not.
    rsi_{primary}_slope_5  5-bar slope, a smoothed version of the above.
    rsi_spread_7_21    Short minus long RSI: a fast momentum-divergence proxy.
    rsi_{primary}_overbought / _oversold  Zone flags at 70 / 30.
    rsi_{primary}_zone   -1 oversold, 0 neutral, +1 overbought.
    rsi_is_warmup      True while the smoothing is still seed-dominated.
    """
    close = df["close"]
    feats = pd.DataFrame(index=df.index)

    for p in periods:
        feats[f"rsi_{p}"] = rsi(close, p)

    key = f"rsi_{primary}"
    feats[f"{key}_norm"] = (feats[key] - 50.0) / 50.0
    feats[f"{key}_chg"] = feats[key].diff()
    feats[f"{key}_slope_5"] = (feats[key] - feats[key].shift(5)) / 5.0

    lo, hi = min(periods), max(periods)
    feats[f"rsi_spread_{lo}_{hi}"] = feats[f"rsi_{lo}"] - feats[f"rsi_{hi}"]

    feats[f"{key}_overbought"] = feats[key].gt(OVERBOUGHT)
    feats[f"{key}_oversold"] = feats[key].lt(OVERSOLD)
    feats[f"{key}_zone"] = (
        feats[f"{key}_overbought"].astype("int8") - feats[f"{key}_oversold"].astype("int8")
    )

    feats["rsi_is_warmup"] = warmup_flag(
        feats, [f"rsi_{p}" for p in periods], bars=5 * max(periods)
    )
    return feats


def attach_rsi(
    df: pd.DataFrame,
    seed: pd.DataFrame | None = None,
    periods: tuple[int, ...] = RSI_PERIODS,
    primary: int = PRIMARY_PERIOD,
    warmup_bars: int = WARMUP_BARS,
) -> pd.DataFrame:
    """Attach the RSI block to `df`, optionally warm-started from `seed`."""
    return attach_block(
        df,
        lambda d: build_rsi_features(d, periods=periods, primary=primary),
        seed=seed,
        warmup_bars=warmup_bars,
    )
