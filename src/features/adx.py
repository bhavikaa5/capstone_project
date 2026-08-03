"""Step 2 (block 4 of 5) — ADX / DMI.

Wilder's Average Directional Index with its two directional indicators. This is
the block that tells the HMM *sideways* from *trending*, which no amount of RSI
or MACD does on its own: both of those measure direction, ADX measures whether
direction is worth anything.

Design notes
------------
* **ADX is already scale-free** (0-100), unlike MACD and ATR. No normalisation
  needed, and none is applied.
* **ADX has no sign.** A strong downtrend and a strong uptrend both read ~40.
  Direction lives in `di_spread` (+DI - -DI), so the pair must be fed together;
  ADX alone would tell a model the 2020 crash and the 2021 rally look identical.
* **Double smoothing means a long warm-up.** DI needs `period` bars and ADX
  averages `period` DX values on top, so the first defined ADX lands at
  ~2 x period and the seed influence persists well beyond that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import (
    WARMUP_BARS,
    attach_block,
    true_range,
    warmup_flag,
    wilder_rma,
)

PERIOD: int = 14
# Wilder's threshold: below 20-25 the market is ranging and trend signals from
# MACD/RSI are noise. 25 is the stricter of the two conventions.
TREND_THRESHOLD: float = 25.0

__all__ = [
    "PERIOD",
    "TREND_THRESHOLD",
    "adx",
    "attach_adx",
    "build_adx_features",
    "directional_movement",
]


def directional_movement(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Wilder's +DM and -DM.

    Only the *larger* of the two moves counts on any given bar; if the high
    extends further than the low, it is a +DM bar and -DM is zero, and vice
    versa. An inside bar produces neither.
    """
    up = df["high"].diff()               # high - prev_high
    down = -df["low"].diff()             # prev_low - low

    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)

    # Preserve the leading NaN: bar 0 has no previous bar, so DM is undefined
    # rather than zero — treating it as zero would bias the first average down.
    undefined = up.isna() | down.isna()
    plus = pd.Series(plus, index=df.index).where(~undefined)
    minus = pd.Series(minus, index=df.index).where(~undefined)
    return plus, minus


def adx(df: pd.DataFrame, period: int = PERIOD) -> pd.DataFrame:
    """Return +DI, -DI, DX and ADX, all on 0-100."""
    plus_dm, minus_dm = directional_movement(df)

    atr_ = wilder_rma(true_range(df), period)
    safe_atr = atr_.replace(0.0, np.nan)

    di_plus = 100.0 * wilder_rma(plus_dm, period) / safe_atr
    di_minus = 100.0 * wilder_rma(minus_dm, period) / safe_atr

    di_sum = di_plus + di_minus
    # di_sum == 0 means neither side moved: direction is undefined, so DX is 0
    # (no directional conviction) rather than 0/0.
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum.replace(0.0, np.nan)
    dx = dx.where(di_sum != 0, 0.0).where(di_plus.notna())

    return pd.DataFrame(
        {"di_plus": di_plus, "di_minus": di_minus, "dx": dx,
         f"adx_{period}": wilder_rma(dx, period)},
        index=df.index,
    )


def build_adx_features(
    df: pd.DataFrame,
    period: int = PERIOD,
    threshold: float = TREND_THRESHOLD,
) -> pd.DataFrame:
    """Return the ADX feature block, indexed to match `df`.

    Columns
    -------
    di_plus, di_minus  Directional indicators, 0-100.
    dx                 Raw directional index, before the final smoothing.
    adx_14             Wilder ADX — trend *strength*, unsigned.
    di_spread          +DI - -DI: trend *direction*, and the reason ADX is never
                       fed to a model on its own.
    adx_norm           adx_14 / 100, scaled to [0, 1] for the networks.
    adx_chg            One-bar change: is the trend strengthening or decaying?
    adx_trending       True when adx_14 > 25 (Wilder's ranging/trending line).
    adx_regime         +1 trending up, -1 trending down, 0 ranging. A compact
                       prior for the HMM's four-state labelling.
    adx_is_warmup      True while the double smoothing is seed-dominated.
    """
    feats = adx(df, period)
    key = f"adx_{period}"

    feats["di_spread"] = feats["di_plus"] - feats["di_minus"]
    feats["adx_norm"] = feats[key] / 100.0
    feats["adx_chg"] = feats[key].diff()

    trending = feats[key].gt(threshold)
    feats["adx_trending"] = trending
    feats["adx_regime"] = (
        np.sign(feats["di_spread"]).fillna(0).astype("int8") * trending.astype("int8")
    ).astype("int8")

    # DI needs `period` bars, ADX averages `period` DX values on top: 2 x period
    # to first value, times the usual factor of 5 for the seed to wash out.
    feats["adx_is_warmup"] = warmup_flag(
        feats, [key, "di_plus", "di_minus"], bars=5 * 2 * period
    )
    return feats


def attach_adx(
    df: pd.DataFrame,
    seed: pd.DataFrame | None = None,
    period: int = PERIOD,
    threshold: float = TREND_THRESHOLD,
    warmup_bars: int = WARMUP_BARS,
) -> pd.DataFrame:
    """Attach the ADX block to `df`, optionally warm-started from `seed`."""
    return attach_block(
        df,
        lambda d: build_adx_features(d, period=period, threshold=threshold),
        seed=seed,
        warmup_bars=warmup_bars,
    )
