"""Step 2 (block 3 of 5) — ATR.

Wilder's Average True Range: the volatility measure the HMM leans on hardest,
since a high-volatility regime is largely defined by it.

Design notes
------------
* **ATR must be normalised before a model sees it.** Like MACD, raw ATR is in
  index points, so it scales with price level — 40 points at Nifty 8,000 is the
  same volatility as 110 points at 22,000. `atr_pct` (ATR as a percentage of
  close) is the scale-free version and the one to feed the networks and the HMM.
* **True range, not high - low.** TR takes the widest of the bar's own range and
  the two gaps against the previous close, so an overnight gap counts as the
  volatility it actually is. On 15-minute index data the overnight bar is where
  most of the gap risk sits, so this matters more here than on daily bars.
* **Wilder's RMA**, shared with RSI and ADX — see `base.py`.
"""

from __future__ import annotations

import pandas as pd

from src.features.base import (
    WARMUP_BARS,
    attach_block,
    true_range,
    warmup_flag,
    wilder_rma,
)

PERIOD: int = 14
# Window for the volatility-regime percentile. 25 bars/session, so 1000 bars is
# ~40 sessions (two trading months) — long enough to be a stable reference,
# short enough to adapt within a regime.
VOL_WINDOW: int = 1000

__all__ = ["PERIOD", "VOL_WINDOW", "atr", "attach_atr", "build_atr_features"]


def atr(df: pd.DataFrame, period: int = PERIOD) -> pd.Series:
    """Wilder's ATR in index points. NaN until `period` true ranges exist."""
    return wilder_rma(true_range(df), period)


def build_atr_features(
    df: pd.DataFrame,
    period: int = PERIOD,
    vol_window: int = VOL_WINDOW,
) -> pd.DataFrame:
    """Return the ATR feature block, indexed to match `df`.

    Columns
    -------
    true_range      Per-bar true range, in points.
    atr_14          Wilder ATR, in points.
    atr_pct         ATR as a percentage of close — **the model input**.
    tr_over_atr     Current bar's TR relative to the running ATR; a spike
                    detector that is already scale-free.
    atr_chg_pct     One-bar percentage change in ATR: volatility *expanding* or
                    contracting, which the level alone does not say.
    atr_pctile_1000 Rank of atr_pct within a trailing 1000-bar window, 0-1. This
                    is what makes "high volatility" comparable across 2015 and
                    2024 without the HMM having to learn the drift itself.
    atr_is_warmup   True while the smoothing is still seed-dominated.
    """
    close = df["close"]
    feats = pd.DataFrame(index=df.index)

    feats["true_range"] = true_range(df)
    feats[f"atr_{period}"] = atr(df, period)

    safe_close = close.where(close != 0)
    feats["atr_pct"] = feats[f"atr_{period}"] / safe_close * 100.0

    safe_atr = feats[f"atr_{period}"].replace(0.0, pd.NA).astype("float64")
    feats["tr_over_atr"] = feats["true_range"] / safe_atr
    feats["atr_chg_pct"] = feats[f"atr_{period}"].pct_change() * 100.0

    # `rank(pct=True)` on the trailing window: the fraction of the last
    # `vol_window` bars whose atr_pct was at or below the current one.
    feats[f"atr_pctile_{vol_window}"] = (
        feats["atr_pct"]
        .rolling(vol_window, min_periods=vol_window // 4)
        .rank(pct=True)
    )

    # The percentile column outlives the smoothing warm-up by a wide margin —
    # its rolling window needs `vol_window // 4` bars before it returns anything
    # — so it has to be in the flag, or the matrix reports usable rows that
    # still carry NaNs.
    feats["atr_is_warmup"] = warmup_flag(
        feats,
        [f"atr_{period}", "atr_pct", f"atr_pctile_{vol_window}"],
        bars=5 * period,
    )
    return feats


def attach_atr(
    df: pd.DataFrame,
    seed: pd.DataFrame | None = None,
    period: int = PERIOD,
    vol_window: int = VOL_WINDOW,
    warmup_bars: int | None = None,
) -> pd.DataFrame:
    """Attach the ATR block to `df`, optionally warm-started from `seed`.

    This block sizes its own seed. The default 250-bar tail is enough for the
    Wilder smoothing but not for `atr_pctile_{vol_window}`, which needs a full
    `vol_window` of history behind the first target bar — otherwise the test
    split's opening percentiles are ranked against a much shorter window than
    the training split's were, and the two are not comparable.
    """
    if warmup_bars is None:
        warmup_bars = max(WARMUP_BARS, vol_window + 5 * period)
    return attach_block(
        df,
        lambda d: build_atr_features(d, period=period, vol_window=vol_window),
        seed=seed,
        warmup_bars=warmup_bars,
    )
