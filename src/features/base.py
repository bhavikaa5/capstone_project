"""Shared machinery for every step-2 feature block.

All five indicator families (RSI, MACD, ATR, ADX, Bollinger) need the same two
things, so they live here once rather than five times:

* **Wilder's RMA** — RSI, ATR and ADX are all *defined* with it.
* **Warm-starting** — an indicator computed on the test split in isolation has
  undefined leading bars and a seed-biased head. Prepending the tail of the
  chronologically preceding split fixes both. This feeds past into future, which
  is the correct direction; the reverse would be leakage, and `attach_block`
  refuses to run if the seed does not end strictly before the target begins.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

# Bars of the preceding split prepended before computing, then discarded.
# Wilder's RMA and pandas' EMA both converge geometrically; 250 bars (10
# sessions) puts residual seed influence for the longest window used anywhere in
# step 2 (Bollinger 20, MACD 26, ADX 14+14) far below float noise.
WARMUP_BARS: int = 250


class FeatureBuilder(Protocol):
    """Takes an OHLC frame, returns a feature frame on the same index."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...


def wilder_rma(values: pd.Series, period: int) -> pd.Series:
    """Wilder's running moving average, seeded with a simple mean.

    Equivalent to `ewm(alpha=1/period, adjust=False)` *after* the seed — and the
    seed is the part a bare `ewm` gets wrong.
    """
    values = values.astype("float64")
    arr = values.to_numpy()
    out = np.full(len(arr), np.nan)

    # Inputs built from `.diff()` carry a leading NaN. Skip it so the seed is the
    # mean of the first `period` *real* observations.
    valid = np.flatnonzero(~np.isnan(arr))
    if valid.size < period:
        return pd.Series(out, index=values.index)

    start = valid[0]
    end = start + period                       # first index with a full window
    if end > len(arr):
        return pd.Series(out, index=values.index)

    prev = arr[start:end].mean()
    out[end - 1] = prev
    alpha = 1.0 / period
    for i in range(end, len(arr)):
        prev += alpha * (arr[i] - prev)
        out[i] = prev
    return pd.Series(out, index=values.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's true range: the widest of today's range and the two gap spans.

    Shared by ATR and ADX, which are both built on it.
    """
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where `fast` crosses above `slow`, -1 where it crosses below, else 0.

    A crossover is a *transition*, so it needs both the current and previous
    relationship — thresholding the current one alone would flag every bar of a
    trend, not the turn. Bars where either input is undefined yield 0, so the
    edge out of a warm-up region is not mistaken for a cross.
    """
    defined = fast.notna() & slow.notna()
    above = (fast > slow).where(defined).astype("float64")
    return above.diff().fillna(0.0).astype("int8")


def warmup_flag(feats: pd.DataFrame, cols: list[str], bars: int) -> pd.Series:
    """TRUE while a value is undefined or still seed-dominated.

    Builders call this assuming a cold start; `attach_block` relaxes it to the
    NaN test alone when the split was warm-started, since there is no warm-up
    region left to flag.
    """
    head = pd.Series(False, index=feats.index)
    head.iloc[:bars] = True
    return feats[cols].isna().any(axis=1) | head


def attach_block(
    df: pd.DataFrame,
    builder: FeatureBuilder,
    seed: pd.DataFrame | None = None,
    warmup_bars: int = WARMUP_BARS,
) -> pd.DataFrame:
    """Run `builder` over `df`, optionally warm-started from `seed`.

    `seed` is the chronologically preceding frame (the train split when `df` is
    test). Its last `warmup_bars` rows prime the smoothing and are then dropped,
    so the returned frame lines up with `df` row for row.

    Any `*_is_warmup` column the builder emits is recomputed after a warm start:
    the leading-bars term no longer applies, only the NaN test.
    """
    if seed is not None and len(seed):
        tail = seed.tail(warmup_bars)
        if tail["timestamp"].iloc[-1] >= df["timestamp"].iloc[0]:
            raise ValueError(
                "seed must end strictly before df begins — otherwise the "
                "warm-up window leaks future data into the past"
            )
        combined = pd.concat([tail, df], ignore_index=True)
        n_seed = len(tail)
    else:
        combined = df.reset_index(drop=True)
        n_seed = 0

    feats = builder(combined).iloc[n_seed:].reset_index(drop=True)

    if n_seed:
        value_cols = [c for c in feats.columns if not c.endswith("_is_warmup")]
        for flag in (c for c in feats.columns if c.endswith("_is_warmup")):
            feats[flag] = feats[value_cols].isna().any(axis=1)

    return pd.concat([df.reset_index(drop=True), feats], axis=1)
