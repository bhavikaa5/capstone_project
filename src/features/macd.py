"""Step 2 (block 2 of 5) — MACD.

Moving Average Convergence/Divergence: EMA(12) - EMA(26), its EMA(9) signal
line, and the histogram between them.

Design notes
------------
* **The raw MACD line is not usable as a model input.** It is measured in index
  points, and the Nifty 50 goes from ~8,000 to ~22,600 across the training
  window — the same *relative* momentum produces a MACD roughly 2.8x larger in
  2024 than in 2015. A model trained on the raw series would read that scale
  drift as signal. Every MACD column is therefore also published as a percentage
  of close (`*_pct`), and those are the columns to feed the networks.
* **`adjust=False`** on the EMAs. That is the recursive form every charting
  package uses; `adjust=True` (pandas' default) computes a different, fully
  weighted average that will not reconcile with TradingView.
* **Crossovers are transitions, not levels.** `macd_cross` fires only on the bar
  where the relationship flips, so it stays sparse; `macd_above_signal` carries
  the persistent state.
"""

from __future__ import annotations

import pandas as pd

from src.features.base import WARMUP_BARS, attach_block, crossover, warmup_flag

FAST: int = 12
SLOW: int = 26
SIGNAL: int = 9

__all__ = ["FAST", "SIGNAL", "SLOW", "attach_macd", "build_macd_features", "ema", "macd"]


def ema(values: pd.Series, span: int) -> pd.Series:
    """Recursive EMA — the `adjust=False` form used by charting packages."""
    return values.astype("float64").ewm(span=span, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = FAST,
    slow: int = SLOW,
    signal: int = SIGNAL,
) -> pd.DataFrame:
    """Return the MACD line, signal line and histogram."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame(
        {"macd": line, "macd_signal": sig, "macd_hist": line - sig},
        index=close.index,
    )


def build_macd_features(
    df: pd.DataFrame,
    fast: int = FAST,
    slow: int = SLOW,
    signal: int = SIGNAL,
) -> pd.DataFrame:
    """Return the MACD feature block, indexed to match `df`.

    Columns
    -------
    macd, macd_signal, macd_hist        Raw values, in index points.
    macd_pct, macd_signal_pct, macd_hist_pct
                                        The same three as a percentage of close
                                        — **use these as model inputs**, see the
                                        module docstring.
    macd_hist_chg                       One-bar change in the histogram; turns
                                        before the crossover does.
    macd_above_signal                   Persistent state: MACD over its signal.
    macd_cross                          +1 bullish cross, -1 bearish, 0 otherwise.
    macd_is_warmup                      True while the EMAs are seed-dominated.
    """
    close = df["close"]
    feats = macd(close, fast=fast, slow=slow, signal=signal)

    # Scale-free versions. Guard the divide even though an index close is never
    # zero — a silent inf would propagate into the model quietly.
    safe_close = close.where(close != 0)
    for col in ("macd", "macd_signal", "macd_hist"):
        feats[f"{col}_pct"] = feats[col] / safe_close * 100.0

    feats["macd_hist_chg"] = feats["macd_hist_pct"].diff()
    feats["macd_above_signal"] = feats["macd"] > feats["macd_signal"]
    feats["macd_cross"] = crossover(feats["macd"], feats["macd_signal"])

    # An EMA is defined from bar 0 but meaningless until it has forgotten its
    # seed. slow + signal spans, times the usual factor of 5.
    feats["macd_is_warmup"] = warmup_flag(
        feats, ["macd", "macd_signal"], bars=5 * (slow + signal)
    )
    return feats


def attach_macd(
    df: pd.DataFrame,
    seed: pd.DataFrame | None = None,
    fast: int = FAST,
    slow: int = SLOW,
    signal: int = SIGNAL,
    warmup_bars: int = WARMUP_BARS,
) -> pd.DataFrame:
    """Attach the MACD block to `df`, optionally warm-started from `seed`."""
    return attach_block(
        df,
        lambda d: build_macd_features(d, fast=fast, slow=slow, signal=signal),
        seed=seed,
        warmup_bars=warmup_bars,
    )
