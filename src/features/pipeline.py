"""Assemble all five step-2 feature blocks into one matrix.

Order is irrelevant — the blocks are independent — but it is fixed so column
order is reproducible across runs.
"""

from __future__ import annotations

import pandas as pd

from src.features.adx import build_adx_features
from src.features.atr import VOL_WINDOW, build_atr_features
from src.features.base import WARMUP_BARS, attach_block
from src.features.bollinger import SQUEEZE_WINDOW, build_bollinger_features
from src.features.macd import build_macd_features
from src.features.rsi import build_rsi_features

BLOCKS = {
    "rsi": build_rsi_features,
    "macd": build_macd_features,
    "atr": build_atr_features,
    "adx": build_adx_features,
    "bollinger": build_bollinger_features,
}

# The trailing-percentile columns (ATR and Bollinger) need a full ranking window
# behind the first target bar, which is far more than the smoothing needs. One
# seed length is used for the whole matrix so every block sees identical history.
FULL_WARMUP_BARS: int = max(WARMUP_BARS, VOL_WINDOW, SQUEEZE_WINDOW) + WARMUP_BARS

# Columns denominated in index points. They drift with the price level — Nifty
# went from ~8,000 to ~22,600 across the training window — so they are kept for
# charting and validation but must NOT be fed to a model. Use the scale-free
# counterpart named alongside each.
RAW_SCALE_COLS: dict[str, str] = {
    "macd": "macd_pct",
    "macd_signal": "macd_signal_pct",
    "macd_hist": "macd_hist_pct",
    "true_range": "tr_over_atr",
    "atr_14": "atr_pct",
    "bb_mid": "bb_dist_mid_pct",
    "bb_upper": "bb_pct_b",
    "bb_lower": "bb_pct_b",
    "bb_std": "bb_bandwidth",
}

# Excluded for a different reason: already scale-free, but an intermediate the
# final indicator is computed from. Feeding both is redundant collinearity.
INTERMEDIATE_COLS: dict[str, str] = {
    "dx": "adx_14 (dx is the unsmoothed input to it)",
}


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every block over `df` and concatenate the results."""
    return pd.concat([fn(df) for fn in BLOCKS.values()], axis=1)


def attach_all_features(
    df: pd.DataFrame,
    seed: pd.DataFrame | None = None,
    warmup_bars: int = FULL_WARMUP_BARS,
) -> pd.DataFrame:
    """Attach the full feature matrix to `df`, optionally warm-started."""
    out = attach_block(df, build_all_features, seed=seed, warmup_bars=warmup_bars)

    # One matrix-level flag: a row is usable only if every block is past warm-up.
    warm_cols = [c for c in out.columns if c.endswith("_is_warmup")]
    out["is_warmup"] = out[warm_cols].any(axis=1)
    return out


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns that are safe to feed a model.

    Excludes OHLC, bookkeeping, the vendor's `src_*` columns, the warm-up flags,
    and every price-denominated column in `RAW_SCALE_COLS`.
    """
    drop = {
        "timestamp", "session_date", "bar_of_day",
        "is_session_open", "is_session_close",
        "open", "high", "low", "close",
        "dx",
    }
    return [
        c
        for c in df.columns
        if c not in drop
        and c not in RAW_SCALE_COLS
        and not c.startswith("src_")
        and not c.endswith("_is_warmup")
        and c != "is_warmup"
    ]
