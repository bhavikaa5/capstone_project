"""Step 4 (part 1) — turn the labelled feature matrix into supervised sequences.

Everything leakage-related lives here: what the target is, which columns the
model may see, where the train/validation boundary sits, and how the scaler is
fit. The architectures in `architectures.py` receive tensors and know nothing
about time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.pipeline import model_feature_columns

LOOKBACK: int = 32          # bars of history per sample (~1.3 sessions)
HORIZON: int = 1            # predict the direction of the next bar
VAL_FRACTION: float = 0.15  # last 15% of train, chronologically

# Past returns. Legitimate inputs — both are known at bar t — and `ret_1` is the
# single most informative column given the 0.18 lag-1 autocorrelation.
RETURN_COLS: list[str] = ["ret_1", "ret_20"]

# Regime information from step 3. The posteriors are causal (forward pass only),
# so they are safe to condition on; see docs/14.
REGIME_COLS: list[str] = [
    "regime_p_bull", "regime_p_bear", "regime_p_sideways", "regime_p_high_vol",
    "regime_confidence", "regime_changepoint", "regime_entropy",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The exact model input list.

    `model_feature_columns()` is defined against the step-2 frame; run against
    the step-3 frame it also sweeps up the return and regime columns, which
    would then be counted twice. The technical block is therefore taken from a
    step-2-shaped view and the rest appended explicitly.
    """
    step2_like = df.drop(columns=[c for c in RETURN_COLS + REGIME_COLS
                                  if c in df.columns], errors="ignore")
    technical = [c for c in model_feature_columns(step2_like)
                 if not c.startswith("regime")]
    cols = technical + RETURN_COLS + REGIME_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return cols


def make_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """1 if close rises over the next `horizon` bars, else 0.

    Uses log close so the comparison is exactly the sign of the forward return.
    The last `horizon` rows have no label and come back as NaN.
    """
    log_close = np.log(df["close"].astype("float64"))
    forward = log_close.shift(-horizon) - log_close
    return (forward > 0).astype("float32").where(forward.notna())


@dataclass
class SequenceData:
    """Windowed tensors plus the row indices they came from."""

    X: np.ndarray               # (n, lookback, n_features)
    y: np.ndarray               # (n,)
    row_index: np.ndarray       # index in the source frame of each window's LAST bar
    columns: list[str]

    def __len__(self) -> int:
        return len(self.y)


def build_sequences(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> SequenceData:
    """Slice `df` into (lookback x features) windows with a forward-direction label.

    A window ending at bar t contains bars t-lookback+1 .. t and is labelled with
    the direction from t to t+horizon. Nothing after t enters the window, which
    is the property the whole step depends on.

    Rows that step 2 or step 3 could not score are dropped *before* windowing,
    and windows are only emitted where the entire span is contiguous in the
    filtered frame — otherwise a window could silently straddle a dropped gap.
    """
    columns = columns or feature_columns(df)

    work = df.reset_index(drop=True).copy()
    work["_target"] = make_target(work, horizon)

    usable = work[columns].notna().all(axis=1) & work["_target"].notna()
    if "is_warmup" in work.columns:
        usable &= ~work["is_warmup"].astype(bool)
    if "regime" in work.columns:
        usable &= work["regime"].notna()

    positions = np.flatnonzero(usable.to_numpy())
    if len(positions) <= lookback:
        raise ValueError("not enough usable rows to build a single window")

    values = work.loc[:, columns].to_numpy(dtype="float32")
    target = work["_target"].to_numpy(dtype="float32")

    # Emit a window only when the previous `lookback-1` usable rows are also the
    # immediately preceding rows in the frame.
    contiguous = np.diff(positions) == 1
    run = np.zeros(len(positions), dtype=int)
    for i in range(1, len(positions)):
        run[i] = run[i - 1] + 1 if contiguous[i - 1] else 0
    ends = positions[run >= lookback - 1]

    idx = ends[:, None] - np.arange(lookback - 1, -1, -1)[None, :]
    return SequenceData(
        X=values[idx],
        y=target[ends],
        row_index=ends,
        columns=list(columns),
    )


def chronological_split(
    data: SequenceData,
    val_fraction: float = VAL_FRACTION,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[SequenceData, SequenceData]:
    """Split into train / validation by time, with a purge gap between them.

    The gap is `lookback + horizon` windows wide. Without it the last training
    windows overlap the first validation windows — they share input bars, and a
    training label can reach forward into the validation period. That is the
    standard way a walk-forward evaluation quietly inflates itself.
    """
    n = len(data)
    n_val = int(n * val_fraction)
    gap = lookback + horizon

    val_start = n - n_val
    train_end = val_start - gap
    if train_end <= 0:
        raise ValueError("validation fraction leaves no training data after purging")

    def _slice(lo: int, hi: int) -> SequenceData:
        return SequenceData(
            X=data.X[lo:hi], y=data.y[lo:hi],
            row_index=data.row_index[lo:hi], columns=data.columns,
        )

    return _slice(0, train_end), _slice(val_start, n)


@dataclass
class Scaler:
    """Per-feature standardisation, fit on the training windows only."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Scaler":
        # Accumulate in float64. Summing 1.5M float32 values leaves a residual
        # mean of ~0.02 std on the wider-ranged features, which is harmless for
        # training but makes the statistics not quite reproducible.
        flat = X.reshape(-1, X.shape[-1]).astype("float64")
        self.mean_ = flat.mean(axis=0)
        # A constant feature would divide by zero; leave it at its raw value.
        std = flat.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler.fit must be called before transform")
        return ((X - self.mean_) / self.std_).astype("float32")


def baseline_scores(df: pd.DataFrame, data: SequenceData, horizon: int = HORIZON) -> dict:
    """Accuracy of the two baselines any model has to beat.

    Reported on exactly the windows the models are scored on, so the comparison
    is like for like.

    - **majority**: always predict the more common class.
    - **persistence**: predict the direction of the most recent bar. On 15-minute
      index data this is a strong baseline — lag-1 return autocorrelation is
      ~0.18 — and a network that cannot beat it has learned nothing.
    """
    work = df.reset_index(drop=True)
    prev_ret = work["ret_1"].to_numpy()[data.row_index]
    y = data.y

    majority = max(y.mean(), 1.0 - y.mean())
    persistence = ((prev_ret > 0).astype("float32") == y).mean()
    return {
        "majority": float(majority),
        "persistence": float(persistence),
        "positive_rate": float(y.mean()),
    }
