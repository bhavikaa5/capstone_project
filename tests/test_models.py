"""Correctness of the step-4 supervised setup.

These tests are about the *data contract*, not about whether a model is any
good. The expensive part (training) is not exercised here; what is exercised is
every place a sequence model can silently see the future.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.dataset import (
    HORIZON,
    LOOKBACK,
    Scaler,
    baseline_scores,
    build_sequences,
    chronological_split,
    feature_columns,
    make_target,
)
from src.models.evaluate import (
    classification_metrics,
    directional_strategy,
    pick_threshold,
    regime_breakdown,
)

DATA = config.PROCESSED_DIR / "regimes_train.parquet"
pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="run `python scripts/03_label_regimes.py` first"
)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_parquet(DATA)


@pytest.fixture(scope="module")
def cols(train_df) -> list[str]:
    return feature_columns(train_df)


@pytest.fixture(scope="module")
def seqs(train_df, cols):
    return build_sequences(train_df, cols, LOOKBACK, HORIZON)


# --------------------------------------------------------------------------- #
# feature contract
# --------------------------------------------------------------------------- #
def test_feature_list_has_no_duplicates(cols):
    assert len(cols) == len(set(cols))


def test_feature_list_excludes_price_and_leaky_columns(cols):
    """Raw price, the vendor columns and the label's ingredients must be absent."""
    banned = {"close", "open", "high", "low", "timestamp", "regime",
              "is_warmup", "session_date", "macd", "atr_14", "bb_upper"}
    assert not (set(cols) & banned)
    assert not any(c.startswith("src_") for c in cols)


def test_regime_posteriors_are_included(cols):
    for name in ("bull", "bear", "sideways", "high_vol"):
        assert f"regime_p_{name}" in cols


# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #
def test_target_is_the_sign_of_the_forward_return(train_df):
    y = make_target(train_df, horizon=1)
    close = train_df["close"].to_numpy()
    for i in (100, 5000, 20000):
        assert y.iloc[i] == float(close[i + 1] > close[i])


def test_target_tail_is_undefined(train_df):
    """The last `horizon` bars have no future to label."""
    y = make_target(train_df, horizon=HORIZON)
    assert y.iloc[-HORIZON:].isna().all()
    assert y.iloc[:-HORIZON].notna().sum() > 0


# --------------------------------------------------------------------------- #
# windowing — where lookahead would hide
# --------------------------------------------------------------------------- #
def test_window_contains_only_past_and_present(train_df, cols, seqs):
    """The window ending at row t must equal rows t-31..t of the source frame."""
    src = train_df.reset_index(drop=True)[cols].to_numpy(dtype="float32")
    for k in (0, 1234, len(seqs) - 1):
        end = seqs.row_index[k]
        np.testing.assert_array_equal(seqs.X[k], src[end - LOOKBACK + 1:end + 1])


def test_label_belongs_to_the_bar_after_the_window(train_df, seqs):
    """y[k] must describe the move AFTER the last bar in window k, not inside it."""
    close = train_df.reset_index(drop=True)["close"].to_numpy()
    for k in (0, 500, len(seqs) - 1):
        end = seqs.row_index[k]
        assert seqs.y[k] == float(close[end + HORIZON] > close[end])


def test_windows_never_straddle_a_dropped_gap(train_df, cols, seqs):
    """Warm-up rows are removed; a window must not silently span the hole."""
    src = train_df.reset_index(drop=True)
    usable = (~src["is_warmup"].astype(bool)) & src["regime"].notna()
    for k in range(0, len(seqs), max(1, len(seqs) // 200)):
        end = seqs.row_index[k]
        assert usable.iloc[end - LOOKBACK + 1:end + 1].all()


def test_row_indices_are_strictly_increasing(seqs):
    assert (np.diff(seqs.row_index) > 0).all()


def test_no_nan_or_inf_in_windows(seqs):
    assert not np.isnan(seqs.X).any()
    assert not np.isinf(seqs.X).any()
    assert set(np.unique(seqs.y)) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# split — purging
# --------------------------------------------------------------------------- #
def test_split_is_chronological(seqs):
    train, val = chronological_split(seqs)
    assert train.row_index.max() < val.row_index.min()


def test_split_purges_overlapping_windows(seqs):
    """The gap must exceed the lookback, or train windows share bars with val.

    Without it the final training samples contain bars that also appear in the
    first validation samples, and a training label reaches into the validation
    period.
    """
    train, val = chronological_split(seqs)
    gap = val.row_index.min() - train.row_index.max()
    assert gap > LOOKBACK, f"gap {gap} does not clear the {LOOKBACK}-bar lookback"


def test_no_window_overlap_between_train_and_val(seqs):
    train, val = chronological_split(seqs)
    last_train_bars = set(range(train.row_index.max() - LOOKBACK + 1,
                                train.row_index.max() + 1))
    first_val_bars = set(range(val.row_index.min() - LOOKBACK + 1,
                               val.row_index.min() + 1))
    assert not (last_train_bars & first_val_bars)


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def test_scaler_is_fit_on_training_data_only(seqs):
    train, val = chronological_split(seqs)
    s = Scaler().fit(train.X)
    scaled = s.transform(train.X).reshape(-1, train.X.shape[-1])
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-2)


def test_scaler_never_divides_by_zero():
    X = np.ones((10, 4, 3), dtype="float32")     # every feature constant
    out = Scaler().fit(X).transform(X)
    assert not np.isnan(out).any() and not np.isinf(out).any()


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit must be called"):
        Scaler().transform(np.zeros((2, 3, 4), dtype="float32"))


# --------------------------------------------------------------------------- #
# baselines and metrics
# --------------------------------------------------------------------------- #
def test_persistence_baseline_is_strong(train_df, seqs):
    """If this drops to ~50% the return autocorrelation has gone, and every
    accuracy number in the report needs rechecking against the new baseline."""
    b = baseline_scores(train_df, seqs)
    assert 0.55 < b["persistence"] < 0.75
    assert b["persistence"] > b["majority"]


def test_metrics_agree_with_the_confusion_matrix():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y)
    assert m["accuracy"] == pytest.approx((m["tp"] + m["tn"]) / len(y))
    assert m["precision"] == pytest.approx(m["tp"] / (m["tp"] + m["fp"]))
    assert m["recall"] == pytest.approx(m["tp"] / (m["tp"] + m["fn"]))


def test_mcc_is_zero_for_a_constant_predictor():
    """The reason MCC is reported: accuracy would read 60% for this predictor."""
    y = np.array([1] * 60 + [0] * 40)
    p = np.full(100, 0.99)
    m = classification_metrics(y, p, threshold=0.5)
    assert m["accuracy"] == pytest.approx(0.60)
    assert m["mcc"] == pytest.approx(0.0)


def test_threshold_is_chosen_within_the_grid():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    assert 0.30 <= pick_threshold(y, p) <= 0.70


def test_regime_breakdown_shares_sum_to_100():
    rng = np.random.default_rng(1)
    n = 400
    rb = regime_breakdown(
        rng.integers(0, 2, n), rng.random(n),
        rng.choice(["bull", "bear", "sideways", "high_vol"], n),
    )
    assert rb["n"].sum() == n
    assert rb["share_pct"].sum() == pytest.approx(100.0)


def test_strategy_charges_costs_only_on_position_changes():
    prob = np.array([0.9, 0.9, 0.9, 0.1, 0.1])      # one flip
    fwd = np.zeros(5)
    s = directional_strategy(prob, fwd, threshold=0.5, cost_bps=10.0)
    assert s["position_flips"] == 2                  # entry, then the reversal
    assert s["gross_return_pct"] == pytest.approx(0.0)
    assert s["net_return_pct"] < 0                   # only costs remain


# --------------------------------------------------------------------------- #
# architectures — causality and serialisation
# --------------------------------------------------------------------------- #
def test_attention_model_survives_a_save_load_round_trip(tmp_path):
    """Regression test: AttentionPooling built its Dense children in __init__
    but never built them, so `load_model` could not reconstruct the saved model.
    A broken reload would only surface in step 5 or at live inference."""
    import keras
    from src.models.architectures import ARCHITECTURES, set_seeds

    set_seeds(0)
    model = ARCHITECTURES["cnn_bilstm_attention"]((8, 5))
    x = np.random.default_rng(0).normal(size=(4, 8, 5)).astype("float32")
    before = model.predict(x, verbose=0)

    path = tmp_path / "rt.keras"
    model.save(path)
    after = keras.models.load_model(path).predict(x, verbose=0)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_cnn_uses_causal_padding():
    """'same' padding would let the convolution at bar t read bars t+1 and t+2."""
    from src.models.architectures import ARCHITECTURES, set_seeds

    set_seeds(0)
    model = ARCHITECTURES["cnn_bilstm_attention"]((16, 4))
    convs = [l for l in model.layers if l.__class__.__name__ == "Conv1D"]
    assert convs, "expected Conv1D layers"
    assert all(l.padding == "causal" for l in convs)


@pytest.mark.parametrize("name", ["lstm", "cnn_bilstm_attention", "transformer"])
def test_prediction_at_t_ignores_later_bars_in_the_window(name):
    """Changing the LAST bar must move the output; the models read the window's
    end. Changing a bar *after* the window cannot reach the model at all, which
    is guaranteed by the windowing tests above."""
    from src.models.architectures import ARCHITECTURES, set_seeds

    set_seeds(0)
    model = ARCHITECTURES[name]((16, 4))
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 16, 4)).astype("float32")
    base = model.predict(x, verbose=0)

    bumped = x.copy()
    bumped[:, -1, :] += 3.0
    assert not np.allclose(base, model.predict(bumped, verbose=0), atol=1e-6)
