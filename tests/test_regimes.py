"""Correctness of the HMM regime labelling.

The load-bearing tests here are the causality ones: `test_filtered_has_no_lookahead`
and `test_filtered_differs_from_smoothed`. Everything else in step 4 depends on
the regime label at bar t being computable from bars 1..t alone.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src import config
from src.regimes.hmm import (
    BEAR,
    BULL,
    HIGH_VOL,
    REGIME_FEATURES,
    REGIME_NAMES,
    SIDEWAYS,
    RegimeModel,
    attach_regimes,
    build_regime_inputs,
)

MODEL_PATH = config.PROJECT_ROOT / "models" / "regime_hmm.pkl"
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="run `python scripts/03_label_regimes.py` first",
)


@pytest.fixture(scope="module")
def model() -> RegimeModel:
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return build_regime_inputs(
        pd.read_parquet(config.PROCESSED_DIR / "features_train.parquet")
    )


@pytest.fixture(scope="module")
def test_split() -> pd.DataFrame:
    return build_regime_inputs(
        pd.read_parquet(config.PROCESSED_DIR / "features_test.parquet")
    )


@pytest.fixture(scope="module")
def labelled(train) -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / "regimes_train.parquet")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def test_log_returns_are_additive(train):
    """ret_20 must be the sum of twenty ret_1s — that is why they are log returns."""
    r1, r20 = train["ret_1"], train["ret_20"]
    rolling = r1.rolling(20).sum()
    both = pd.DataFrame({"a": r20, "b": rolling}).dropna()
    np.testing.assert_allclose(both["a"], both["b"], atol=1e-10)


# --------------------------------------------------------------------------- #
# fitted model
# --------------------------------------------------------------------------- #
def test_model_converged(model):
    assert model.hmm.monitor_.converged


def test_transition_rows_are_distributions(model):
    T = model.hmm.transmat_
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-10)
    assert (T >= 0).all()


def test_all_four_regimes_are_named_exactly_once(model):
    assert sorted(model.label_map.values()) == sorted(REGIME_NAMES)
    assert set(model.label_map.keys()) == set(range(model.n_states))


def test_label_rule_matches_state_statistics(model):
    """The names must follow from the data, not from a hand assignment."""
    s = model.state_stats
    assert s.loc[s["regime"] == HIGH_VOL, "mean_atr_pct"].iloc[0] == s["mean_atr_pct"].max()
    others = s[s["regime"] != HIGH_VOL]
    assert others.loc[others["regime"] == BULL, "mean_trend"].iloc[0] == others["mean_trend"].max()
    assert others.loc[others["regime"] == BEAR, "mean_trend"].iloc[0] == others["mean_trend"].min()


def test_regimes_are_economically_coherent(model):
    """Sanity the names actually describe the states they are attached to."""
    s = model.state_stats.set_index("regime")
    assert s.loc[BULL, "mean_ret_1"] > 0
    assert s.loc[BEAR, "mean_ret_1"] < 0
    assert abs(s.loc[SIDEWAYS, "mean_trend"]) < abs(s.loc[BULL, "mean_trend"])
    # A ranging state should be the least trending of the four.
    assert s.loc[SIDEWAYS, "mean_adx"] == s["mean_adx"].min()


def test_regimes_are_persistent(model):
    """Diagonal dominance — a regime that flips every bar is not a regime."""
    T = model.hmm.transmat_
    assert (np.diag(T) > 0.5).all()
    for i in range(model.n_states):
        assert T[i, i] == T[i].max()


def test_fit_is_deterministic(train):
    """Same seed, same data, same labelling — otherwise results are not reproducible."""
    sub = train.iloc[:12000]
    a = RegimeModel(seed=7).fit(sub, n_iter=40)
    b = RegimeModel(seed=7).fit(sub, n_iter=40)
    np.testing.assert_allclose(a.hmm.transmat_, b.hmm.transmat_, atol=1e-10)
    assert a.label_map == b.label_map


# --------------------------------------------------------------------------- #
# causality — the tests that matter
# --------------------------------------------------------------------------- #
def test_filtered_posteriors_are_distributions(model, train):
    p = model.filtered_posteriors(train)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert (p >= 0).all() and (p <= 1).all()


def _perturb(model, df, row=3000, factor=1.03):
    """Bump one bar and return (baseline, perturbed, index of the bump in X space).

    `_matrix` drops rows with undefined features, so the frame's row number is
    not the posterior's row number — the caller must compare in X space or it
    will test the wrong bars.
    """
    _, valid = model._matrix(df)
    x_index = int(valid[:row].sum())
    bumped = df.copy()
    bumped.loc[row, ["close", "high", "low"]] *= factor
    return build_regime_inputs(bumped), x_index


def test_filtered_has_no_lookahead(model, test_split):
    """Perturbing a bar must leave every filtered posterior before it untouched.

    This is the property Viterbi does not have, and the whole reason the
    pipeline does not use `predict()`.
    """
    df = test_split.head(4000).copy()
    base = model.filtered_posteriors(df)
    bumped_df, k = _perturb(model, df)
    bumped = model.filtered_posteriors(bumped_df)

    # Bit-identical, not merely close: the past cannot depend on the future.
    np.testing.assert_array_equal(base[:k], bumped[:k])
    assert not np.allclose(base[k:], bumped[k:], atol=1e-9)


def test_smoothed_does_have_lookahead(model, test_split):
    """The counterexample that makes the previous test meaningful.

    Smoothing propagates the change backwards. The influence decays
    geometrically with the transition matrix, so it is only visible close to the
    perturbation — checking 100 bars back would find nothing and prove nothing.
    """
    df = test_split.head(4000).copy()
    base = model.smoothed_posteriors(df)
    bumped_df, k = _perturb(model, df)
    bumped = model.smoothed_posteriors(bumped_df)

    assert not np.allclose(base[k - 20:k], bumped[k - 20:k], atol=1e-9), \
        "smoothed posteriors before the perturbation should have moved"


def test_filtered_and_smoothed_agree_only_at_the_final_bar(model, test_split):
    """At T there is no future left, so the two must coincide there and only there."""
    f = model.filtered_posteriors(test_split)
    s = model.smoothed_posteriors(test_split)
    np.testing.assert_allclose(f[-1], s[-1], atol=1e-8)
    assert np.abs(f - s).mean() > 0.01      # they genuinely differ elsewhere


def test_seed_must_precede_target(model, test_split):
    with pytest.raises(ValueError, match="strictly before"):
        attach_regimes(test_split, model, seed=test_split)


# --------------------------------------------------------------------------- #
# attached output
# --------------------------------------------------------------------------- #
def test_attach_preserves_row_alignment(model, test_split):
    out = attach_regimes(test_split, model, seed=None)
    assert len(out) == len(test_split)
    pd.testing.assert_series_equal(
        out["timestamp"], test_split.reset_index(drop=True)["timestamp"]
    )


def test_posterior_columns_sum_to_one(labelled):
    cols = [f"regime_p_{n}" for n in REGIME_NAMES]
    scored = labelled[labelled["regime"].notna()]
    np.testing.assert_allclose(scored[cols].sum(axis=1), 1.0, atol=1e-9)


def test_regime_is_argmax_of_posteriors(labelled):
    cols = [f"regime_p_{n}" for n in REGIME_NAMES]
    scored = labelled[labelled["regime"].notna()]
    expected = scored[cols].idxmax(axis=1).str.removeprefix("regime_p_")
    assert (scored["regime"].astype(str) == expected).all()


def test_confidence_and_changepoint_are_complementary(labelled):
    scored = labelled[labelled["regime"].notna()]
    np.testing.assert_allclose(
        scored["regime_confidence"] + scored["regime_changepoint"], 1.0, atol=1e-9
    )
    assert scored["regime_confidence"].between(0.25, 1.0).all()   # >= 1/k


def test_unscored_rows_have_no_regime(labelled):
    """Warm-up rows must be null, not silently assigned a state."""
    unscored = labelled[labelled["regime"].isna()]
    assert len(unscored) > 0
    cols = [f"regime_p_{n}" for n in REGIME_NAMES]
    assert unscored[cols].isna().all().all()
    assert unscored["regime_confidence"].isna().all()


# --------------------------------------------------------------------------- #
# behavioural validation — this is what justifies k=4, since BIC does not
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "month,expected",
    [
        ("2020-03", HIGH_VOL),   # COVID crash
        ("2020-04", HIGH_VOL),   # the rebound was just as violent
        ("2022-06", BEAR),       # 2022 bear leg
        ("2017-06", SIDEWAYS),   # quiet range
    ],
)
def test_known_periods_get_the_right_regime(labelled, month, expected):
    d = labelled.dropna(subset=["regime"])
    sub = d[d["timestamp"].dt.strftime("%Y-%m") == month]
    assert len(sub) > 100, f"{month} has too few bars to judge"
    assert sub["regime"].value_counts(normalize=True).idxmax() == expected


def test_crash_month_is_more_volatile_than_calm_month(labelled):
    """Guards against the labels being coherent but attached to the wrong states."""
    d = labelled.dropna(subset=["regime"])
    month = d["timestamp"].dt.strftime("%Y-%m")
    assert d.loc[month == "2020-03", "atr_pct"].mean() > \
           3 * d.loc[month == "2017-06", "atr_pct"].mean()


def test_every_regime_is_actually_used(labelled):
    share = labelled["regime"].value_counts(normalize=True)
    assert set(share.index) == set(REGIME_NAMES)
    assert share.min() > 0.02      # no vestigial state


def test_shipped_model_was_fit_on_train_only(model, train, test_split):
    """The HMM must be fit on TRAIN alone; test is only decoded by it.

    Checked via the scaler rather than the HMM parameters: `mean_`/`std_` are a
    deterministic function of exactly the rows used for fitting, with no EM in
    between, so this is both instant and unambiguous. Fitting on train+test
    moves the scaler by ~0.15, far above any floating-point noise.
    """
    train_only = RegimeModel()
    both = RegimeModel()
    for target, frame in ((train_only, train),
                          (both, pd.concat([train, test_split], ignore_index=True))):
        usable = frame[target.features].notna().all(axis=1) & ~frame["is_warmup"].astype(bool)
        raw = frame.loc[usable, target.features].to_numpy(dtype="float64")
        target.mean_, target.std_ = raw.mean(axis=0), raw.std(axis=0)

    np.testing.assert_allclose(model.mean_, train_only.mean_, atol=1e-12)
    np.testing.assert_allclose(model.std_, train_only.std_, atol=1e-12)
    assert np.abs(model.mean_ - both.mean_).max() > 1e-3, \
        "shipped scaler matches a train+test fit — test data leaked into fitting"
