"""Step 3 — HMM regime labelling.

A Gaussian HMM fit on the training split labels every bar with a market regime:
bull, bear, sideways or high-vol. The label is what conditions the three model
architectures in step 4.

The two decisions that matter
-----------------------------
**1. Filtered posteriors, not Viterbi.** `hmmlearn`'s `predict()` runs Viterbi
over the *whole* sequence, so the state it assigns to bar t depends on bars
t+1..T. For a chart that is fine and even desirable. For a forecasting model it
is lookahead: the regime label would carry information from the future the model
is being asked to predict. On this data the mean absolute gap between filtered
and smoothed posteriors is ~0.11, so it is not a rounding detail.

`filtered_posteriors()` runs the forward pass only — p(s_t | x_1..x_t) — and is
the one that feeds step 4. `smoothed_posteriors()` and `viterbi_path()` are
provided for charts and diagnostics and are labelled as such.

**2. The model is fit on train only.** The scaler's mean/std and every HMM
parameter come from the training split. The test split is transformed with the
frozen scaler and decoded by the frozen model, warm-started from the train tail
in the same past-into-future way the features are.

On choosing k
-------------
BIC does not select k on this dataset. Fit on 55,298 bars it falls monotonically
from k=2 (468,511) through k=10 (273,676) with every state roughly equally
occupied — the extra components partition the feature space more finely rather
than finding new regimes. At this sample size the p·log(n) penalty simply cannot
outrun the likelihood gain from more Gaussian components.

k is therefore fixed at 4 on interpretability grounds (the four regimes the
project specifies) and validated behaviourally instead: the labelling must put
the COVID crash in high-vol/bear and a quiet stretch in sideways. See
`tests/test_regimes.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn import _hmmc
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

# Chosen for near-orthogonality: of the 34 model-input features, these four have
# max pairwise |corr| = 0.23. The discarded alternatives were redundant —
# di_spread/rsi_14_norm correlate at 0.96, atr_pct/bb_bandwidth at 0.75. A full
# covariance HMM on collinear inputs produces ill-conditioned covariances.
REGIME_FEATURES: list[str] = ["ret_1", "ret_20", "atr_pct", "adx_14"]

N_STATES: int = 4
TREND_WINDOW: int = 20
RANDOM_SEED: int = 42

BULL, BEAR, SIDEWAYS, HIGH_VOL = "bull", "bear", "sideways", "high_vol"
REGIME_NAMES: list[str] = [BULL, BEAR, SIDEWAYS, HIGH_VOL]

# Bars of the preceding split used to warm-start the forward pass and to make
# the trend feature defined on the first target bar.
WARMUP_BARS: int = 250


def build_regime_inputs(df: pd.DataFrame, trend_window: int = TREND_WINDOW) -> pd.DataFrame:
    """Add the two return columns the HMM needs but the feature matrix lacks.

    Log returns, not simple returns: they are additive across bars, so `ret_20`
    is exactly the sum of twenty `ret_1`s, and they are far closer to symmetric,
    which matters for a model whose emissions are Gaussian.
    """
    out = df.copy()
    log_close = np.log(out["close"])
    out["ret_1"] = log_close.diff()
    out[f"ret_{trend_window}"] = log_close.diff(trend_window)
    return out


@dataclass
class RegimeModel:
    """A fitted HMM plus the train-only scaler and the state->regime mapping."""

    n_states: int = N_STATES
    features: list[str] = field(default_factory=lambda: list(REGIME_FEATURES))
    seed: int = RANDOM_SEED

    hmm: GaussianHMM | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    label_map: dict[int, str] = field(default_factory=dict)
    state_stats: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def _matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return the scaled feature matrix and a mask of usable rows."""
        valid = df[self.features].notna().all(axis=1)
        if "is_warmup" in df.columns:
            valid &= ~df["is_warmup"].astype(bool)
        X = df.loc[valid, self.features].to_numpy(dtype="float64")
        if self.mean_ is not None:
            X = (X - self.mean_) / self.std_
        return X, valid.to_numpy()

    def fit(self, train: pd.DataFrame, n_iter: int = 200) -> "RegimeModel":
        """Fit the scaler and the HMM on the training split, then label states."""
        valid = train[self.features].notna().all(axis=1)
        if "is_warmup" in train.columns:
            valid &= ~train["is_warmup"].astype(bool)
        raw = train.loc[valid, self.features].to_numpy(dtype="float64")

        # Standardise. The four features span ~4 orders of magnitude (ret_1 has
        # std 0.0017, adx_14 has 12.0); without this the covariance is dominated
        # by ADX and the returns contribute nothing.
        self.mean_ = raw.mean(axis=0)
        self.std_ = raw.std(axis=0)
        X = (raw - self.mean_) / self.std_

        self.hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            tol=1e-3,
            random_state=self.seed,
        ).fit(X)

        if not self.hmm.monitor_.converged:
            raise RuntimeError(
                f"HMM did not converge in {n_iter} iterations — raise n_iter "
                "rather than accepting the parameters"
            )

        self._assign_labels(train.loc[valid], X)
        return self

    def _assign_labels(self, rows: pd.DataFrame, X: np.ndarray) -> None:
        """Map state indices to regime names from each state's own statistics.

        Deterministic, so a refit with the same seed gives the same names:
        the most volatile state is high-vol, and the remaining three are ranked
        by mean trend return into bull / sideways / bear.
        """
        states = self.hmm.predict(X)
        trend = f"ret_{TREND_WINDOW}"
        stats = pd.DataFrame({
            "state": states,
            "ret_1": rows["ret_1"].to_numpy(),
            trend: rows[trend].to_numpy(),
            "atr_pct": rows["atr_pct"].to_numpy(),
            "adx_14": rows["adx_14"].to_numpy(),
        }).groupby("state").agg(
            n=("ret_1", "size"),
            mean_ret_1=("ret_1", "mean"),
            mean_trend=(trend, "mean"),
            mean_atr_pct=("atr_pct", "mean"),
            mean_adx=("adx_14", "mean"),
        )
        stats["share"] = stats["n"] / stats["n"].sum()

        if self.n_states != 4:
            # No four-regime story to tell; name by trend rank so the output is
            # still deterministic and ordered.
            order = stats["mean_trend"].sort_values(ascending=False).index
            self.label_map = {int(s): f"state_{i}" for i, s in enumerate(order)}
        else:
            high_vol = int(stats["mean_atr_pct"].idxmax())
            rest = stats.drop(index=high_vol)["mean_trend"].sort_values(ascending=False)
            self.label_map = {
                high_vol: HIGH_VOL,
                int(rest.index[0]): BULL,
                int(rest.index[1]): SIDEWAYS,
                int(rest.index[2]): BEAR,
            }

        stats["regime"] = [self.label_map[int(s)] for s in stats.index]
        self.state_stats = stats

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def _forward_log_alpha(self, X: np.ndarray) -> np.ndarray:
        frame_logprob = self.hmm._compute_log_likelihood(X)
        _, log_alpha = _hmmc.forward_log(
            self.hmm.startprob_, self.hmm.transmat_, frame_logprob
        )
        return log_alpha

    def filtered_posteriors(self, df: pd.DataFrame) -> np.ndarray:
        """p(s_t | x_1..x_t) — causal. **This is the one models may use.**"""
        X, _ = self._matrix(df)
        log_alpha = self._forward_log_alpha(X)
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    def smoothed_posteriors(self, df: pd.DataFrame) -> np.ndarray:
        """p(s_t | x_1..x_T) — uses future bars. Charts and diagnostics only."""
        X, _ = self._matrix(df)
        return self.hmm.score_samples(X)[1]

    def viterbi_path(self, df: pd.DataFrame) -> np.ndarray:
        """Most likely state sequence. Uses future bars — diagnostics only."""
        X, _ = self._matrix(df)
        return self.hmm.predict(X)


def attach_regimes(
    df: pd.DataFrame,
    model: RegimeModel,
    seed: pd.DataFrame | None = None,
    warmup_bars: int = WARMUP_BARS,
) -> pd.DataFrame:
    """Attach causal regime columns to `df`, optionally warm-started from `seed`.

    Columns added
    -------------
    regime_p_{name}      Filtered posterior for each regime, summing to 1.
    regime               argmax of those posteriors.
    regime_confidence    max posterior — how sure the model is.
    regime_changepoint   1 - max posterior. High means the model is between
                         regimes, which is exactly when a prediction should be
                         trusted least.
    regime_entropy       Posterior entropy, a smoother version of the same idea.
    regime_changed       True where `regime` differs from the previous bar.

    Rows the model cannot score (feature warm-up, undefined returns) get NaN
    posteriors and a null regime rather than a guess.
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

    _, valid = model._matrix(combined)
    posteriors = model.filtered_posteriors(combined)

    names = [model.label_map[i] for i in range(model.n_states)]
    full = pd.DataFrame(np.nan, index=combined.index, columns=names)
    full.loc[valid, names] = posteriors

    out = combined.copy()
    for name in names:
        out[f"regime_p_{name}"] = full[name]

    p = full.to_numpy(dtype="float64")
    # Reduce only over rows that were actually scored; np.nanmax over an
    # all-NaN row warns and returns NaN, which is noise rather than information.
    scored = ~np.isnan(p).all(axis=1)
    best = np.full(len(p), np.nan)
    idx = np.zeros(len(p), dtype=int)
    if scored.any():
        best[scored] = p[scored].max(axis=1)
        idx[scored] = p[scored].argmax(axis=1)

    out["regime"] = pd.Series(
        [names[i] if ok else None for i, ok in zip(idx, scored)], index=out.index
    ).astype("string")
    out["regime_confidence"] = np.where(scored, best, np.nan)
    out["regime_changepoint"] = np.where(scored, 1.0 - best, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    out["regime_entropy"] = np.where(scored, ent, np.nan)

    out["regime_changed"] = out["regime"].ne(out["regime"].shift(1)) & out["regime"].notna()

    return out.iloc[n_seed:].reset_index(drop=True)
