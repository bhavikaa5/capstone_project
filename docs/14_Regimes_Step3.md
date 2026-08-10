# Step 3 — HMM Regime Labelling

The flowchart's *HMM regime labelling* box. A Gaussian HMM fit on the training
split labels every 15-minute bar as **bull**, **bear**, **sideways** or
**high-vol**. That label is what conditions the three architectures in step 4.

```bash
python scripts/03_label_regimes.py
python -m pytest tests/ -q          # 100 tests
```

## Files

| Path | Role |
|---|---|
| `src/regimes/hmm.py` | `RegimeModel`, `build_regime_inputs`, `attach_regimes` |
| `scripts/03_label_regimes.py` | Runs the stage |
| `tests/test_regimes.py` | 24 tests |
| `data/processed/regimes_{train,test}.parquet` | **Pipeline output** |
| `models/regime_hmm.pkl` | Frozen model — step 4 loads this, does not refit |
| `reports/03_regimes.xlsx` | Deliverable workbook (7 MB) |

## The decision that matters most

**The regime columns are causal. They come from the forward pass only.**

`hmmlearn`'s `predict()` runs Viterbi over the *entire* sequence, so the state it
assigns to bar *t* is chosen using bars *t+1 … T*. On a chart that is desirable —
it gives clean, smooth regime blocks. As a model input it is **lookahead**: the
regime label would carry information from precisely the future the model is being
asked to forecast.

Measured on this data:

| | |
|---|---|
| Mean abs difference, filtered vs smoothed posteriors | **0.039** |
| Bars where the two disagree on the argmax regime | **8.0%** |

So one bar in twelve would have been mislabelled with future information. Not a
rounding detail.

`filtered_posteriors()` — p(s_t │ x_1..x_t) — is what `attach_regimes` writes.
`smoothed_posteriors()` and `viterbi_path()` exist for charts and diagnostics and
are documented as such. Two tests enforce the distinction:

- `test_filtered_has_no_lookahead` — perturbing a bar leaves every earlier
  filtered posterior **bit-identical**.
- `test_smoothed_does_have_lookahead` — the same perturbation *does* move the
  smoothed posteriors before it. Without this counterexample the first test
  could pass trivially.

## Features fed to the HMM

Four, chosen for near-orthogonality (max pairwise |corr| = 0.23):

| Feature | Role |
|---|---|
| `ret_1` | bar return |
| `ret_20` | 20-bar trend |
| `atr_pct` | volatility |
| `adx_14` | trend strength |

Log returns, not simple returns: they are additive, so `ret_20` is exactly the
sum of twenty `ret_1`s (asserted in `test_log_returns_are_additive`), and they
are closer to symmetric, which matters when the emissions are Gaussian.

Rejected as collinear: `di_spread`/`rsi_14_norm` (r = 0.96 with each other),
`bb_bandwidth` (r = 0.75 with `atr_pct`). A full-covariance HMM on collinear
inputs is ill-conditioned.

Features are standardised with the **train** mean and std. Without it the
covariance is dominated by `adx_14` (std 12.0) and the returns (std 0.0017)
contribute nothing.

## Choosing k — a negative result

**BIC does not select k on this dataset.** It falls monotonically:

| k | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| BIC | 468,511 | 407,583 | 371,161 | 348,038 | 321,065 | 291,015 | 273,676 |

At k = 10 every state is still roughly equally occupied (largest share 12.4%),
so the extra components are partitioning the feature space more finely, not
discovering new regimes. At n = 55,298 the `p·log(n)` penalty simply cannot
outrun the likelihood gain from more Gaussian components.

k is therefore **fixed at 4** on interpretability grounds — the four regimes the
project specifies — and validated behaviourally instead. Reporting this honestly
is better than quoting a BIC-optimal k that would be 10 or higher and
uninterpretable. Chart: `reports/figures/03_regime_bic.png`.

## What the four states turned out to be

| Regime | Share | Mean bar return | Mean 20-bar trend | Mean ATR% | Mean ADX |
|---|---|---|---|---|---|
| bull | 23.8% | +1.33 bps | +0.78% | 0.185 | 38.0 |
| bear | 30.1% | −1.12 bps | −0.43% | 0.247 | 31.7 |
| sideways | 36.1% | +0.03 bps | −0.00% | 0.153 | 22.7 |
| high_vol | 10.0% | +1.91 bps | −0.19% | 0.460 | 37.5 |

Naming is a **deterministic rule**, not a manual assignment: highest mean
`atr_pct` becomes high-vol, and the remaining three rank by mean trend into
bull / sideways / bear. A refit with the same seed reproduces the same names
(`test_fit_is_deterministic`).

Note `high_vol` has a *positive* mean bar return but a *negative* trend — it
captures violent moves in both directions, which is what a volatility state
should do.

Transition matrix diagonals run 0.875–0.965, i.e. expected durations of 8 bars
(high-vol) to 29 bars (sideways). The model is not flickering.

## Validation

**Behavioural** — none of these dates were used to fit or tune anything:

| Period | Expected | Result |
|---|---|---|
| Mar 2020, COVID crash | high-vol | **96% high-vol** |
| Apr 2020, COVID rebound | high-vol | **92% high-vol** |
| Jun 2022, bear leg | bear | **64% bear** |
| Jun 2017, quiet range | sideways | **76% sideways** |

**Independent corroboration:** the sideways state has the lowest mean ADX of the
four at 22.7 — just under Wilder's ranging threshold of 25. Nothing in the
fitting or the naming rule forced that.

**Also tested:** posteriors sum to 1; `regime` is the argmax of the posteriors;
`confidence + changepoint = 1`; warm-up rows are null rather than guessed;
transition rows are distributions with dominant diagonals; every regime is used
(min share > 2%); the seed must end strictly before the target split.

## Columns for step 4

| Column | Use |
|---|---|
| `regime_p_{bull,bear,sideways,high_vol}` | Filtered posteriors — the soft conditioning input |
| `regime` | argmax label — hard conditioning, or for stratified evaluation |
| `regime_confidence` | max posterior |
| `regime_changepoint` | 1 − max posterior; high = model is between regimes |
| `regime_entropy` | smoother version of the same |
| `regime_changed` | regime differs from previous bar |

Prefer the **posteriors** over the hard `regime` label where the architecture
allows it — they carry the model's uncertainty, and `regime_changepoint` marks
exactly the bars where a step-4 prediction should be trusted least. 2.8% of bars
have confidence below 0.6.

## Carrying forward

- The regime mix differs between splits: test is 5.9% high-vol against train's
  10.6%. The 2024-05 → 2025-04 test window was genuinely calmer than the
  2015–2024 average. Step 6 evaluation should be **regime-stratified**, or a
  model will look better on test simply for having faced fewer violent bars.
- 55,318 of 55,586 train bars are scored; the rest are step-2 feature warm-up.
- `models/regime_hmm.pkl` is the frozen artefact. Step 4 must load it, never
  refit — refitting on train+test would leak.
