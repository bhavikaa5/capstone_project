# Step 4 — Model Training and Accuracy

The flowchart's three competing architectures, trained on the regime-labelled
feature matrix to predict the direction of the next 15-minute bar.

```bash
python scripts/04_train_models.py --epochs 30      # ~40 min on 12 CPU cores
python scripts/04_train_models.py --report-only    # rebuild report, no retrain
python -m pytest tests/ -q                         # 122 tests
```

## Headline result

| Model | Test accuracy | AUC | MCC | vs persistence |
|---|---|---|---|---|
| **LSTM** | **63.42%** | 0.694 | 0.270 | **+2.01 pp** |
| CNN-BiLSTM-Attention | 63.08% | 0.685 | 0.261 | +1.67 pp |
| Transformer | 61.60% | 0.675 | 0.237 | +0.19 pp |
| *persistence baseline* | *61.41%* | — | — | — |
| *majority class* | *50.76%* | — | — | — |

All three beat both baselines. The simplest architecture wins.

## Read the baseline before the accuracy

**63% accuracy sounds good and mostly is not.** Two baselines have to be cleared:

- **Majority class** — always predict "up": 50.76%.
- **Persistence** — "the next bar goes the same way as this one": **61.41%**.

Persistence is strong because 15-minute Nifty returns are positively
autocorrelated. Measured properly, on consecutive bars *within* a session:

| | Train | Test |
|---|---|---|
| Intraday lag-1 autocorrelation | **+0.30** | **+0.25** |
| Overnight (last bar → next open) | −0.08 | −0.01 |

So the momentum is genuinely intraday, not a session-boundary artifact. A naive
`.autocorr(1)` over the whole series reports 0.18 because it mixes the ~0
overnight pairs in; grouping by `bar_of_day` measures day-over-day correlation
at the same clock time, which is a different quantity again. Both are easy to
quote by accident.

**MCC is reported alongside accuracy** because MCC collapses to 0 for a constant
predictor while accuracy would still read 50.8%.

## Setup

| | |
|---|---|
| Target | direction of next bar (1 = close rises) |
| Lookback | 32 bars (~1.3 sessions) |
| Features | 43 = 34 technical + `ret_1`, `ret_20` + 7 regime |
| Windows | 46,961 train / 8,292 val / 5,758 test |
| Capacity | 41k / 72k / 72k parameters |

Capacity is matched deliberately so the comparison is between architectures, not
between a big model and a small one.

## Leakage controls

1. **Purge gap of 33 windows** between train and validation. Without it the last
   training windows share input bars with the first validation windows, and a
   training label reaches into the validation period.
2. **Scaler fit on training windows only**, applied frozen to val and test.
3. **Threshold chosen on validation**, then frozen for test. Tuning it on test
   is scoring against the answer key.
4. **`padding="causal"` in the CNN.** `"same"` padding would let the convolution
   at bar *t* read bars *t+1* and *t+2*.
5. **Causal attention mask in the transformer**, pooling at the last position
   rather than averaging over time.

Tests assert the window ending at row *t* equals source rows *t−31..t* exactly,
that the label describes the move *after* it, and that no window straddles a
dropped warm-up gap.

## Overfitting

| Model | Train | Val | Test |
|---|---|---|---|
| LSTM | 66.43% | 65.36% | 63.42% |
| CNN-BiLSTM-Attention | 67.60% | 64.40% | 63.08% |
| Transformer | 64.52% | 63.78% | 61.60% |

Monotone and mild — about 3 pp train→test. The CNN-BiLSTM has the widest gap
(4.5 pp), consistent with its higher capacity. Early stopping on validation AUC
with patience 6 and best-weight restore.

## Accuracy by regime (LSTM, test)

| Regime | n | Share | Accuracy | MCC |
|---|---|---|---|---|
| high_vol | 342 | 5.9% | **66.67%** | 0.324 |
| bear | 2,007 | 34.9% | 64.62% | 0.295 |
| bull | 1,225 | 21.3% | 63.10% | 0.239 |
| sideways | 2,184 | 37.9% | 62.00% | 0.246 |

The model holds up in high-vol — where it would be most useful and most likely
to fail. Sideways is weakest, which is expected: that is where directional
signal genuinely is thinnest.

## The finding that matters most: the edge is not tradeable

A naive long/short strategy on the LSTM signal returns **+165.8% gross** over the
test year with an annualised Sharpe of 12.7. **That number is not credible and
should not be reported without its cost sensitivity:**

| Cost per position change | Net return |
|---|---|
| 0 bps | +165.8% |
| 1 bp | +136.8% |
| 2 bps | +107.7% |
| 5 bps | +20.4% |
| **5.70 bps** | **break-even** |
| 10 bps | −125.0% |

The model changes position **12.6 times per session** — roughly 3,000 round trips
a year — so cost enters 3,000 times. Real Nifty round-trip cost (bid-ask spread
plus STT, exchange fee, GST and stamp duty) is comfortably above 5.7 bps.

**The classification result is real; the profitability is not.** Reporting the
gross figure alone would be the single most misleading thing this project could
do.

Two further caveats worth stating:

- The models agree with the plain persistence rule on **75–81%** of bars. Most of
  what they have learned is the autocorrelation; the +2 pp is the genuinely new
  part.
- Positive index autocorrelation is partly a **non-synchronous trading artifact**
  — the index is a weighted average of 50 stocks that do not all print at the
  same instant, so the level lags. A single continuously-traded instrument
  (futures, ETF) does not carry it to the same degree, so even the 5.7 bps
  break-even is optimistic for anything actually executable.

## Bug found and fixed

`AttentionPooling` created its Dense sub-layers in `__init__` but never built
them, so `keras.models.load_model` refused to reconstruct the saved
CNN-BiLSTM-Attention model — "Layer 'dense' was never built … the weights file
lists 2 variables". Adding an explicit `build()` fixed it. This would have
surfaced later as a broken step 5 or a live-inference failure. Verified by a
save/load round-trip returning identical predictions.

The scaler also accumulated in float32 over 1.5M values, leaving 3 of 43
features with a mean off by ~0.02 std. Harmless for training but not
reproducible; it now accumulates in float64.

## For step 5 (Optuna)

- Tune on **validation only**; test stays sealed.
- Optimise **AUC or MCC**, not accuracy — accuracy is threshold-dependent and the
  threshold is itself tuned.
- The bar to beat is **63.42%** (LSTM), and the honest target is lift over
  persistence at 61.41%, i.e. +2.01 pp.
- Include cost sensitivity in the objective if profitability is a goal. Tuning
  purely for accuracy will happily produce a model that flips more often and
  loses more money.
