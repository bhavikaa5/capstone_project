# Step 2 (part 1 of 5) — RSI Feature Block

First of the five feature families in the flowchart's *Feature engineering* box
(RSI, MACD, ATR, ADX, Bollinger). MACD/ATR/ADX/Bollinger plug in alongside this
one and share the same warm-start machinery.

```bash
python scripts/02a_rsi_features.py
python -m pytest tests/ -q
```

## Files

| Path | Role |
|---|---|
| `src/features/rsi.py` | `wilder_rma`, `rsi`, `build_rsi_features`, `attach_rsi` |
| `scripts/02a_rsi_features.py` | Runs the stage, writes parquet + Excel + figure |
| `tests/test_rsi.py` | 16 tests incl. Wilder's published example |
| `data/processed/rsi_{train,test}.parquet` | Pipeline output |

The RSI-only workbook this script used to produce was removed once step 2 was
complete — every one of its columns is in `reports/02_features.xlsx`, and two
workbooks could drift out of sync. Re-run the script if you want it back.

## Columns produced

| Column | Meaning |
|---|---|
| `rsi_7`, `rsi_14`, `rsi_21` | Wilder RSI, 0–100. 14 is canonical; 7/21 give the model a fast/slow contrast |
| `rsi_14_norm` | `(rsi_14 − 50) / 50` — zero-centred, roughly [−1, 1]; networks train better on this than on 0–100 |
| `rsi_14_chg` | One-bar change — RSI *direction* carries information the level does not |
| `rsi_14_slope_5` | 5-bar slope, smoothed direction |
| `rsi_spread_7_21` | Fast − slow RSI, a momentum-divergence proxy |
| `rsi_14_overbought` / `_oversold` / `_zone` | 70/30 threshold flags and a −1/0/+1 zone code |
| `rsi_is_warmup` | TRUE while the smoothing is still seed-dominated |

## Three decisions worth defending in the viva

**1. Wilder's RMA, not `ewm`.** RSI is defined with Wilder's running mean
(α = 1/n) *seeded by the simple mean of the first n changes*. A bare
`ewm(alpha=1/n)` adapts faster and produces a visibly different series that will
not reconcile with TradingView or with the vendor's own indicator columns. The
seed is implemented explicitly in `wilder_rma`.

**2. Continuous series, not reset per session.** Overnight gaps are genuine index
moves — resetting RSI each morning would discard 14 of every 25 bars in a
session, i.e. 56% of the data. The step-1 session flags remain available for any
later feature that genuinely needs a reset.

**3. Test warm-started from the train tail (250 bars).** Computing the test split
in isolation would leave its first 14 bars undefined and the next ~100 biased by
the seed. Prepending the last 250 train bars, computing, then discarding them
fixes both. This feeds *past* into *future*, which is the correct direction —
`attach_rsi` raises if the seed does not end strictly before the target begins,
and `test_warm_start_uses_only_past_data` guards it.

## Validation

- **Wilder's published worked example** reproduced to within 0.07 — the residual
  is the rounding in his printed table (`test_matches_wilders_published_example`).
- **Cross-checked against the `ta` library**: identical to machine precision
  beyond bar 1000. All disagreement sits inside the warm-up region, where `ta`
  seeds off a 13-change window instead of 14.
- **Edge cases**: unbroken advance → 100, unbroken decline → 0, flat market → 50.
- **Scale invariance**: doubling every price leaves RSI unchanged.
- **No look-ahead**: perturbing bar 400 leaves bars 0–399 bit-identical.
- **Ordering**: `std(RSI 7) > std(RSI 14) > std(RSI 21)`, so the periods are not
  transposed.

## Output stats

| | Train | Test |
|---|---|---|
| Rows | 55,586 | 5,810 |
| `rsi_14` NaN | 14 (definitional) | 0 (warm-started) |
| Warm-up flagged | 105 (0.19%) | 0 |
| Mean `rsi_14` | 52.4 | 51.0 |
| % bars overbought | 17.9% | 16.0% |
| % bars oversold | 12.0% | 13.8% |

The train-set skew toward overbought (17.9% vs 12.0%) is consistent with a
2015–2024 window that is net strongly upward — worth remembering when the HMM
labels regimes, since RSI alone will lean bullish.

## Note on the Excel workbook

`reports/02_rsi_features.xlsx` holds computed values, not Excel formulas: it is a
feature export, not a financial model, and 55,586 rows of live formulas would be
unusable. The README sheet documents every column, the method, the assumptions
and the validation. Arial throughout, frozen panes and autofilter on both data
sheets.

## Still blocked on

No volume column in the source data, so no volume-confirmed RSI variant (RSI on
OBV, volume-weighted RSI). RSI itself needs only closes and is unaffected.
