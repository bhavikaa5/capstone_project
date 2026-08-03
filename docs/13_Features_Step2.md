# Step 2 — Feature Engineering (all five blocks)

The flowchart's *Feature engineering* box: RSI, MACD, ATR, ADX, Bollinger.
All five are built, cross-validated and exported. Output feeds step 3 (HMM
regime labelling).

```bash
python scripts/02_build_features.py
python -m pytest tests/ -q          # 76 tests
```

## Files

| Path | Role |
|---|---|
| `src/features/base.py` | Wilder RMA, true range, crossover, warm-start (`attach_block`) |
| `src/features/rsi.py` | Block 1 — RSI(7/14/21) |
| `src/features/macd.py` | Block 2 — MACD(12,26,9) |
| `src/features/atr.py` | Block 3 — ATR(14) |
| `src/features/adx.py` | Block 4 — ADX/DMI(14) |
| `src/features/bollinger.py` | Block 5 — BB(20, 2σ) |
| `src/features/pipeline.py` | Assembles all five; `model_feature_columns()` |
| `scripts/02_build_features.py` | Runs the stage |
| `tests/test_rsi.py`, `tests/test_features.py` | 60 feature tests |
| `data/processed/features_{train,test}.parquet` | **Pipeline output** |
| `reports/02_features.xlsx` | Deliverable workbook (35 MB) |

`scripts/02a_rsi_features.py` and `reports/02_rsi_features.xlsx` are the earlier
RSI-only deliverable; still valid, superseded for pipeline use.

## Output

| | Train | Test |
|---|---|---|
| Rows | 55,586 | 5,810 |
| Total columns | 64 | 64 |
| Usable (past warm-up) | 55,318 | 5,810 |
| **Model-input columns** | **34** | **34** |
| NaN / inf past warm-up | 0 | 0 |

## The one decision that matters most

**Three of the five blocks produce values in index points, and those must never
reach a model.** Nifty 50 goes from ~8,000 to ~22,600 across the training
window, so a price-denominated indicator carries the price level as a hidden
trend. Measured on the training set:

| Year | mean \|MACD\| | mean ATR | mean `macd_pct` | mean `atr_pct` |
|---|---|---|---|---|
| 2015 | 14.06 | 18.62 | 0.173 | 0.230 |
| 2020 | 28.80 | 37.61 | 0.277 | 0.360 |
| 2024 | 27.94 | 42.48 | 0.127 | 0.193 |

The raw columns roughly double while the normalised ones stay in a stable band.
A model trained on raw MACD would learn "2024 = big numbers", not momentum.

Every affected column is kept for charting and chart-package validation, and
flagged `Model input? = no` on the workbook's **Columns** sheet with its
scale-free counterpart named. `model_feature_columns()` enforces this in code —
that is the function to call when building the model input, not a hand-written
list.

## Per-block notes

**RSI (7/14/21)** — Wilder's RMA with an explicit simple-mean seed. A bare
`ewm(alpha=1/n)` is a different, faster-adapting series. Already 0–100.

**MACD (12,26,9)** — `adjust=False` EMAs, the recursive form charting packages
use. `macd_cross` fires only on the bar the relationship flips (+1/−1), so it
stays sparse; `macd_above_signal` carries the persistent state.

**ATR (14)** — true range, not high−low, so overnight gaps count as the
volatility they are. `atr_pctile_1000` ranks ATR% against the trailing 1000 bars
(~40 sessions), which is what makes "high volatility" mean the same thing in
2015 and 2024.

**ADX/DMI (14)** — the only block that separates *trending* from *sideways*,
which no amount of RSI or MACD does. Already 0–100. **ADX is unsigned** — a
crash and a rally both read ~40 — so it must be paired with `di_spread`;
verified by test (Mar-2020 `di_spread` −5.35, Nov-2020 +6.08).

**Bollinger (20, 2σ)** — population standard deviation (`ddof=0`), per
Bollinger's definition. pandas defaults to `ddof=1`, which widens every band by
**2.6%** — small, silent, and wrong against every chart.

## Warm-up and leakage

The test split is warm-started from the last **1,250** train bars, which are
then discarded. That number is set by the two trailing-percentile columns, not
by the smoothing: `atr_pctile_1000` and `bb_bandwidth_pctile` need a full
ranking window behind the first target bar, or the test split's opening
percentiles are ranked against a shorter window than the training split's were
and the two are not comparable.

`attach_block()` raises if the seed does not end strictly before the target
begins. Feeding past into future is correct; the reverse would be leakage, and
`test_seed_must_precede_target` guards all five blocks.

`is_warmup` is TRUE where *any* block is still undefined or seed-dominated —
268 train rows, 0 test rows. Drop those before training. An earlier version of
the flag missed 180 NaNs from the percentile columns, which is why each block's
flag now covers its own percentile column explicitly.

## Validation

- **Every block matches the `ta` library to machine precision** beyond the
  warm-up region: MACD, ATR, ADX and Bollinger all give max difference 0.0.
- **RSI additionally reproduces Wilder's published worked example** to within
  0.07 — the rounding in his printed table.
- **Scale-invariance tests**: doubling every price leaves `macd_pct`, `atr_pct`,
  `bb_pct_b`, `bb_bandwidth` and all of ADX unchanged, and exactly doubles the
  raw counterparts.
- **No-lookahead tests** on all five blocks: perturbing bar 1500 leaves bars
  0–1499 bit-identical.
- **Warm-start convergence**: seeded and cold-start results agree to 1e-6 past
  bar 2000, so the seed genuinely washes out.
- **Behavioural**: ATR% and Bollinger bandwidth both separate March 2020 from
  June 2017 by >3x; `+DM`/`-DM` never both positive; `bb_break` agrees with
  `bb_pct_b`; ADX starts at bar 27 (2×period), ATR at 13, RSI at 14.

## Findings to carry into step 3

1. **15.0% of closes fall outside the 2σ bands**, against ~5% expected under
   normality. Returns are fat-tailed — an argument for the multi-state HMM over
   a single Gaussian, and worth stating explicitly in the report.
2. **ADX > 25 flags 62% of bars as trending.** Wilder calibrated that threshold
   on daily bars; on 15-minute data it barely discriminates. Tune it against the
   HMM's own regime labels rather than accepting it.
3. **RSI(14) is overbought on 17.9% of train bars vs oversold on 12.0%**,
   reflecting the net-upward window. RSI will lean bullish when regimes are
   labelled.
4. `atr_pctile_1000` reads 0.91 through March 2020 vs 0.38 in June 2017 — the
   percentile construction works, and it is the natural input to the HMM's
   high-volatility state.

## Still blocked on

No volume column, so OBV, VWAP deviation and volume-confirmed breakouts remain
unavailable. All five blocks above need only OHLC and are unaffected.
