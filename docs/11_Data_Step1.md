# Step 1 — Nifty 50 Market Data

First stage of the pipeline (top box of the flowchart). Takes the two delivered
CSVs and produces a clean, audited, tz-aware 15-minute OHLC series that steps
2-8 can rely on without re-checking.

```bash
python scripts/01_prepare_data.py
python -m pytest tests/ -q
```

## Files

| Path | Role |
|---|---|
| `src/config.py` | Paths, NSE session constants, column groups |
| `src/data/loader.py` | `load_raw` / `clean` / `assert_no_leakage` / `load_clean` |
| `scripts/01_prepare_data.py` | Runs the stage, writes parquet + report + figure |
| `tests/test_data_step1.py` | 16 invariants guarding the cleaned output |
| `data/interim/nifty50_{train,test}.parquet` | Stage output |
| `reports/01_data_quality.md` | Generated audit |

## What the raw data actually is

NSE Nifty 50 **index** bars, 15 minutes, `09:15`-`15:15` (25 bars/session).
Columns: `date`, `time`, `close/open/high/low`, plus vendor-computed `pivot`,
`200_EMA`, and three Supertrend variants.

| | Train | Test |
|---|---|---|
| Raw rows | 57,478 | 6,014 |
| Clean rows | **55,586** | **5,810** |
| Sessions | 2,226 | 234 |
| Range | 2015-04-23 → 2024-04-30 | 2024-05-02 → 2025-04-08 |

The split is already chronological with no overlap — asserted in code, not assumed.

## Cleaning decisions and the evidence for them

**1. Duplicate timestamps — 1,850 rows dropped (train), 200 (test).**
74 dates in train and 8 in test appear *twice*, as two consecutive 25-bar blocks.
These are chunk boundaries from the vendor's export (roughly one per month), not
random noise. Between the two copies `close` is bit-identical and `open/high/low`
match on most bars. The decider is continuity: the **second** copy's first open
sits ~95 index points away from the previous session's close, versus ~1.3 points
for the first copy. The first copy is the one that chains correctly, so
`keep="first"`. Rows are sorted with a *stable* sort so this ordering survives.

**2. Special sessions — 42 rows dropped (train), 4 (test).**
Ten dates carry only 4-5 bars, timed 17:30-19:15: Muhurat (Diwali) sessions plus
the 24-Feb-2021 special live session. They have no regular-session bars on the
same date. Keeping them would break the fixed 25-bar session grid, inject
artificial overnight gaps into the indicators, and hand the HMM a handful of
low-liquidity bars to model as their own volatility regime.

**3. Timestamps.** `date` + `time` merged into one `timestamp`, localised to
`Asia/Kolkata` (the raw `+05:30` offset is constant). Every bar is verified to
sit on the 15-minute grid.

**4. Vendor indicators kept but quarantined.** Renamed `src_pivot`,
`src_200_EMA`, `src_Supertrend(...)`. Their warm-up history is unknown and the
two copies of the duplicated bars disagree on `200_EMA` by up to 1%, which means
they were computed per-chunk rather than over the full series. **Step 2 recomputes
every indicator from OHLC.** The `src_*` columns are reference only — do not feed
them to a model.

**5. Session bookkeeping added.** `session_date`, `bar_of_day` (0-24),
`is_session_open`, `is_session_close`. Step 2 needs these so no rolling window
silently spans an overnight gap.

## Known characteristics (not defects — carried forward)

- **No volume.** The flowchart says OHLCV; the delivered data is OHLC only. Any
  volume-dependent feature (OBV, VWAP deviation, volume-confirmed breakouts) is
  off the table unless a volume series is sourced separately. RSI, MACD, ATR,
  ADX and Bollinger — the five named in step 2 — need only OHLC, so step 2 is
  unaffected.
- **64 missing intraday bars in train, 40 in test**, across 4 and 2 short
  sessions. Mostly real: 2015-04-23 is a 1-bar partial first day, 2024-03-02 was
  the 7-bar special Saturday session, 2025-04-08 is a 3-bar partial final day.
  They are left as holes rather than forward-filled — synthetic bars would
  fabricate returns the HMM would read as genuine low-volatility states.
- **Largest single-bar move: 829 bps (train), 436 bps (test).** Both plausible
  gap-opens, not corrupt prints; the test asserts nothing exceeds 10%.
- Zero nulls, zero OHLC-consistency violations, zero non-positive prices.

## Open question for step 2

Test coverage ends **2025-04-08**, roughly 16 months before the current date. If
the live-validation stage is to run against Kite Connect, the gap between the
end of the test set and go-live will need either a data top-up or an explicit
statement that the held-out period is fixed at 2024-05 → 2025-04.
