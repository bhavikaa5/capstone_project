# VeriAlpha — Complete System Pipeline

End-to-end technical pipeline: data → features → regimes → hierarchical agents → portfolio → certification → live paper trading → dashboard. This document is the implementation blueprint; requirement IDs refer to the SRS.

---

## 0. Repository Layout

```
verialpha/
├── configs/                  # YAML experiment configs (single source of truth)
│   ├── data.yaml             # universe, date ranges, sources
│   ├── env.yaml              # cost model, constraints
│   ├── agents/hla_ppo.yaml   # per-agent hyperparameters
│   ├── agents/lla_sac.yaml
│   └── certify.yaml          # fold spec, PBO/DSR thresholds, ensemble size
├── src/verialpha/
│   ├── data/                 # ingestion, indicators, leakage-safe scalers
│   ├── regimes/              # HMM fit/predict, labeling, change-point score
│   ├── envs/                 # Gymnasium envs: ExecutionEnv, StrategyEnv
│   ├── agents/               # PPO/SAC/DDQN wrappers, ensemble utilities
│   ├── portfolio/            # engine, cost model, constraints
│   ├── certify/              # folds, PBO, DSR, rejection gate, report builder
│   ├── live/                 # Kite daemon, decision logger, divergence job
│   └── dashboard/            # Streamlit app
├── tests/                    # pytest: engine, cost model, folds, leakage test
├── scripts/                  # train.py, evaluate.py, certify.py, run_daemon.py
└── notebooks/                # exploration only — nothing load-bearing
```

Rule: **notebooks are for exploration only.** Anything used in a result lives in `src/` with a config and a test.

---

## 1. Data & Feature Pipeline (FR-D)

### 1.1 Ingestion
```
yfinance / NSE bhavcopy ──► raw_daily  (2015 → present, NIFTY 50 + 5 sector ETFs)
Kite historical REST    ──► raw_minute (5 liquid symbols, rolling 2 years)
NSE / yfinance          ──► india_vix, nifty_index
```
- Daily incremental job (`scripts/update_data.py`): fetch → validate (no gaps, no negative prices, split/bonus adjustment check) → upsert.
- Every table carries `ingested_at` and `source` columns for auditability.

### 1.2 Feature Engineering
Per symbol, per bar: RSI(14), MACD(12,26,9), EMA(20/50/200) ratios, ATR(14)/price, ADX(14), Bollinger %B and bandwidth, OBV slope, Stoch-RSI, VWAP deviation (minute data), realized vol 20/60d, rolling 60d avg pairwise correlation of the universe, India VIX level and 20d change.

### 1.3 Leakage-Safe Normalization (C-1 — the #1 failure mode)
- Scalers (`RollingZScore`, `ExpandingMinMax`) are **fit on the fold's training window only**, then applied frozen to val/test.
- `tests/test_leakage.py`: shifts the test period into training, asserts feature values change — if a feature is invariant, something is peeking. CI fails on violation.

---

## 2. Regime Detection (FR-R)

```
returns + realized vol + VIX ──► GaussianHMM(k∈{3,4,5}, full cov) fit on fold-train
                              ──► per-bar state posteriors p(s|x)
                              ──► change-point confidence = 1 − max_s p(s)  (+ posterior entropy)
                              ──► label map: state → {bull, bear, sideways, high-vol}
                                  by (mean return, vol) of each state's bars
```
- Model selection by BIC on the training window; k fixed per fold thereafter.
- Outputs feed three consumers: HLA state vector, option-termination trigger β(s), and the regime-stratified evaluation splits.

---

## 3. Hierarchical RL Core

### 3.1 Two Gymnasium Environments

**`ExecutionEnv`** (per-bar, for the LLA):
- Episode = one HLA decision window with a target position to reach.
- `obs`: remaining target, time remaining, short-term vol, VWAP dev, spread proxy, active option one-hot.
- `action`: Box[0,1] order-size fraction (+ discrete wait / stop-loss adjust in DDQN variant).
- `reward = −slippage_vs_arrival − txn_cost − λ·inventory_risk` — **no directional PnL** (C-2).

**`StrategyEnv`** (per-window, for the HLA):
- Wraps `ExecutionEnv` + portfolio engine: an HLA step unrolls one full execution episode inside.
- `obs`: FR-H2 state vector (regime posteriors, vol, ADX, VIX, correlation, portfolio state, trailing per-option Sharpe).
- `action`: MultiDiscrete [option(4), allocation(5: 0/10/25/50/75%), vol-budget(3)].
- `reward = differential_sharpe(window) − λ·drawdown_penalty − κ·1[option_switched]`.
- Option termination: episode-internal β(s) fires on regime-change confidence > θ or max duration.

### 3.2 Cost Model (inside both envs, FR-P2)
```python
def transaction_cost(trade_value, trade_size, adv,
                     spread_bps=5, impact_coef=0.1):
    fixed  = trade_value * spread_bps / 1e4
    impact = trade_value * impact_coef * sqrt(trade_size / adv) if adv else 0
    charges = indian_charges(trade_value)   # brokerage, STT, exch txn, GST, stamp
    return fixed + impact + charges
```
Sensitivity runs at 5/10/20 bps are part of every certification report.

### 3.3 Training Order (avoids the moving-target problem)
1. **Warm-start LLA**: train on randomized target-execution tasks against the cost model; validate it beats naive TWAP execution on slippage.
2. **Train HLA** with the frozen LLA in the loop.
3. **Joint fine-tune** both (lower learning rates), optional.
4. **Confidence layer**: train N-seed HLA ensemble; agreement score `a ∈ [0,1]` multiplies allocation; `a < τ` ⇒ abstain (allocation 0). τ tuned on validation folds only.

---

## 4. Certification Pipeline (FR-C) — the differentiator

```
configs/certify.yaml
        │
        ▼
┌─ Fold Generator ────────────────────────────────────────────┐
│ rolling: train 3y → purge gap 5d → val 6m → test 6m, step 6m│
│ (expanding-window variant as robustness check)              │
└──────────────┬──────────────────────────────────────────────┘
               ▼   per fold × per config × N seeds
┌─ Train & Evaluate ──────────────────────────────────────────┐
│ retrain both levels on fold-train · tune on fold-val ·      │
│ FROZEN policy on fold-test · all runs → MLflow              │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─ Statistics ────────────────────────────────────────────────┐
│ • PBO via CSCV over the run matrix                          │
│ • Deflated Sharpe Ratio (accounts for #trials)              │
│ • Block-bootstrap CIs (1000 resamples) on Sharpe            │
│ • Jobson–Korkie (Memmel) vs. each baseline                  │
│ • Regime-stratified metric table (bull/bear/sideways/hi-vol)│
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─ Rejection Gate ────────────────────────────────────────────┐
│ PROMOTE  iff  PBO < 0.5  AND  DSR p < 0.05                  │
│ else REJECT → logged with reasons (this is a *result*)      │
└──────────────┬──────────────────────────────────────────────┘
               ▼
      Integrity Report (HTML/PDF) + promoted policy artifact
```

Baselines evaluated through the identical pipeline: buy-and-hold, MA-crossover, risk-parity rebalance, flat PPO, hierarchy-without-regime-conditioning (fixed-interval switching), full system ± abstention.

---

## 5. Live Paper-Trading Daemon (FR-T)

```
market open ──► Kite websocket ticks ──► bar aggregator ──► feature update
                                                        │
                              promoted policy inference ◄┘
                                        │
                        decision logged (state, regime, confidence,
                        option, allocation, rationale)
                                        │
                              sandbox order ──► simulated fill + cost
                                        │
                 nightly: divergence job (live vs. backtest deltas:
                 return, slippage, hit rate, regime mix) ──► DB
```
- Systemd/Task-Scheduler managed; auto-restart; stale-data alarm (> 2 min without ticks).
- Hard guard in the API client: only sandbox endpoints are reachable (C-4); real order routes raise.

---

## 6. Dashboard (FR-U)

Streamlit, three tabs reading the DB + MLflow:
1. **Portfolio** — live equity curve overlaid on backtest expectation; positions; drawdown; exposure.
2. **Agent Brain** — regime posterior bar, active option, confidence gauge, allocation, and a plain-English rationale line ("Abstaining: regime confidence 0.41 < τ=0.6").
3. **Integrity** — raw vs. deflated Sharpe side-by-side, rejected-candidates table with reasons, live-vs-backtest divergence series, cost-sensitivity table, ablation results.

---

## 7. Experiment Matrix (drives Chapter 4 of the report)

| ID | Configuration | Answers |
|---|---|---|
| E1 | Buy-and-hold / MA-crossover / risk-parity | Classical baselines |
| E2 | Flat PPO (single agent, same features) | Is the hierarchy worth it? |
| E3 | Hierarchy, fixed-interval switching | Value of regime conditioning |
| E4 | Hierarchy, regime-conditioned (full) | Main system |
| E5 | E4 − abstention (always ≥10% allocated) | Value of abstain policy |
| E6 | E4 with cost 5/10/20 bps | Cost-model sensitivity |
| E7 | E4 expanding vs. rolling windows | Adaptation robustness |

Each cell: mean ± std across folds and seeds; per-regime breakdown; significance vs. E2.

---

## 8. Definition of Done (per component)

| Component | Done means |
|---|---|
| Data pipeline | Daily job green 7 days straight; leakage test passing in CI |
| ExecutionEnv + LLA | Beats TWAP slippage benchmark on held-out execution tasks |
| Regime module | Labeled states match known regime periods (2020 crash, 2022 bear) on inspection |
| HLA + hierarchy | E4 completes full walk-forward; results logged in MLflow |
| Certification | Report artifact generated; ≥1 candidate rejected with documented reason |
| Daemon | 3 consecutive weeks live, ≥95% market-hour uptime, divergence report produced |
| Dashboard | Mentor can navigate all three tabs unassisted |
