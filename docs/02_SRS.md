# Software Requirements Specification (SRS)

## VeriAlpha — Regime-Aware Hierarchical RL Trading System with Backtest Integrity Certification

Version 1.0 · Based on IEEE 830 structure · MPSTME Capstone Project

---

## 1. Introduction

### 1.1 Purpose
This SRS specifies the functional and non-functional requirements of VeriAlpha, a hierarchical reinforcement-learning trading system for Indian equities that (a) separates strategic decision-making from trade execution, (b) sizes positions by model confidence including full abstention, and (c) certifies its own backtests against overfitting before any agent is promoted to live paper trading.

### 1.2 Scope
**In scope:** historical + live NSE equity/ETF data ingestion; feature engineering; HMM regime detection; two-level RL agent hierarchy; realistic Indian cost model; walk-forward training/evaluation; overfitting-rejection certification; paper trading via broker sandbox API; Streamlit dashboard.
**Out of scope:** real-money trading; order-book (L2) data; options/F&O strategies; multi-asset (crypto/FX) support; SEBI algo registration filing (design is compatible, filing is not performed).

### 1.3 Definitions and Abbreviations
| Term | Meaning |
|---|---|
| HLA / LLA | High-Level (strategic) Agent / Low-Level (execution) Agent |
| Regime | Latent market state (bull / bear / sideways / high-vol) estimated by HMM |
| Option | A temporally-extended strategy choice in the Options framework (trend, mean-reversion, defensive, breakout) |
| PBO | Probability of Backtest Overfitting (CSCV method) |
| DSR | Deflated Sharpe Ratio |
| Walk-forward fold | Rolling train/validation/test split preserving temporal order |
| Abstention | HLA action allocating 0% capital due to low confidence |
| Paper trading | Simulated order placement against live market prices via Kite sandbox |

### 1.4 References
See Synopsis §7 (HRT 2024; EarnHFT 2023; ELTRA 2025; Bailey & López de Prado 2014; arXiv:2209.05559; SEBI Feb-2025 circular; Moody & Saffell 2001).

---

## 2. Overall Description

### 2.1 Product Perspective
Standalone research system composed of six subsystems communicating through a shared database and typed Python interfaces:

1. **Data & Feature Service** (batch + streaming)
2. **Regime Detection Module**
3. **HRL Core** (HLA + LLA + Gymnasium environments + portfolio engine + cost model)
4. **Certification Pipeline** (training, validation, rejection gate)
5. **Paper-Trading Daemon**
6. **Integrity Dashboard**

### 2.2 User Classes
| User | Interaction |
|---|---|
| Student/Researcher | Runs training, configures experiments, reads MLflow logs |
| Evaluator (mentor/panel) | Views dashboard, inspects reports, verifies live logs |
| (Future) Retail trader / broker | Consumes certification report for a submitted strategy |

### 2.3 Operating Environment
- Windows 11 / Linux, Python 3.11, 16 GB RAM minimum; GPU optional (CUDA) for training speed-up.
- Internet access for yfinance/Kite APIs. SQLite in development; PostgreSQL for the live daemon.

### 2.4 Design Constraints
- Temporal integrity: no feature, scaler, or hyperparameter may use information from after its fold's training cutoff (leakage constraint C-1).
- LLA reward must exclude directional PnL (separation-of-objectives constraint C-2).
- All randomness seeded and logged for reproducibility (C-3).
- Paper trading only; no real-money order endpoints may be called (C-4).
- Rate limits: Kite Connect ≤ 3 req/s historical, websocket for ticks (C-5).

### 2.5 Assumptions and Dependencies
- Zerodha Kite Connect developer account available (or Fyers API as fallback).
- Daily bars sufficient for HLA; minute bars for LLA realism on a subset of symbols.
- India VIX and index data obtainable from NSE/yfinance.

---

## 3. Functional Requirements

### 3.1 Data & Feature Service (FR-D)
- **FR-D1** Ingest daily OHLCV for a configurable universe (default: NIFTY 50 constituents + 5 sector ETFs) from yfinance/NSE bhavcopy, ≥ 2015-present.
- **FR-D2** Ingest minute OHLCV for ≥ 5 liquid symbols via Kite historical API.
- **FR-D3** Compute indicators: RSI, MACD, EMA(20/50/200), ATR, ADX, Bollinger Bands, OBV, Stochastic RSI, VWAP deviation, realized vol (20/60d), rolling correlations; India VIX joined as macro feature.
- **FR-D4** Persist raw + feature data with schema versioning; incremental daily update job.
- **FR-D5** Fit all normalizations within each fold's training window only (enforces C-1); provide a leakage self-test that fails CI if violated.

### 3.2 Regime Detection (FR-R)
- **FR-R1** Fit a Gaussian HMM (k = 3–5 states) on returns + volatility features per fold training window.
- **FR-R2** Output per-bar regime probabilities and a change-point confidence score.
- **FR-R3** Map latent states to interpretable labels (bull/bear/sideways/high-vol) via posterior statistics.
- **FR-R4** Expose regime output as features to the HLA state and as the trigger signal for option termination β(s).

### 3.3 High-Level Strategic Agent (FR-H)
- **FR-H1** Act every N bars (configurable; default daily) or on regime-change trigger.
- **FR-H2** State: regime probabilities, realized vol, ADX, VIX level, avg pairwise correlation, portfolio state (allocation, unrealized PnL, drawdown, days-since-rebalance), trailing Sharpe of each option.
- **FR-H3** Action: discrete option ∈ {trend-following, mean-reversion, defensive/cash, breakout} × capital allocation ∈ {0, 10, 25, 50, 75}% × volatility budget.
- **FR-H4** Reward: differential Sharpe ratio over the decision window − λ·drawdown penalty − κ·option-switching cost.
- **FR-H5** Confidence-scaled sizing: allocation is multiplied by an ensemble-agreement score; below threshold τ the system must abstain (0% allocation). τ, λ, κ configurable.
- **FR-H6** Regime-conditioned termination: an active option terminates when regime-change confidence exceeds threshold or max duration is reached.
- **FR-H7** Trained with PPO (Stable-Baselines3); A2C as comparison.

### 3.4 Low-Level Execution Agent (FR-L)
- **FR-L1** Act every bar within the HLA's window; state includes remaining target position, time remaining, short-term vol, VWAP deviation, spread proxy, and the active option as conditioning input.
- **FR-L2** Action: order-size fraction of remaining target ∈ [0,1], wait action, stop-loss adjustment.
- **FR-L3** Reward: −slippage (vs. arrival/VWAP benchmark) − transaction cost − λ·inventory-risk penalty. Must not include directional PnL (C-2).
- **FR-L4** Pretrained in isolation on randomized target-execution tasks before joint training (warm start).
- **FR-L5** Implemented with SAC (continuous) and Dueling DDQN (discrete) for comparison.
- **FR-L6** Realized execution cost is fed back into the HLA's effective-return computation.

### 3.5 Portfolio Engine & Cost Model (FR-P)
- **FR-P1** Track cash, positions, unrealized/realized PnL, drawdown, exposure, turnover per bar.
- **FR-P2** Cost model: `cost = trade_value × spread_bps/10⁴ + trade_value × impact_coef × √(trade_size/ADV)` plus Indian charges (brokerage, STT, exchange txn, GST, stamp duty) as configurable schedule.
- **FR-P3** Enforce constraints: no leverage (v1), max single-position weight, volatility budget from HLA.

### 3.6 Certification Pipeline (FR-C)
- **FR-C1** Generate rolling walk-forward folds (train 3y / val 6m / test 6m, step 6m) with purge gap; expanding-window mode as robustness check.
- **FR-C2** Retrain/fine-tune both levels per fold; freeze policies during val/test.
- **FR-C3** Train an ensemble of N ≥ 5 seeds per configuration; store all runs in MLflow.
- **FR-C4** Compute PBO via combinatorially symmetric cross-validation; compute Deflated Sharpe Ratio accounting for number of trials.
- **FR-C5** **Rejection gate:** a candidate is promotable only if PBO < 0.5 threshold (configurable) and DSR significant at 95%; rejected candidates are logged with reasons.
- **FR-C6** Produce a standardized **Integrity Report** (HTML/PDF): per-fold equity curves, regime-stratified metrics, cost-sensitivity table (5/10/20 bps), PBO, DSR, bootstrap CIs, Jobson–Korkie comparison vs. baselines.

### 3.7 Paper-Trading Daemon (FR-T)
- **FR-T1** Subscribe to live prices (Kite websocket) for the trading universe during market hours.
- **FR-T2** Run inference with the promoted (certified) policy; place simulated orders in the sandbox; never call real-money endpoints (C-4).
- **FR-T3** Log every decision: timestamp, state snapshot, regime label, confidence, option, allocation, order, fill, cost.
- **FR-T4** Daily job computes live-vs-backtest divergence (return, slippage, hit-rate deltas) and appends to the divergence series.
- **FR-T5** Auto-recover from disconnects; alert on stale data (> 2 min without ticks during market hours).

### 3.8 Integrity Dashboard (FR-U)
- **FR-U1** Portfolio view: equity curve, positions, drawdown, exposure (live + backtest overlay).
- **FR-U2** Agent-brain view: current regime probabilities, active option, confidence score, allocation and the sizing rationale ("why this size / why abstaining").
- **FR-U3** Integrity view: deflated vs. raw metrics, rejected-candidate table, live-vs-backtest divergence chart, cost-sensitivity results.
- **FR-U4** Experiment browser: per-fold results, ablation tables, significance tests.

---

## 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | Reproducibility: any reported result reproducible from a config file + seed; CI runs a smoke-training job. |
| NFR-2 | Performance: full walk-forward training of one configuration completes in < 12 h on a single GPU / < 36 h CPU. |
| NFR-3 | Inference latency: daemon decision loop < 1 s per bar per symbol. |
| NFR-4 | Reliability: daemon uptime ≥ 95% of market hours over the evaluation period; automatic restart on crash. |
| NFR-5 | Data integrity: all trades and decisions durably logged (PostgreSQL, daily backup). |
| NFR-6 | Security: API keys in environment variables/secret store, never in the repo; sandbox-only endpoints (C-4). |
| NFR-7 | Code quality: pytest coverage ≥ 70% on portfolio engine, cost model, and fold generator (the correctness-critical paths); type hints on public interfaces. |
| NFR-8 | Usability: dashboard usable by a non-ML evaluator; every metric has a tooltip definition. |

---

## 5. External Interface Requirements

- **EI-1 yfinance / NSE bhavcopy:** daily OHLCV pull (batch, retry with backoff).
- **EI-2 Zerodha Kite Connect:** historical minute bars (REST), live ticks (websocket), sandbox order placement. OAuth login flow; daily session refresh per SEBI API norms.
- **EI-3 MLflow tracking server** (local): parameters, metrics, artifacts per run.
- **EI-4 PostgreSQL/SQLite:** schemas `market_data`, `features`, `decisions`, `orders`, `runs`, `divergence`.
- **EI-5 Streamlit app:** read-only over the database and MLflow store.

---

## 6. Acceptance Criteria (mapped to objectives)

| # | Criterion | Verifies |
|---|---|---|
| AC-1 | Full hierarchy trains end-to-end and beats flat PPO baseline on risk-adjusted metrics in ≥ 60% of walk-forward folds | Objective 1, 5a |
| AC-2 | Abstention configuration shows lower MaxDD and CVaR than always-invested configuration, with Jobson–Korkie significance on Sharpe reported | Objective 2, 5c |
| AC-3 | Certification pipeline demonstrably rejects ≥ 1 overfit candidate (good raw Sharpe, PBO ≥ threshold) with report artifact | Objective 3 |
| AC-4 | Paper-trading daemon runs ≥ 3 consecutive weeks with ≥ 95% market-hour uptime and produces divergence report | Objective 4 |
| AC-5 | Ablation tables (hier vs. flat; ±regime conditioning; ±abstention) with bootstrap CIs included in final report | Objective 5 |
| AC-6 | Leakage self-test passes; all results reproducible from config + seed | NFR-1, C-1 |
