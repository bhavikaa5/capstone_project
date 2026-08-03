# VeriAlpha — How Each Part Works, and Why We Chose These Techniques

Defense document for Review 1. For each of the four parts: **how it works**, **why this technique**, and **why not the alternatives**.

---

# PART 1 — High-Level Strategic Agent (PPO)

## How it works

1. Once per decision window (default: daily, or when the regime changes), it receives a **state vector**:
   - Regime probabilities from the HMM (e.g. bull 0.7, sideways 0.2, bear 0.1)
   - Realized volatility (20d/60d), ADX trend strength, India VIX
   - Portfolio state: current allocation, unrealized P&L, current drawdown, days since last change
   - Trailing Sharpe of each candidate strategy
2. It outputs three things together:
   - **Which strategy** to run: trend-following / mean-reversion / defensive-cash / breakout
   - **How much capital**: 0%, 10%, 25%, 50%, 75%
   - **Volatility budget** the execution agent must respect
3. It gets a **reward** at the end of the window:
   `reward = differential Sharpe − λ·(drawdown) − κ·(1 if strategy was switched)`
   The switching penalty κ is what stops it from flip-flopping strategies every day.
4. It does **not** place orders. It hands a target position to the low-level agent.

## Why PPO

| Reason | Explanation |
|---|---|
| **Action space fits** | Our action is discrete/multi-discrete (pick 1 of 4 strategies × 1 of 5 allocations). PPO handles discrete and MultiDiscrete natively. |
| **Stability under noisy rewards** | Market returns are extremely noisy. PPO's clipped objective limits how far the policy can move in one update, so one lucky or unlucky batch can't destroy the policy. This is the single biggest reason. |
| **Gives a probability distribution** | PPO outputs a *stochastic* policy — a probability over strategies. We need this: our confidence/abstention logic reads the spread of that distribution across the ensemble. A value-based method gives us numbers, not a calibrated distribution. |
| **Parallel rollouts** | The high level acts only once per day, so real decision data is scarce (~250/year). PPO is designed for vectorized parallel environments — we run many symbols and time-slices simultaneously to generate enough experience. |
| **Reproducibility** | PPO is the most hyperparameter-tolerant modern policy-gradient method and is well-tested in Stable-Baselines3. In a one-semester project we cannot afford to spend weeks stabilizing an algorithm. |

## Why not the alternatives

| Alternative | Why we rejected it |
|---|---|
| **DQN / DDQN at high level** | Value-based. Our combined action space (4 strategies × 5 allocations × 3 vol budgets = 60 combinations) blows up as a flat Q-table output, and it gives a deterministic argmax instead of the action distribution our confidence measure needs. |
| **A2C** | Same actor-critic family but without PPO's clipping → higher variance and less stable on noisy financial reward. We keep it as a *comparison baseline*, not the main method. |
| **SAC at high level** | SAC is built for continuous actions. Discrete-SAC variants exist but are less mature. Wrong tool for a categorical strategy choice. |
| **Supervised classifier ("just predict the best strategy")** | There is no ground-truth label for "correct strategy" at each time step. And it would ignore sequential effects — switching costs, current drawdown, existing position. Strategy choice is a decision problem, not a labeling problem. |
| **Rule-based switching (if VIX > X then defensive)** | This is our *baseline*, not our method. It cannot learn allocation sizing or adapt thresholds; it's what we must beat. |

---

# PART 2 — Low-Level Execution Agent (SAC / Dueling DDQN)

## How it works

1. Receives a **target position** from the high-level agent (e.g. "reach 25% allocation in RELIANCE this window").
2. Every bar inside the window it observes:
   - Remaining quantity still to fill, time remaining in the window
   - Short-term volatility, VWAP deviation, spread proxy, recent volume
   - The active strategy option, passed down as conditioning input
3. Outputs **how much of the remaining target to execute right now** (a fraction 0–1), plus wait/stop-loss handling.
4. Reward: `−slippage (vs arrival price) − transaction cost − λ·inventory risk`
   **Profit is deliberately excluded.** Direction is the high level's job; this agent is only graded on execution quality.
5. **Training order:** warm-start it alone on randomized "fill this target" tasks until it beats TWAP, *then* train the high level on top of it. This avoids both levels learning against a random moving target.

## Why SAC (continuous version)

| Reason | Explanation |
|---|---|
| **Order sizing is naturally continuous** | "Execute 37% of the remainder now" is a real, meaningful action. Forcing it into 5 buckets throws away precision. |
| **Sample efficiency** | SAC is off-policy with a replay buffer, so every execution step is reused many times. Execution generates many short episodes — replay makes training far cheaper than on-policy methods. |
| **Entropy bonus prevents collapse** | SAC's maximum-entropy objective explicitly rewards keeping the policy varied. Without it, execution agents collapse to a degenerate policy ("dump everything in bar 1"). This keeps it exploring different timing schedules — genuinely valuable here. |
| **Robust in practice** | SAC is far less hyperparameter-sensitive than DDPG/TD3, which matters on our timeline. |

## Why Dueling DDQN (discrete version, run as comparison)

| Reason | Explanation |
|---|---|
| **Dueling matches the problem structure** | Dueling nets split *state value* from *action advantage*. In execution, most states are ones where the action barely matters (lots of time left, calm market) and a few are critical (window closing, price moving). Dueling learns exactly this distinction. |
| **Double-Q fixes overestimation** | Plain DQN overestimates action values. In execution, overestimating how good aggressive trading is → the agent over-trades and pays more cost. Double-Q directly corrects this. |
| **Discrete is auditable** | Bucketed actions (0/25/50/75/100% of remaining) are easier to constrain and explain to a regulator or examiner. |
| **Gives us a real ablation** | Continuous vs discrete execution becomes a reported experiment, not an unexplained choice. |

## Why not the alternatives

| Alternative | Why we rejected it |
|---|---|
| **DDPG** | Deterministic policy, notoriously brittle and hyperparameter-sensitive, weak exploration. SAC supersedes it. |
| **TD3** | Better than DDPG, but still no entropy term; SAC generally matches or beats it with less tuning. |
| **PPO at low level** | On-policy → discards data after each update. Execution has many steps and we specifically want replay-based sample reuse. Also weaker at fine-grained continuous control. |
| **TWAP / VWAP classical execution** | These are our **benchmark**, not our method. Our acceptance criterion is that the learned agent beats TWAP on slippage — if it doesn't, we report that honestly. |

---

# PART 3 — Certification Pipeline (the main contribution)

## How it works

```
1. Rolling walk-forward folds  → train 3y | purge gap | validate 6m | test 6m | step 6m
2. For each config, train N ≥ 5 seeds        → an ensemble, not a single lucky run
3. Freeze policy on the test fold            → no gradient updates, ever, on test data
4. Compute statistics on the resulting matrix of trials:
      • PBO   (Probability of Backtest Overfitting, via CSCV)
      • DSR   (Deflated Sharpe Ratio)
      • Block-bootstrap confidence intervals on Sharpe
      • Jobson–Korkie (Memmel-corrected) vs each baseline
5. REJECTION GATE: promote only if PBO < 0.5 AND DSR significant at 95%
      → rejected models are logged with reasons. A rejection is a RESULT, not a failure.
```

## Why these specific statistics

| Technique | Why it |
|---|---|
| **PBO via CSCV** | It measures exactly our failure mode: *what is the probability that the model which looked best in-sample is actually below-median out-of-sample?* It is symmetric, distribution-free, and it operates on the trial matrix our walk-forward already produces. Nothing else answers this question so directly. |
| **Deflated Sharpe Ratio** | A Sharpe of 1.5 means nothing if you tried 200 configurations — someone always wins by luck. DSR corrects Sharpe for the **number of trials**, plus **skewness, fat tails, and sample length**. Financial returns violate normality, and plain Sharpe assumes it. |
| **Rolling walk-forward with purge gap** | Time-series data must be tested strictly forward in time. The purge gap removes bars adjacent to the split so overlapping indicator windows can't leak information across the boundary. |
| **Block bootstrap** | Returns are autocorrelated and volatility clusters. Standard i.i.d. bootstrap destroys that structure and gives falsely narrow confidence intervals. Block bootstrap resamples *chunks*, preserving it. |
| **Jobson–Korkie + Memmel** | The standard test for whether two Sharpe ratios genuinely differ when the two strategies' returns are **correlated** — which ours are, since they trade the same assets. A naive t-test would be wrong here. |

## Why not the alternatives

| Alternative | Why we rejected it |
|---|---|
| **Single train/test split** | One split means one lucky period decides everything. This is precisely the practice our project criticizes. |
| **Standard k-fold cross-validation** | It shuffles time, so the model trains on the future and tests on the past. Invalid for financial time series. |
| **Just reporting Sharpe / returns** | Ignores selection bias across trials — this *is* the overfitting problem, so reporting it alone would defeat the project's purpose. |
| **White's Reality Check / Hansen's SPA** | Legitimate and complementary — they test whether the best strategy beats a benchmark after accounting for data snooping. We chose PBO because it answers the *selection procedure's* overfitting probability directly, and it composes naturally with walk-forward. Worth citing as future extension. |

---

# PART 4 — Live Paper Trading + Integrity Dashboard

## How it works

```
Kite websocket ticks → aggregate into bars → compute features → policy inference
   → simulated (sandbox) order → fill + realistic cost applied
   → log EVERY decision: timestamp, state, regime, confidence, strategy, size, rationale
Nightly job → compare live vs backtest for the same period:
   return delta · slippage delta · hit-rate delta · regime mix delta
   → this divergence series is a primary result of the project
```

The Streamlit dashboard has three tabs: **Portfolio** (live equity vs backtest expectation), **Agent Brain** (current regime, confidence, why this size / why abstaining), **Integrity** (raw vs deflated metrics, rejected models, divergence chart).

## Why this design

| Choice | Reason |
|---|---|
| **Live paper trading at all** | Our entire thesis is "backtests lie." We cannot prove that with another backtest. Running live and measuring the gap is the only honest evidence — and almost no published paper in our literature review has it. |
| **Paper, never real money** | SEBI's 2026 retail-algo framework requires broker partnership and algo registration for real deployment — out of scope. It's also the ethical choice for a student project. Enforced as a hard code guard, not a promise. |
| **Zerodha Kite Connect** | Largest Indian retail broker with a proper documented API, websocket feed, and a sandbox. Indian market focus matches our dataset. Fyers API is our fallback if approval is delayed. |
| **Logging confidence + rationale with every decision** | Makes the system explainable. We can show the panel *why* it abstained on a given day, which is impossible if you only log orders. |
| **Streamlit over React** | Pure Python, no separate frontend stack, days instead of weeks. We're graded on the ML system, not on frontend engineering. React is future scope if this becomes a product. |

---

# Quick-Answer Table (memorize this for the viva)

| Component | Chosen | Rejected | One-line reason |
|---|---|---|---|
| High-level agent | **PPO** | DQN, A2C, SAC | Discrete actions + stable under noisy rewards + gives action distribution for confidence |
| Low-level agent | **SAC** (+ Dueling DDQN compare) | DDPG, TD3, PPO | Continuous order sizing + replay efficiency + entropy stops policy collapse |
| Regime detection | **Gaussian HMM** | LSTM, Transformer, K-Means | Interpretable states, data-efficient, fast — deep models need more data than we have |
| Overfitting test | **PBO + Deflated Sharpe** | Plain Sharpe, k-fold CV | Directly measures selection bias; corrects for number of trials and fat tails |
| Validation | **Rolling walk-forward + purge** | Single split, k-fold | Tests strictly forward in time; purge prevents leakage at boundaries |
| Significance | **Jobson–Korkie + block bootstrap** | t-test, i.i.d. bootstrap | Handles correlated strategies and autocorrelated returns |
| Execution benchmark | **TWAP** | — | Standard industry baseline our agent must beat |
| Deployment | **Kite sandbox paper trading** | Real money | SEBI constraint + ethics; still gives real live-vs-backtest evidence |
| Dashboard | **Streamlit** | React | Python-native, fast to build, we're graded on ML not frontend |

---

# The single most important framing

If a panel member asks *"why not just use the newest/biggest model?"*, the answer is:

> "Every choice we made optimizes for **reproducibility and honest measurement**, not for maximum complexity. Our contribution is not a bigger model — it's proving that a model's results are real. A heavier architecture we can't train reliably in one semester would actively undermine that claim."

That answer works for PPO over exotic RL, HMM over Transformers, and Streamlit over React — all at once.
