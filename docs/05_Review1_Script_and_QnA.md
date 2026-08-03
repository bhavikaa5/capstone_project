# VeriAlpha — Review 1 Script + Panel Questions (Simple Version)

**Group 17** · Bhavika Bhojwani · Lakshya Sitlani · Vishal Singh
Goal: full 10/10 → **A. Problem + Need + Novelty + Objectives (5)** and **B. Feasibility with constraints (5)**.

---

## 0. First, understand how marks are given

The panel checks only two things. So make both very clear:

- **Row A (5 marks):** They want a clear problem, why it matters, **what is new in your idea**, and **clear objectives**. So you must actually *say the word "new" or "novelty"* and show the gap you found in existing papers. Don't expect them to guess.
- **Row B (5 marks):** They want to see that you thought about **all the constraints** (limits) and can still finish it. So you must actually *say the word "constraints"* and list them: time, data, computer power, SEBI rules, and correctness. Then say how you handle each one.

Simple rule:
- Every time you say "this is new" → show which paper is missing it.
- Every time you say "this is doable" → name the problem and your backup plan.

---

## 1. Who speaks what (around 8–9 minutes)

| Slides | Speaker | What to cover | Time |
|---|---|---|---|
| 1 Title, 2 Overview | **Bhavika** | Problem + our solution | ~2.5 min |
| 3 Objective, 4–6 Literature | **Lakshya** | Objectives + what is new | ~3.5 min |
| 7–8 Tools, 9 Plan, 12 Thanks | **Vishal** | Tools + constraints + timeline | ~2.5 min |

The person not speaking should be ready to answer questions on their part. Decide **now** who answers which type of question (see section 4).

---

## 2. The script (speak like this)

### SLIDE 1 — Title (Bhavika)
> "Good morning. We are Group 17 — Bhavika, Lakshya and Vishal. Our project is **VeriAlpha — a regime-aware AI trading system with reliable backtesting**.
>
> In one line: most AI trading models look very profitable on old data, but lose money in the real market. This happens because the model just memorised the past. VeriAlpha is built to be the version you can actually trust. It changes its strategy based on market conditions, and it checks its own results before we believe them."

*This gives the problem and your special point in 20 seconds. Panels like a clear opening line.*

### SLIDE 2 — Overview (Bhavika)
> "**The problem.** The market is not one single environment. It keeps changing between bull, bear, sideways and high-volatility phases. A strategy that earns well in a bull market can fail badly in a crash. But most reinforcement learning trading systems use only **one agent** to do everything, and they test it on only **one backtest**. Research shows this causes overfitting — the model memorises history instead of learning how to trade. So it works on paper but fails in real trading.
>
> **Our solution has four parts.**
> First, a **high-level agent** trained with PPO. It looks at the market, understands which phase we are in, and decides the strategy and how much money to risk.
> Second, a **low-level agent** using SAC or DDQN. It does the actual buying and selling, and tries to reduce slippage and cost.
> Third — and this is our main point — a **certification pipeline**. It statistically tests our own models and rejects the ones that are overfitted, before we use them.
> Fourth, a **live paper trading dashboard**, which compares real-time performance with the backtest, so we can see if the model is really behaving the way it promised.
>
> So basically, we separate *thinking* from *doing*, and we don't trust any result until it passes proper testing."

*This one slide covers problem, need and your new idea. Most of Row A comes from here.*

### SLIDE 3 — Objectives (Lakshya)
> "From this, we have four clear objectives.
>
> **One — build the hierarchical system.** A PPO agent for choosing the strategy, and a SAC or DDQN agent for executing trades, on Indian markets — the NIFTY stocks.
>
> **Two — confidence-based risk management.** The system decides how much capital to use. And most importantly, if it does not recognise the current market condition, it puts **zero** money — it simply stays out. Knowing when *not* to trade is also a result for us.
>
> **Three — make sure the model is reliable.** We use walk-forward testing, Probability of Backtest Overfitting, and Deflated Sharpe Ratio, specially to catch overfitting.
>
> **Four — proper evaluation.** We compare against normal deep RL models using risk-adjusted returns, and we check whether the improvement is statistically real or just luck.
>
> Every objective has a deliverable, which you will see in our plan."

*You literally said "four clear objectives" — that is what the top rubric band asks for.*

### SLIDES 4–6 — Literature Review (Lakshya) — **this is where your "new idea" marks come from**
> "We studied fifteen recent papers. Instead of reading every row, let me explain the pattern we found.
>
> The strong hierarchical RL papers — HRT, the AAAI 'Commission Fee' paper, EarnHFT and MacroHFT — all prove that separating strategy from execution works. **But** they have two common gaps. First, they work on portfolio weights or simple buy/sell direction. **None of them connects a proper, separate regime classifier to the RL agents.** EarnHFT and MacroHFT come closest, but they are made for crypto high-frequency trading and use very heavy designs — like a pool of hundreds of agents, or memory-augmented networks — which we cannot realistically build in one semester.
>
> On the other side, the regime detection papers — the Gaussian HMM papers and the regime-aware LightGBM paper — do very good regime detection, **but they stop there.** They only detect the regime. There is no RL agent connected to it.
>
> And paper 14, *When Valid Signals Fail*, is very important for us. It is a 2026 paper which shows that if you simply throw regime features into an RL model, the model actually breaks. But that paper only points out the problem — it gives no solution.
>
> **So the gap, and our novelty, is exactly this:** nobody has connected a **supervised HMM regime classifier** to a **light two-level RL system**, on **Indian equity markets**, and then **certified the whole thing against overfitting**. That combination is empty in current research. That is what VeriAlpha is building."

*This is the most important 60 seconds of your whole review. Say it slowly. Say the word "novelty" clearly.*

### SLIDES 7–8 — Tools and Techniques (Vishal) — feasibility starts here
> "Now about feasibility. First the tools — and every tool we chose is the **simple, proven** one, not the fanciest one.
>
> We use Python 3.11. For RL, we use Stable-Baselines3, which already has tested PPO and SAC, plus our own Dueling DDQN in PyTorch, all inside Gymnasium environments. So we are not writing algorithms from scratch, we are combining ready ones.
>
> For regime detection we use a Gaussian HMM from hmmlearn. We chose it over a heavy deep learning model because it is fast, it needs less data, and we can actually understand what each regime means.
>
> Data is free — yfinance and NSE bhavcopy for history, and Zerodha Kite Connect for minute data and live paper trading. No paid data subscription needed. For features we use Pandas and the `ta` library for indicators like RSI, MACD, ATR and so on.
>
> The part we take most seriously is validation — walk-forward testing with purged folds, Probability of Backtest Overfitting, Deflated Sharpe Ratio, and the Jobson–Korkie significance test. And we use a realistic Indian cost model with brokerage, STT and market impact, so our results are not fake zero-cost results. Streamlit for the dashboard, Git and pytest for clean development."

### SLIDE 9 — Plan / Gantt (Vishal) — **say the constraints clearly here**
> "This is our 15-week plan, matched with the three reviews. Let me clearly mention our **constraints**, because feasibility means accepting the limits honestly.
>
> - **Time constraint** — we have only one semester. So we work in phases. First data and a simple baseline model, then the regime module, then the full hierarchy, then the certification part. Every phase gives us a working output, so we are never left with nothing.
> - **Data and API constraint** — Kite Connect takes time for developer approval, so we are applying in the first week itself. If it is delayed, our backup is the Fyers API, or running the system on delayed data.
> - **Computing constraint** — training takes heavy computation. So we use daily data for the strategy agent to keep it light, limit ourselves to around 20–50 liquid stocks, and use free GPU whenever we need speed.
> - **Regulatory constraint** — because of SEBI's 2026 algo trading rules, we stay strictly on **paper trading**. No real money at all. This is blocked in our code itself, not just a promise.
> - **Correctness constraint** — the most common mistake in student trading projects is look-ahead bias, meaning the model accidentally sees future data. So we calculate everything only from training data, and we have an automatic test that fails our build if this rule is broken.
>
> Also, the riskiest part — the live system — we are starting in week 12, not week 15. This way we collect real live data before the final demo. That is why we are confident this project can be completed in one semester."

*This slide alone is your 5 marks for Row B. Five constraints, five solutions.*

### SLIDE 12 — Thank You (Vishal)
> "To summarise — VeriAlpha separates strategy from execution, adapts to changing market conditions, and most importantly, checks its own reliability before trusting any result. Thank you. We are happy to take questions."

---

## 3. Two lines you MUST say (mark insurance)

Even if you forget everything else, say these:

1. **For novelty:** *"Our novelty is connecting a supervised regime classifier with a light two-level RL system on Indian markets, along with a built-in overfitting certification pipeline — this combination does not exist in current research."*
2. **For feasibility:** *"We have considered five constraints — time, data and API, computation, SEBI rules, and correctness — and each one has a proper solution. That is why one semester is realistic for us."*

---

## 4. Questions the panel may ask (with simple answers)

Decide who answers what. Each answer should take 20–40 seconds.

### A. Basic RL questions
**Q1. Why reinforcement learning? Why not normal ML or LSTM prediction?**
> Trading is a decision problem, not just a prediction problem. The result of a trade depends on future events and on our own earlier actions, like how much cash and position we are holding. A supervised model only predicts price, it does not learn *what action to take* after considering cost and risk. RL directly learns the action policy that maximises risk-adjusted return.

**Q2. What are PPO, SAC and DDQN? Why did you choose these?**
> PPO is a stable policy-gradient method, good for our high-level agent which picks a strategy from a few options. SAC is good for continuous actions, so it suits execution where the agent decides *how much* to buy. DDQN is a value-based method for discrete actions, and we compare it with SAC. We use ready-made, tested versions from Stable-Baselines3.

**Q3. What is your state, action and reward?**
> High level — state: regime probability, volatility, ADX, VIX, portfolio status. Action: choose strategy and capital allocation (0, 10, 25, 50 or 75 percent). Reward: risk-adjusted return, minus penalty for drawdown and for changing strategy too often.
> Low level — state: remaining quantity to buy, time left, short-term volatility, VWAP difference. Action: how much to execute now. Reward: negative slippage and cost — deliberately **not** profit.

**Q4. Why is profit not the reward for the low-level agent?**
> Because then both agents would be doing the same job and the split would be useless. The high-level agent decides *what and how much*. The low-level agent only decides *how to execute cheaply*. Separating the goals is what makes the hierarchy meaningful.

### B. Questions about your new idea (most risky — your own slides show similar papers)
**Q5. EarnHFT and MacroHFT already do regime-based hierarchical RL. So what is new in yours?**
> Three differences. First, they are crypto high-frequency systems with very heavy designs — EarnHFT uses hundreds of agents, MacroHFT uses memory-augmented networks. We use one light two-level pair. Second, they understand regime indirectly, whereas we connect a **separate supervised HMM classifier**. Third, and most important, neither of them tests for overfitting. Our PBO and Deflated Sharpe rejection pipeline is our main contribution, and we are doing it on Indian markets where this has not been done.

**Q6. HRT and the AAAI paper are also two-level. How is yours different?**
> They work on portfolio weights or direction only, without any regime classifier. We condition the entire system on a proper regime label and add the certification layer. Also, paper 14 shows that simply adding regime features into RL makes it fail. Our contribution is a design that connects them properly and then proves it statistically.

**Q7. Why two agents? Why not just one?**
> With one agent, the same model has to learn strategy and execution together for all market conditions. That is exactly what causes overfitting and real-market failure. By splitting, each agent learns a simpler task, and we can also test which part is actually giving the improvement.

**Q8. What is the Options framework and regime-conditioned termination?**
> An "option" means the high-level agent picks a strategy and sticks with it for some time, instead of changing every single step. "Regime-conditioned termination" means it stops that strategy when our regime classifier says the market has actually changed. So we switch strategy on real market change, not on a fixed timer.

### C. Regime detection questions
**Q9. Why HMM? Why not LSTM or Transformer?**
> HMM is the standard method for finding hidden market states. It is fast, needs less data, and it is explainable — we can clearly see what each state means. That suits a one-semester project. We keep K-Means and GMM as comparison. A deep model is future scope if HMM does not perform well.

**Q10. How many regimes, and how do you know they are correct?**
> Three to five states, selected using BIC on training data. We label them using their average return and volatility — high volatility with negative return becomes "bear", and so on. Then we verify with known periods: the 2020 COVID crash should show as high-volatility bear, and 2021 should show as bull. That is a real check, not assumption.

**Q11. Regime detection is always late. You know the regime only after it changes. Then how is it useful?**
> That is a genuine limitation of any regime model, and we accept it. That is why we also calculate a confidence score. When confidence is low, the system does not trade at all instead of acting on an old label. We prefer staying out during unclear periods rather than trading wrongly.

### D. Validation and overfitting (your strong area — answer confidently)
**Q12. What is PBO and Deflated Sharpe Ratio?**
> PBO tells us the probability that the strategy which looked best on training data is actually below average on unseen data. In simple words, it tells us the chance that our best model is just luck. Deflated Sharpe Ratio corrects the Sharpe value based on how many models we tried — because if you try 100 models, one will look good by chance. Together they let us reject fake-good models.

**Q13. How do you avoid look-ahead bias?**
> Every calculation, including normalisation, is done only using training data of that fold, then frozen for testing. We also have an automatic test that checks this. It shifts test data into training and checks whether values change. If a feature does not change, it means the model is seeing future data, and our build fails.

**Q14. What is walk-forward validation, and why rolling window?**
> Instead of one train-test split, we move a window forward in time — train, validate, test, then shift ahead and repeat. So the model is always tested on data that comes *after* its training period. We use a rolling window because it checks whether the model adapts to changing markets, which is exactly our claim.

**Q15. How do you prove your improvement is real and not random?**
> We use the Jobson–Korkie test with Memmel correction to compare Sharpe ratios, and bootstrap confidence intervals. So we report whether the difference is statistically significant, not just a bigger number.

### E. Data, feasibility, rules
**Q16. Where is your data from? Is it enough?**
> Daily data for NIFTY stocks from yfinance and NSE bhavcopy, from 2015 onwards, and minute data for a few stocks from Kite. We adjust for splits and bonuses, and check for missing or wrong prices. It is free, enough for daily-level learning, and it covers multiple market phases including the 2020 crash and the 2022 bear market.

**Q17. Is real money involved? What about SEBI rules?**
> No real money at any stage. It is only paper trading on live prices using Kite's sandbox, and real-order functions are blocked in our code. Under SEBI's 2026 rules, real deployment needs broker partnership and algo registration, which is outside our project scope, although our design is compatible with it.

**Q18. Can you actually complete this in one semester?**
> Yes, because it is divided into phases and each phase gives a working result. A basic model is ready by week 5, the full hierarchy by week 10, and certification with live paper trading by week 12. Even in the worst case, if the hierarchy does not beat the baseline, our certification and abstention results are separate contributions. So we always have a valid result.

**Q19. What if your hierarchical model does not beat a simple model?**
> That is still a valid scientific result and we will report it honestly with proper significance testing. A well-tested negative result on Indian markets is also a contribution, and our ablation study will show the reason. But we are fairly confident, because we use warm-start training and separated rewards, which research shows makes hierarchical training stable.

### F. Evaluation and general
**Q20. Which metrics will you use? Why not just profit?**
> Sharpe, Sortino, Calmar, maximum drawdown and CVaR. We use risk-adjusted metrics because profit alone hides how much risk was taken. Maximum drawdown especially matters for us, because our claim is that staying out of bad markets should reduce losses.

**Q21. What is the biggest risk in your project?**
> Kite API approval time for the live part — so we are applying in week one and keeping Fyers as backup. On the technical side, look-ahead bias is the main risk, which is why we added the automatic leakage test from week three itself.

**Q22. Is this publishable or can it become a product?**
> Yes. The research gap we found is publishable. And commercially, SEBI's 2026 rules created a real need for exactly this — independent checking of whether a trading algorithm is genuine or overfitted. That is a natural direction after the capstone.

---

## 5. Tips while presenting

- **Explain, don't read the slide.** The highest marks are for "detailed explanation", so talk *about* the slide instead of reading it.
- On literature slides, **do not read all 15 rows.** Just explain the two gaps and say your novelty line. Reading tables wastes time and bores the panel.
- Answer in **two sentences first**, then stop. Give more detail only if they ask again.
- If you don't know something, say: *"We have not finalised that yet. Our current plan is X, and we will confirm it in phase Y."* Never make up an answer — panels respect honesty, and the rubric actually gives marks for knowing your limits.
- **Pass questions to the right person.** Saying "Vishal is handling validation, Vishal will answer this" looks like a proper team.
