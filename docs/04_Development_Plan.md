# VeriAlpha — Development Plan

15-week execution plan mapped to the MPSTME review cycle (Review 1: Week 2 · Review 2: Week 7 · Review 3: Week 15), with team roles, fortnightly deliverables for the log book, risk management, and the startup runway beyond the capstone.

---

## 1. Team Structure (2–3 members)

| Role | Owns | Primary deliverables |
|---|---|---|
| **ML Lead** | Agents, envs, regime module | LLA/HLA training, ablations, confidence layer |
| **Platform Lead** | Data, certification pipeline, daemon | Ingestion jobs, fold generator, PBO/DSR stats, Kite integration |
| **Product/Eval Lead** *(or shared if 2 members)* | Dashboard, report, experiments | Streamlit app, integrity reports, black-book report, presentations |

All members rotate through the weekly log book entry and mentor meeting (guidelines §4.2 require weekly meetings and individual-contribution evidence at Review 2 — keep per-member commit history clean).

---

## 2. Phase Plan

### Phase 0 — Foundation (Weeks 1–2) → **Review 1**
- Literature survey: ≥15 references (HRT, EarnHFT, ELTRA, HARLF, RegimeFolio, Bailey & López de Prado, PBO paper, Moody & Saffell, SEBI circular + India retail-algo landscape). Written comparatively — this feeds Chapter 2 of the report directly.
- Finalize universe (NIFTY 50 + 5 sector ETFs, daily; 5 symbols minute-level), date range (2015–present), and regime periods for stratified evaluation (2020 COVID crash, 2021 melt-up, 2022 bear, 2023–24 chop).
- Repo scaffold, CI, config system; Kite developer account application submitted (lead time!).
- **Exit criteria:** synopsis + Gantt approved at Review 1; topic approval form (A.2) signed.

### Phase 1 — Data & Baseline (Weeks 3–5)
- W3: ingestion + indicators + leakage-safe scalers + leakage CI test.
- W4: `ExecutionEnv`/`StrategyEnv` skeletons, Indian cost model with unit tests, walk-forward fold generator.
- W5: classical baselines (buy-and-hold, MA-crossover, risk-parity) + **flat PPO baseline** through the full fold pipeline → first results table (this is Table 1 of every future comparison).
- **Exit criteria:** E1+E2 experiment rows complete and reproducible from config; data job green daily.

### Phase 2 — Hierarchy (Weeks 6–10) → **Review 2 at Week 7**
- W6: HMM regime module; regime-stratified re-evaluation of Phase-1 baselines (instant midterm-ready result: "here is *when* the flat agent fails").
- W7: **Review 2** — present pipeline, baselines, regime analysis, interim report. Guidelines expect ~40–50% implementation with documented results; Phase 0–1 output clears that bar with margin.
- W8: LLA warm-start training vs. cost model; TWAP slippage benchmark beaten.
- W9: HLA training with frozen LLA; switching-cost reward tuning.
- W10: interfaces + joint fine-tune; full hierarchy (E4) runs end-to-end on all folds.
- **Exit criteria:** E3/E4 rows complete; hierarchy vs. flat comparison exists (even if not yet favorable — tuning continues in Phase 4).

### Phase 3 — Integrity Layer (Weeks 11–12)
- W11: N-seed ensembles; PBO (CSCV) + Deflated Sharpe implementation; rejection gate; first Integrity Report artifact.
- W12: confidence-scaled sizing + abstention (τ tuned on validation only); **paper-trading daemon goes live** on Kite sandbox — earliest possible, to maximize live-log runway before Week 15.
- **Exit criteria:** ≥1 candidate demonstrably rejected by the gate; daemon logging decisions daily.

### Phase 4 — Evaluation & Delivery (Weeks 13–15) → **Review 3**
- W13: full ablation matrix (E1–E7), Jobson–Korkie + bootstrap CIs, cost-sensitivity runs; failure-case analysis (which regime hurts, why — the guidelines' rubric explicitly rewards this).
- W14: dashboard complete (three tabs); live divergence report; draft black-book report per format A.8 (TNR 12pt, 1.5 spacing, chapter structure Ch1–Ch6, appendices with code flowcharts).
- W15: report hard-bound + mentor-signed; presentation + demo rehearsal (dashboard-driven demo: live tab → agent brain → integrity view); **Review 3: final presentation + viva.**
- **Exit criteria:** all acceptance criteria AC-1…AC-6 evidenced; log book complete and signed.

---

## 3. Fortnightly Deliverables (log book / mentor review checkpoints)

| Fortnight | Demonstrable artifact |
|---|---|
| F1 (W1–2) | Synopsis, literature matrix, approved Gantt |
| F2 (W3–4) | Data pipeline demo + env/cost-model unit tests passing |
| F3 (W5–6) | Baseline results table + regime-stratified analysis |
| F4 (W7–8) | Midterm interim report + LLA beating TWAP benchmark |
| F5 (W9–10) | Full hierarchy training run + MLflow experiment browser |
| F6 (W11–12) | Integrity Report artifact + live daemon screenshot/logs |
| F7 (W13–14) | Ablation tables + dashboard walkthrough |
| F8 (W15) | Final report, presentation, demo |

---

## 4. Risk Management

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kite API approval delayed | Medium | Live daemon slips | Apply Week 1; fallback: Fyers API, or replay-based "shadow live" on delayed NSE data |
| HRL training unstable / hierarchy underperforms flat | Medium | Core claim weakens | Warm-start protocol; separated rewards; honest reporting — the certification pipeline and abstention results stand on their own even if E4 ≈ E2 |
| Compute bottleneck (full fold matrix) | Medium | Schedule slip | Daily bars for HLA keep episodes short; reduce universe to 20 symbols; Colab/Kaggle GPU for training bursts; cache fold datasets |
| Overfitting to validation during tuning | High | Invalid results | Tune only on val folds; test folds touched once; #trials logged for DSR correction |
| Look-ahead leakage bug | High | All results invalid | Leakage CI test from Week 3; code review rule: any `fit()` call must take a fold-scoped dataset |
| Scope creep (LLM sentiment, crypto, etc.) | High | Nothing finishes | Frozen scope per SRS §1.2; ideas parked in FUTURE.md |
| Team member unavailability | Low | Slip | Every component has a config-driven CLI; no single-owner black boxes; weekly sync |

---

## 5. Engineering Practices

- **Branch/PR flow** with review by another member; `main` always runs the smoke test.
- **CI (GitHub Actions):** lint + pytest + 2-minute smoke training run + leakage test.
- **Experiment discipline:** no result exists unless it is in MLflow with its config and seed. The report's tables are generated by `scripts/make_tables.py` from MLflow — never hand-typed.
- **Weekly cadence:** Mon plan → Fri demo among team → mentor meeting → log book entry signed.

---

## 6. Report Mapping (black book, format A.8)

| Report chapter | Source material |
|---|---|
| Ch 1 Introduction | Synopsis §2–3 (problem, motivation, salient contribution) |
| Ch 2 Literature survey | Phase-0 literature matrix (≥15 refs, comparative) |
| Ch 3 Methodology & Implementation | Pipeline doc (block diagram, algorithms, flowcharts → Appendix A) |
| Ch 4 Results & Analysis | Experiment matrix E1–E7, integrity reports, live divergence |
| Ch 5 Advantages, Limitations, Applications | Abstention/certification benefits; daily-bar + sandbox limits; SEBI-2026 certification application |
| Ch 6 Conclusion & Future Scope | FUTURE.md (LLM sentiment, multi-asset, RA-licensed black-box certification service, Gen-Z paper-trading arena) |

---

## 7. Beyond Week 15 — Startup Runway (not part of capstone scope)

The certification pipeline (Phase 3) is the seed of the product: under SEBI's retail-algo framework (mandatory since April 2026), every algo provider needs registration and performance disclosure, and brokers carry liability for vetting. Post-capstone path: package the Integrity Report as a self-serve service for Tradetron/Streak strategy authors → broker compliance pilot → certification badge as the standard. The capstone deliberately produces the three assets that pitch needs: a working pipeline, real rejected-strategy case studies, and months of live divergence data.
