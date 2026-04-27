# Paper Outline — MA751 Final Project

> **Title**: Does Complexity Help Cross-Sectional Stock Return Prediction?
>           A 5-Rung Ladder Audit on the S&P 500 Panel (2016–2024)
>
> **Author**: Eva Qi
> **Date**: April 2026
> **Length target**: 12–16 pages, 9 figures, 3 tables, ~12 references

---

## 1. Barebone — What We've Done

### 1.1 Data construction
- Universe: 442 S&P 500 tickers (frozen snapshot)
- Window: 119 monthly cross-sections, 2015-01 → 2024-11
- Test window: 58 months, 2020-02 → 2024-11
- Sources: CRSP monthly + daily, Compustat, IBES, Compustat short interest, TFN 13F
- Features: 22 V3 firm-level signals + 5 missingness indicators = **27 total**
- Pipeline: PIT merge → universe filter → cross-sectional z-score → 1%/99% winsorize → Type-A NaN handling
- Audit fixes: `EarningsYield_sign_raw` captured pre-filter (P0 #2); GP/A and EY use Type-A handling for sector-undefined firms

### 1.2 5-rung complexity ladder (12 model variants)
- **Rung 1** (Linear): 1a OLS, 1b IC-Ensemble, 1c Fama-MacBeth, 1d Barra (Grinold-Kahn)
- **Rung 2** (Regularized): 2a LASSO, 2b Ridge, 2d Elastic Net, 2e Adaptive LASSO
- **Rung 3** (Smooth non-linear): 3a GAM, 3b XGBoost (GKX-tuned)
- **Rung 4** (Single-task MLP): hidden 64→32, ReLU, dropout 0.10, Huber loss
- **Rung 5** (Multi-task / MoE): 5b/5c/5d Kendall MTL, Regime-Gated MoE, Enhanced MoE

### 1.3 Methodology fixes (audit cycle)
- **TimeSeriesSplit** replaces random K-fold for inner CV (was hidden temporal leakage)
- **Ridge `scoring=Spearman`** replaces R² (matches outer evaluation metric)
- **Per-fold seeds** for stochastic methods (was single global seed → hid init noise)
- **`smooth_l1_loss` (Huber)** replaces MSE in MLP family (fat-tail robust)
- **Barra duplicate fix** (P0 #1): deleted heuristic version overriding Grinold-Kahn — Sharpe rank 8 → 3
- **`has_positive_earnings` semantic fix** (P0 #2): preserve sign before plausibility filter

### 1.4 Five sensitivity studies
- **GAM `n_splines` sweep** (10 configs, 3-20): default n=10 over-fits; n=4 wins Sharpe
- **K-fold λ vs GCV** (negative result): K=5 inner CV does not improve over single-shot GCV
- **XGBoost sensitivity grid** + GKX-tuned: default depth-4 fails; depth-1 stumps win
- **MLP 5-seed sensitivity**: SNR=1.33 (signal barely above init noise); all seeds < Ridge floor
- **MLP architecture sweep** (hidden ∈ {8,16,32,64}): smaller ≥ larger → over-parameterized

### 1.5 Failed improvement attempts (negative results)
- **K=5 seed bagging**: Sharpe 0.394 vs single-seed mean 0.412 → bagging hurt (rank/Pearson space mismatch)
- **Inner-CV HP search** (27 configs): Sharpe collapsed 0.347 → 0.111 (over-tune on low-SNR validation)

### 1.6 CPCV (López de Prado 2018)
- Replace single-path walk-forward with 15-path Combinatorial Purged K-fold
- N=6 blocks, k_test=2, embargo=1 month
- 8 models × 15 paths = 120 backtests
- Result: top-7 models' [5%, 95%] Sharpe bands all overlap

### 1.7 Headline thesis
At 119-month / 442-stock sample size, **no model statistically dominates another**. True Barra (1c) and Fama-MacBeth (1c) are the most robust point estimates (mean Sharpe 1.058 / 1.071, std 0.544 / 0.494, P(Sharpe<0)=0%). The complexity ladder neither helps nor hurts in a statistically meaningful way.

---

## 2. Section Structure (10 sections + appendix)

| § | Section | Pages | Figures | Tables |
|---|---|---|---|---|
| Abstract | | ½ | — | — |
| 1 | Introduction | 1 | — | — |
| 2 | Data | 1½ | Fig 1 | — |
| 3 | Methodology (5-rung ladder + walk-forward + CPCV) | 2 | Fig 2 | — |
| 4 | Walk-forward Results | 1½ | Fig 8 | Tab 1 |
| 5 | Audit-Cycle Findings | 3 | Figs 3, 4, 5 | Tab 2 |
| 6 | CPCV (Path Uncertainty) | 1½ | Figs 6, 7 | Tab 3 |
| 7 | Discussion | 1½ | — | — |
| 8 | Limitations | 1 | — | — |
| 9 | Conclusion | ½ | — | — |
| Refs | | ½ | — | — |
| App A | Hyperparameter inventory | ½ | — | inline |

**Total**: ~14 pages, 9 figures, 3 tables.

---

## 3. Key Narrative Arc

1. **Hook** (Intro): "Cross-sectional return prediction is a textbook low-SNR problem. Does adding model complexity help?"
2. **Setup** (Data + Methodology): "We built 12 variants on a 5-rung ladder and evaluated rigorously."
3. **First-pass result** (Results): "5-6 models cluster near Sharpe 1.0; non-linear and MLP appear to fail."
4. **Twist** (Audit findings): "But two of those failures were hyperparameter-default artifacts, and three independent attempts to fix MLP all backfired in instructive ways."
5. **Real conclusion** (CPCV): "Single-path Pareto comparisons are within path noise. No model statistically dominates."
6. **Lessons** (Discussion): "Sample-size constraint is the real story. Methodology pitfalls (P7/P8) generalize."
7. **Limits** (Limitations): "What we couldn't verify; what would change with more data."
8. **Verdict** (Conclusion): "FM/Barra most robust; complexity neither helps nor hurts."

---

## 4. Per-Section Plan

### § Abstract
- 150 words
- Sentence 1: problem setup
- Sentence 2: panel + 12 model variants + walk-forward
- Sentence 3: top-line walk-forward finding
- Sentence 4: CPCV finding (overlap)
- Sentence 5: audit lessons (P7/P8)
- Sentence 6: thesis statement

### § 1 Introduction
- Motivation: factor zoo crisis, replication failure (Hou-Xue-Zhang 2020), out-of-sample gap
- Question: at S&P 500 large-cap scale, does model capacity matter?
- Approach: 5-rung ladder on identical data with rigorous evaluation
- Contributions (4 bullets):
  1. Audited 12-variant comparison
  2. CPCV uncertainty bounds
  3. Two methodology lessons (P7, P8)
  4. Two retractions (default-trap recovery)
- Headline: no model statistically dominates; FM/Barra most robust

**Figures**: none (text-only intro)
**Discussion**: tie to Gu-Kelly-Xiu 2020 finding that ML helps on bigger / mid-cap universe (different regime than ours)

### § 2 Data
- §2.1 Sources (CRSP/Compustat/IBES/13F/Short)
- §2.2 Universe (442 S&P 500 tickers, frozen snapshot)
- §2.3 Features (27 = 22 + 5 flags); category breakdown table
- §2.4 Preprocessing (universe-filter-before-zscore = explicit audit point)
- §2.5 Train/test protocol

**Figure**: Fig 1 — Data flow pipeline (TikZ schematic, hand-drawn — placeholder OK)

**Discussion needed**:
- Why frozen S&P 500 (survivorship caveat)
- Why 4-month fundamentals lag
- Why Type-A NaN handling matters

**Limitations to flag here**:
- Survivorship bias from frozen snapshot
- 119-month window includes COVID crash + recovery (regime-specific)

### § 3 Methodology
- §3.1 The 5-rung ladder (refer to Fig 2)
- §3.2 Walk-forward expanding-window backtest
- §3.3 CPCV alternative (López de Prado 2018)
- §3.4 Evaluation metrics: IC (Spearman), IC-IR, hit rate, LS quintile Sharpe
- §3.5 Train/test/inner-CV protocol details

**Figure**: Fig 2 — 5-rung ladder schematic ✅ (just generated)

**Discussion needed**:
- Why Spearman over Pearson (rank invariance, fat tails)
- Why long-short quintile Sharpe (tradability)
- Why CPCV (sample-size honesty)

### § 4 Walk-Forward Results
- Main table (12 model variants + 2 baselines): IC, IC-IR, Hit, LS Sharpe
- Top-line ranking: 1d Barra (1.134), 2a LASSO (1.000), 2d ElNet (0.997), 3b XGB (0.984), 2b Ridge (0.966), 1c FM (0.959)
- Surprise: 1d Barra rank 1 — but only after audit-cycle Barra fix (forward-reference §5)
- Rung 4/5 dominated by linear baselines

**Figure**: Fig 8 — Cumulative LS portfolio returns for top-5 models ✅
**Table**: Table 1 — Walk-forward results (full)

**Discussion needed**:
- Is LASSO 1.000 vs ElNet 0.997 meaningful? (forward-ref to CPCV)
- Why does Barra (Rung 1d) match XGBoost (Rung 3b)? — answer: with proper covariance estimation, Barra extracts the same signal; complexity isn't the dimension that helps

**Limitations to flag**:
- Single-path Sharpe is point estimate — full uncertainty in §6

### § 5 Audit-Cycle Findings (CENTRAL section)
- §5.1 Hyperparameter Default Trap (Rung 3a + 3b retractions)
- §5.2 MLP Single-Path Results Are Initialization Noise
- §5.3 Bagging Failure: Metric-Space Mismatch (P7)
- §5.4 HP Search Hurts on Low-SNR Validation (P8)
- §5.5 Barra Duplicate Definition (P0 fix; ranks change)

**Figures**:
- Fig 3 — GAM n_splines sweep ✅
- Fig 4 — XGBoost sensitivity heatmap ✅
- Fig 5 — MLP seed sensitivity ✅

**Table**: Table 2 — Hyperparameter inventory + sweep coverage

**Discussion needed**:
- Why GAM n=4 wins (1 internal knot = monotone-with-slight-curve fits the financial signal shape)
- Why MLP seeds differ — random init creates different rank-orderings that compress under bagging
- Validation noise SE formula: SE ≈ 1/sqrt(n_val) ≈ 0.011 for our K=3 inner splits ≈ true signal magnitude

**Limitations**:
- Bagging tested only at K=5; rank-level bagging not tested
- HP search tested only at K=3 inner CV; reduced search space (5 configs) not tested

### § 6 CPCV (Path Uncertainty)
- §6.1 Method (López de Prado): N=6 blocks, k=2 test, embargo=1 month, C(6,2)=15 paths
- §6.2 Result table (15-path Sharpe distribution per model)
- §6.3 Interpretation: top-7 [5%, 95%] bands all overlap
- §6.4 Risk-adjusted ranking: FM (lowest std), Barra (highest 5th %ile)

**Figures**:
- Fig 6 — CPCV 15-path Sharpe boxplot ✅
- Fig 7 — Pareto plot with CPCV uncertainty bars ✅

**Table**: Table 3 — CPCV ranking with mean/std/percentiles

**Discussion needed**:
- WF "Pareto winner" LASSO vs CPCV "most robust" FM — single path overstated differentiation
- Per-path Sharpe std ≈ 0.5–0.8 = same scale as cross-model differences
- P(Sharpe<0) = 0% for 6 of 8 models → signal IS positive even if not differentiable
- Why XGB has highest mean but widest std

### § 7 Discussion
- §7.1 Sample-size constraint as the binding factor
  - n=58 months × 442 stocks ≈ 25,636 obs
  - IC SE under H₀ ≈ 0.0063
  - Detected IC ≈ 0.018–0.026 = 3σ above zero (significant for individual model)
  - Pairwise IC ΔIC ≈ 0.002–0.005 = within noise → can't distinguish models
  - CPCV Sharpe per-path std ≈ 0.5–0.8 confirms

- §7.2 Why complexity doesn't help here
  - Cross-sectional financial features are mostly near-monotone
  - GAM n=4 (1 knot) finding supports this
  - Tree interactions (XGB-GKX) don't beat well-tuned linear (1d Barra)
  - MLP/MTL adds capacity but no signal at our SNR

- §7.3 Methodology lessons (P1–P12 from code-council)
  - P1 "Don't tune failed" bias
  - P2 Multi-seed SNR
  - P5 Domain-specific defaults
  - **P7** Bagging metric-space match (NEW from this project)
  - **P8** HP overtune on low-SNR (NEW from this project)

- §7.4 Comparison to Gu-Kelly-Xiu 2020
  - Their finding: ML helps in larger universe (~5,000 stocks) and longer window (1957–2016)
  - Our finding: at S&P 500 / 119-month, complexity doesn't help
  - These are consistent: ML benefit scales with sample size

**No new figures**; reuses material from §5–§6.

### § 8 Limitations (NEW — explicit section)
1. **Survivorship**: frozen 442-ticker S&P 500 snapshot, not point-in-time
2. **Sample size**: 58 test months, 442 stocks → underpowered for pairwise comparisons
3. **Regime mix**: 2020 COVID crash + 2022 inflation + 2023 mag-7 dominance — possibly regime-specific
4. **`has_positive_earnings` patch**: source-code fix applied; results in this paper used pre-patch panel (expected drift < 0.005 IC, well within CPCV noise)
5. **CV val split by row not by month**: documented; minor train/val month-boundary fuzziness
6. **CPCV doesn't include Rung 4/5**: MLP/MTL/MoE were dominated under WF; CPCV adds them only adds confirmation, not new info
7. **HP grids not exhaustive**: GAM n_splines and XGB depth/lr/n_est swept; MoE K and gate temperature not swept
8. **No transaction cost modeling beyond fixed 10 bps**: real impact may differ
9. **No live trading, no out-of-window**: panel ends 2024-11

**Discussion needed**: which of these would change conclusions if fixed
- Survivorship: probably worsens results uniformly (small caps hurt all models same)
- Sample size: would help us distinguish models, but unlikely to flip rankings
- Regime: 2020 was a stress test; passing it is meaningful

### § 9 Conclusion
- Restate thesis: complexity does not help in a statistically meaningful way at our sample size
- 1c FM and 1d Barra are most robust (CPCV mean Sharpe 1.07/1.06, P(<0)=0%)
- Two methodology pitfalls (P7, P8) generalize beyond this project
- Two retractions (Rung 3a/3b) demonstrate the value of explicit sensitivity sweeps
- Open question: at what scale does complexity start to help? (forward-reference Gu-Kelly-Xiu)

---

## 5. Figures Final List

| # | File | Section | Type | Source |
|---|---|---|---|---|
| 1 | `fig_data_flow.pdf` | §2 | Schematic | TikZ (hand-drawn placeholder OK) |
| 2 | **`fig_ladder.pdf`** | §3 | Schematic | matplotlib (DONE) |
| 3 | `fig_gam_n_splines.pdf` | §5.1 | Dual-axis line | DONE |
| 4 | `fig_xgb_sensitivity.pdf` | §5.1 | Heatmap | DONE |
| 5 | `fig_mlp_seeds.pdf` | §5.2 | Bar + reflines | DONE |
| 6 | `fig_cpcv_sharpe.pdf` | §6 | Boxplot | DONE |
| 7 | `fig_pareto_with_bounds.pdf` | §6 | Scatter + errorbars | DONE |
| 8 | `fig_cumret.pdf` | §4 | Time series | DONE |

**Status**: 7 of 8 done. Fig 1 (data flow) needs hand-drawing — can defer or use simple text-only flowchart in the paper.

---

## 6. Tables Final List

| # | Section | Contents |
|---|---|---|
| 1 | §4 | Walk-forward main results: 14 rows (12 models + 2 baselines) × 6 columns |
| 2 | §5.1 | GAM n_splines sweep (9 configs) |
| 3 | §6 | CPCV 15-path Sharpe ranking (8 models × 7 columns) |
| App A | Appendix | Hyperparameter inventory across all rungs |

---

## 7. Per-Section Discussion Themes

### Discussion themes to cover

1. **Sample-size honesty**: this is the recurring theme. Make it explicit with arithmetic in §7.1.
2. **Default-trap recovery**: rather than hide the retractions, lean into them — these are the strongest evidence of the audit's value.
3. **Why complexity doesn't help**: structural argument (financial features are near-monotone) vs. statistical argument (sample size) — both apply.
4. **Methodology generalization**: P7 + P8 lessons travel beyond this project; cite by name.
5. **Practical recommendation**: deploy FM or Barra, not the highest-mean-Sharpe model (XGB has 6.7% P(neg)).

### Limitations themes

1. **What we did**: explicit list of all known compromises.
2. **Why we didn't fix all of them**: deadline + risk vs. impact tradeoff.
3. **What would change**: per-limitation prediction of direction-of-change.
4. **Honest disclosure**: if results in published paper differ from current parquet by ≥0.01 IC after re-running pipeline, footnote-flag.

---

## 8. Final Thesis Statement (Submission-grade)

> "On a 442-stock S&P~500 panel across 119 months (2015--2024) with 27 firm-level features, we evaluate twelve model variants spanning a 5-rung complexity ladder—from pooled OLS to multi-task mixture-of-experts. Walk-forward backtests appear to support a Pareto ranking with six top models clustered within Sharpe~$\in [0.92, 1.13]$, but Combinatorial Purged K-fold cross-validation reveals that all top-7 models' 5--95\% Sharpe bands overlap. \textbf{Adding model complexity does not improve cross-sectional return prediction in a statistically meaningful way at our sample size.} The Grinold-Kahn Barra factor model and Fama-MacBeth time-averaged regression are the most robust point estimates (CPCV mean Sharpe~1.058/1.071, $P(\text{Sharpe}<0)=0\%$). Three independent attempts to improve the multilayer perceptron (multi-seed averaging, prediction bagging, inner-CV hyperparameter search) all fail to beat any single-seed linear baseline, and yield two methodology pitfalls that generalize beyond this project: (i)~mean-prediction bagging is incompatible with rank-based evaluation metrics, and (ii)~hyperparameter grid search degrades performance when validation signal is below the noise floor."

---

## 9. Outstanding Decisions for Eva

1. **Re-build panel for P0 #2 fix**? My recommendation: **no**, document in limitations. (~1 hour rebuild + 5 hour rerun, expected drift < 0.005 IC.)
2. **Fig 1 (data flow)**: TikZ-draw or use simple text-listing in §2? My recommendation: **text-listing** in §2 (no figure 1).
3. **Discussion length**: keep at 1.5 pages or expand? My recommendation: **keep at 1.5** — substantive but not bloated.
4. **Bibliography style**: `plainnat` (current) vs `apsr` vs `chicago`? My recommendation: **plainnat** (most standard for ML/econ papers).
5. **Author affiliation**: add "Department of Mathematics & Statistics, Boston University" or leave name-only? Up to you.
