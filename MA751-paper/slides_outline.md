# MA751 Final-Project Presentation — Slide Outline

> Target deck: 22 slides for ~15 minutes (~45s/slide). Action-title style throughout. One exhibit per slide max. Speaker notes are talking points, not script.
>
> Paper: *Does Complexity Help Cross-Sectional Stock Return Prediction? A 5-Rung Ladder Audit on the S&P 500 Panel (2016-2024)*. Authors: Alan Xi, Eva Qi, Reetom Gangopadhyay.

---

# Slide 1: Complexity does not improve cross-sectional return prediction

- 442 stocks, 119 months, 27 features, 12 model variants
- 5-rung ladder: linear -> regularized -> GAM/XGB -> MLP -> MTL/MoE
- CPCV: top-7 models' 5-95% Sharpe bands all overlap
- Best by robustness: Fama-MacBeth and Grinold-Kahn Barra

**Figure**: none (title slide)

## Speaker Notes
This is a null-result paper with a positive methodology contribution. Headline: at S&P 500 scale, no complexity rung statistically beats linear. We will walk through five complexity rungs, show the path-uncertainty story via CPCV, and document three audit-cycle lessons that travel beyond this project. The thesis sits in deliberate contrast to Gu-Kelly-Xiu 2020, who saw ML wins on a much broader 30,000-stock panel. This sets up the motivation slide.

---

# Slide 2: Low-SNR + factor zoo + open question motivate the test

- Cross-sectional IC typically 0.02-0.05 (textbook low SNR)
- Harvey-Liu-Zhu factor zoo: replication crisis in finance
- Gu-Kelly-Xiu 2020: ML wins on broad universe, 1957-2016
- Does the GKX result transfer to one-decade large-cap?

**Figure**: none (text slide; could embed inline citation list)

## Speaker Notes
The motivating open question is whether the GKX deep-learning result on 30,000 stocks generalizes to the typical buy-side scale of about 500 large-caps over a single decade. Classical Fama-MacBeth ICs sit in the 0.02-0.05 range, the literature is awash in "factor zoo" alpha that fails to replicate, and the practitioner question is whether a neural network beats Ridge once we constrain to a working buy-side universe. Our contribution is to test that question with audit-graded protocol. Next slide introduces the framework we use to organize the test.

---

# Slide 3: Five-rung ladder isolates each capacity step

- Rung 1 Linear: OLS, IC-Ensemble, Fama-MacBeth, Barra-GK
- Rung 2 Regularized: LASSO, Ridge, Elastic Net, Adaptive LASSO
- Rung 3 Smooth non-linear: GAM (splines), XGBoost (trees)
- Rung 4-5: single-task MLP, multi-task MLP, regime-gated MoE

**Figure**: `fig_ladder.pdf` -- Each rung adds one structural ingredient (regularization, smooth non-linearity, deep capacity, shared representation).

## Speaker Notes
The ladder is the experimental design: each step adds exactly one structural ingredient over the previous step, so any Sharpe gain is attributable to that ingredient. Parameter count grows from about 27 at Rung 1 to over 3,000 at Rung 4. All twelve variants share an identical panel, identical preprocessing, identical evaluation metric (Spearman IC + LS-quintile Sharpe). The framework lets us answer "does the next step buy us anything" rather than comparing apples to oranges. Next we describe the panel.

---

# Slide 4: V3 panel (27 features) beats V2 (14) on most models

- 442 S&P 500 tickers, frozen membership snapshot
- 119 monthly cross-sections, 2015-01 to 2024-11
- WRDS sources: CRSP + Compustat + IBES + 13F + short interest
- V3 adds Phase-2 IBES signals + missingness flags

**Figure**: `fig_v2_vs_v3.pdf` -- Phase-2 expansion lifts most models, with Barra benefiting most (0.982 -> 1.134).

## Speaker Notes
We use 27 features in the V3 panel: 7 classical factors, 4 IBES signals, 8 Phase-2 firm-level signals, 5 missingness flags, plus the 3 imputed risk features. V3 expanded V2 (14 features) by adding IBES revisions, dispersion, breadth, plus liquidity and 52-week high signals. The V2-vs-V3 chart shows Phase-2 features helped most models, with Barra-GK benefiting most. Survivorship bias from the frozen snapshot is acknowledged in Limitations. Next we describe the evaluation protocol.

---

# Slide 5: Walk-forward backtest with strict purge protects against leakage

- Expanding window, 60-month minimum train
- 1-month purge buffer between train and test
- 58 test months: 2020-02 through 2024-11
- Inner CV: TimeSeriesSplit K=5, gap=1 on month boundaries

**Figure**: none (text slide; could add a simple timeline schematic if Eva wants)

## Speaker Notes
Outer evaluation is expanding-window walk-forward with a strict 1-month purge to prevent train/test contamination from monthly publication lags. Inner CV for hyperparameter selection is TimeSeriesSplit with K=5 and a 1-month gap, computed on month boundaries not row indices. The 58-month test window includes COVID 2020, the 2022 inflation regime, and the 2023 AI concentration -- a stress test, not a benign window. This single-path setup gives us point estimates; the next slide introduces CPCV for path uncertainty.

---

# Slide 6: CPCV converts one path into 15 to expose model robustness

- Walk-forward gives one realization; sample variability is hidden
- Lopez de Prado 2018: 6 contiguous blocks, 2 test blocks per path
- Binomial(6,2) = 15 paths per model, 1-month embargo
- Applied to 8 top models from Rung 1, 2, 3b

**Figure**: none (text slide; could add CPCV path diagram if time permits)

## Speaker Notes
Walk-forward returns one historical realization -- one specific train/test partition. CPCV picks 15 alternative partitions, holding out two blocks at a time with a one-month embargo on either side. Each model is refit and evaluated on each of the 15 paths. We then report the distribution of per-path Sharpe rather than a single number. We apply CPCV to the eight strongest models from Rungs 1-3b; Rung 4-5 are already dominated under walk-forward and adding them to CPCV would only confirm dominance. Next is the walk-forward results table.

---

# Slide 7: Six top models cluster within Sharpe [0.92, 1.13]

- Linear top: Barra-GK 1.134, Fama-MacBeth 0.959
- Regularized top: LASSO 1.000, Ridge 0.966, Elastic Net 0.997
- All top-6 within 0.20 Sharpe of each other
- Spread foreshadows CPCV path-noise finding

**Figure**: none (this slide is the condensed Tab 1 -- show only Rungs 1-2 + Ridge floor)

## Speaker Notes
On single-path walk-forward, six models cluster tightly within Sharpe 0.92 to 1.13: Barra-GK at the top with 1.134, then LASSO 1.000, Elastic Net 0.997, Ridge 0.966, Fama-MacBeth 0.959. The Ridge alpha=1 floor gave 0.925, so the gain over a baseline-tuned Ridge is small. The narrowness of the spread is the first hint that ranking is path-dependent rather than skill-dependent. Next we go to the GAM result, which initially looked like a Rung 3 failure.

---

# Slide 8: GAM defaults are 2x too complex for IC~0.01 SNR

- pygam default n_splines=10 -> Sharpe 0.555
- Sweep n_splines in {3..20}; peak at n=4 (Sharpe 0.828)
- Knots > 2 fit noise; financial factors are near-monotone
- Lesson: framework defaults assume moderate-SNR benchmarks

**Figure**: `fig_gam_n_splines.pdf` -- IC and Sharpe both peak at n in {4,5}, monotone decay for n >= 6.

## Speaker Notes
Our first Rung 3 run used pygam's default n_splines=10 and produced Sharpe 0.555 -- a "GAM failed" result. The audit-cycle sweep over n_splines in {3..20} reveals that defaults were the source of failure, not the method. n_splines=4 gives Sharpe 0.828, competitive with the linear cluster. Each spline has n_splines minus spline_order internal knots; at IC~0.01 SNR, one or two internal knots is the right capacity, and pygam's default of 10 is calibrated for moderate-SNR generic regression. We will return to this as the "Hyperparameter Default Trap" lesson. Next is the parallel XGBoost story.

---

# Slide 9: GKX-tuned XGBoost matches linear top; defaults misled us

- sklearn-style default (depth=4, eta=0.05, T=200) -> Sharpe 0.335
- Gu-Kelly-Xiu config (depth=1 stumps, eta=0.01, T=1000) -> 0.984
- Sign of IC stable across all 9 sensitivity-grid configs
- Domain-canonical default beats generic grid maximum

**Figure**: `fig_xgb_sensitivity.pdf` -- 9-config grid; sign-stable IC; GKX-tuned config dominates.

## Speaker Notes
XGBoost has the same story: defaults gave Sharpe 0.335, the literature-canonical Gu-Kelly-Xiu finance-tuned configuration gave 0.984. The 9-config sensitivity grid showed all configs had positive IC (sign-stable), but the Sharpe spread was 0.27 to 0.88 -- defaults landed in the middle of the grid, while GKX-tuned (depth=1 stumps, eta=0.01, 1000 trees) landed above the grid maximum. The lesson generalizes: at low SNR, before you declare a method failed, run at least one literature-canonical configuration. Next we look at the MLP, which does not have this saving grace.

---

# Slide 10: MLP single-seed result is initialization noise, not signal

- 5-seed mean Sharpe 0.412; std 0.30+ across seeds
- Best seed 0.648, worst seed 0.097 -- 8x range
- All five seeds fall below Ridge baseline 0.925
- Per-seed SNR (|mean|/std) = 1.33

**Figure**: `fig_mlp_seeds.pdf` -- All five seeds below Ridge floor; bars show per-seed Sharpe.

## Speaker Notes
We initially reported Rung 4 with a single seed (42) at Sharpe 0.347. The 5-seed audit shows that a single-seed report would have given anywhere from 0.097 to 0.648 depending on the random initialization. The 5-seed mean is 0.412 with very high relative noise; per-seed SNR (mean over std) is just 1.33. Critically, none of the five seeds beats Ridge. We therefore report 5-seed mean and standard deviation. This is also the first of four independent attempts to improve the MLP, none of which succeeds. Next is the consolidated walk-forward table.

---

# Slide 11: Walk-forward table -- linear and regularized dominate

- Top: Barra 1.134, LASSO 1.000, ElNet 0.997, XGB-GKX 0.984
- Mid: Ridge 0.966, FM 0.959, GAM tuned 0.828
- Rung 4 MLP single-seed 0.347; 5-seed mean 0.412
- Rung 5 MTL: 5b 0.229 (sign-flip), 5c 0.391, 5d 0.664 — all below Ridge floor

**Figure**: none (text-heavy condensed Tab 1; one slide; show the top half only)

## Speaker Notes
This is the condensed version of paper Table 1. The top six clustered within 0.20 Sharpe of each other; tuned non-linear methods (GAM n=4, XGB-GKX) sit just below the linear-regularized cluster; MLP and all Rung 5 variants land below the Ridge floor (0.925). 5b is the negative-transfer case — sign flips when adding the 3-month auxiliary task. 5d is the best Rung 5 variant at 0.664 — still well under Ridge. The MoE variants follow the same pattern and are excluded for brevity. Next we move from point estimates to path uncertainty.

---

# Slide 12: CPCV reveals all top-7 models' Sharpe bands overlap

- 15 paths per model, 8 models tested
- Top-6 mean Sharpe in [0.997, 1.086]
- 5-95% Sharpe bands all overlap heavily
- 6 of 8 models have P(Sharpe<0) = 0% across all paths

**Figure**: `fig_cpcv_sharpe.pdf` -- Boxplots overlap across top six; XGB has widest tail; FM/Barra tightest.

## Speaker Notes
This is the punchline. Once we account for path uncertainty, the single-path "+0.05 Sharpe" deltas we cared about in Table 1 are within path noise. Top-6 mean Sharpe sits in a 0.09 wide band; the 5-95% inner bands overlap heavily. Six of the eight CPCV-tested models have positive Sharpe in every one of the 15 paths -- the predictive signal is real, but the ranking of models is not statistically supported. XGB-GKX wins by mean (1.086) but has the widest tail (std 0.766) and the only P(Sharpe<0) of 6.7%. Fama-MacBeth has the tightest band. Next we look at the same data on the IC-Sharpe Pareto frontier.

---

# Slide 13: Pareto with bounds -- Ridge floor sits inside the cluster

- Models cluster in (IC, Sharpe) ~ (0.02-0.04, 0.7-1.1)
- Error bars: +/- 1 std across 15 CPCV paths
- Walk-forward Ridge baseline well inside the cluster
- For deployment: pick by lowest tail risk (FM or Barra)

**Figure**: `fig_pareto_with_bounds.pdf` -- Cluster + error bars; baseline-tuned Ridge benchmark sits among the cluster.

## Speaker Notes
The Pareto plot shows the same overlap visually: error bars on each model cover most of the neighbors. The orange Ridge floor lands well inside the cluster, meaning a baseline alpha=1 Ridge is statistically indistinguishable from the best models. The deployment recommendation flips from "highest mean Sharpe" (XGB-GKX) to "lowest tail risk" -- Fama-MacBeth (tightest band) or Barra-GK (highest 5th percentile, zero negative paths). Next we open three audit-cycle lessons that travel beyond this project.

---

# Slide 14: P7 Bagging hurt -- mean-of-predictions != mean-of-IC

- K=5 seed bagging: Sharpe 0.394 < 5-seed single mean 0.412
- Variance theorem holds in Pearson space; we evaluate in Spearman
- Mean-averaging pulls extreme rankings toward consensus middle
- Fix: rank-level averaging, or median-of-predictions

**Figure**: none (text-heavy diagnostic slide; could add bagging-vs-non-bagging bar chart if time)

## Speaker Notes
We tried K=5 seed bagging on the MLP to reduce per-seed variance. Bagged Sharpe was 0.394, lower than the mean-of-five-single-seeds Sharpe of 0.412. Diagnosis: the bagging variance-reduction theorem holds in Pearson/MSE space, but our evaluation metric is Spearman rank. Mean-averaging predictions creates a rank-space compromise that pulls extreme rankings toward consensus middle, exactly where LS quintile Sharpe lives. The fix is rank-level averaging (rankdata each prediction, then mean) or the median prediction. Lesson: match the ensemble operator to the evaluation metric's space. This is one of two methodology contributions of the paper. Next is the second.

---

# Slide 15: P8 HP search hurt -- bimodal HP picks signal validation noise

- 27-config grid search on MLP (hidden, lr, dropout)
- Tuned Sharpe 0.111 vs hardcoded default 0.347
- Validation-noise SE = 1/sqrt(10,000) ~ 0.011 ~ true IC
- Smoking gun: hidden_dim bimodal (16: 31%, 64: 62%)

**Figure**: none (text-heavy diagnostic slide; could embed the bimodal histogram if Eva makes one)

## Speaker Notes
We tried HP grid search via inner CV on the MLP -- 27 configs across hidden in {16,32,64}, lr in {1e-4,1e-3,1e-2}, dropout in {0,0.1,0.2}. Tuned Sharpe collapsed to 0.111 from 0.347. Diagnosis: with about 10,000 inner-validation observations, validation-noise SE is roughly 1/sqrt(10000) = 0.011, comparable to true IC magnitude (0.01-0.02). Grid search picks the HP whose validation realization was luckiest, not whose true IC is highest. The smoking-gun signature is bimodal HP selection across folds: hidden_dim was 16 (31%) and 64 (62%); learning rate was 1e-3 (47%) and 1e-2 (50%). Signal-driven HP search converges to a single mode; ours did not. Lesson: estimate validation-noise SE before HP search; if signal magnitude is comparable, prefer literature defaults.

---

# Slide 16: P5 Default trap -- Rung 3 GAM/XGB retracted from "failed"

- Both GAM (n=10) and XGB (sklearn defaults) initially looked broken
- Sweeps revealed defaults were 2-3x too complex for IC~0.01
- GAM(n=4) Sharpe 0.555 -> 0.828; XGB-GKX Sharpe 0.335 -> 0.984
- Lesson: run literature-canonical config before declaring failure

**Figure**: none (text slide synthesizing P5 from slides 8 + 9)

## Speaker Notes
Slides 8 and 9 each documented half of the same lesson: framework defaults (sklearn, pygam, XGBoost) target moderate-SNR generic ML benchmarks. At our IC~0.01 SNR, defaults overshoot optimal complexity. We retracted two "failed" results: GAM jumped from Sharpe 0.555 to 0.828 with n_splines=4, XGBoost jumped from 0.335 to 0.984 with the GKX-tuned config. The discipline that prevents this: before declaring a method failed, run at least one literature-canonical configuration plus a coarse sensitivity sweep. We log this as P5 -- the "Hyperparameter Default Trap." Next we look at why Rung 5 also fails, but for different reasons.

---

# Slide 17: Rung 5 fails three different ways: transfer, collapse, under-id

- 5b MTL (ret + ret_3m) IC -0.0036: textbook negative transfer
- MoE gate weights collapsed to ~uniform 0.33/0.33/0.34 across experts
- HMM K=3 has 68 free params on 60-month early folds (under-id)
- Each pathology is well-known (Yu 2020, Shazeer 2017, Hamilton 1989)

**Figure**: none (text slide; the three sub-points are the three bullets)

## Speaker Notes
Rung 5 failures are not just "more data needed" -- each variant has a specific structural pathology. First, 5b multi-task with ret + ret_3m gives IC -0.0036, textbook negative transfer because 1-month and 3-month forward returns conflict during 2022's regime shift. Uncertainty weighting only adjusts loss magnitude, not gradient direction; PCGrad would fix this but is 40+ LOC of second-order gradients. Second, the MoE gate collapsed -- across the three experts, gate weights ended at 0.33/0.33/0.34, no specialization. K=3 MoE is empirically equivalent to K=1 MTL because Shazeer's load-balancing regularizers were not added. Third, the HMM has 68 free parameters on early folds with 60 observations -- under-identified, producing noisy regime posteriors that destabilize the gate. Three independent failure modes, each with a known fix. Next we synthesize why complexity does not help here.

---

# Slide 18: Why complexity doesn't help -- features near-monotone, sample tight

- IC standard error 0.0063 across 58 months at our scale
- Detected IC 0.018-0.029 is ~3sigma; pairwise deltas 0.002-0.005 are noise
- 27-param linear and 3,000-param MLP land in same Sharpe band
- GAM peak at 1-2 internal knots: factors are near-monotone

**Figure**: `fig_effective_df.pdf` -- Effective DoF spans 100x range; Sharpe spans <0.3 within the cluster.

## Speaker Notes
The arithmetic is direct. With 442 stocks and 58 test months, time-averaged IC standard error is about 0.0063. Detected IC values of 0.018 to 0.029 are 3-sigma above zero (signal exists), but pairwise model differences of 0.002 to 0.005 fall within noise. Two complementary arguments: structurally, our features are near-monotone factor aggregates that 1-2 internal knots in a GAM spline already capture (the n_splines=4 finding); statistically, signal magnitude is below the model-selection noise floor. The effective-DoF chart visualizes this: 100x range in capacity, identical Sharpe band. This is consistent with -- not contradicting -- Gu-Kelly-Xiu, who saw ML wins on a much larger universe. Next is limitations.

---

# Slide 19: Limitations bound the result honestly

- Survivorship bias: frozen 442-ticker snapshot, not point-in-time
- Sample size: ~25,000 obs is detect-IC-positive, not pairwise-decisive
- Regime composition: 2020 COVID + 2022 inflation are stress test
- has_positive_earnings filter bias: documented; <0.005 IC drift expected

**Figure**: none (text slide)

## Speaker Notes
Four key limitations. Survivorship bias from the frozen S&P 500 snapshot likely depresses all results uniformly without flipping the ranking; quantification is deferred. Sample size of 25,000 panel observations is enough to detect IC>0 for the best models but not enough for pairwise comparisons -- exactly what CPCV makes explicit. Regime composition: 2020 plus 2022 plus 2023 makes this a stress test, not a benign window. We document a known has_positive_earnings filter bias whose expected drift is below CPCV path noise. Plus six other items in the paper: validation-split granularity, CPCV not run on Rung 4-5, HP search not exhaustive, transaction-cost simplification, no live trading, look-ahead in 2c. Next is the conclusion.

---

# Slide 20: Conclusion -- deploy linear, raise sample size for ML

- Adding complexity does not improve cross-sectional return prediction here
- Deploy: Fama-MacBeth or Barra-GK (lowest tail risk)
- Audit lessons P5/P7/P8 generalize beyond this paper
- Future: at what sample size does complexity start to help?

**Figure**: none (text slide)

## Speaker Notes
Three takeaways. First, on this 119-month, 442-stock, 27-feature panel, no complexity rung statistically beats linear. Second, for deployment choose by tail risk: Fama-MacBeth and Grinold-Kahn Barra both hit zero negative-Sharpe paths in CPCV. Third, the audit cycle yielded three methodology lessons that generalize: P5 hyperparameter default trap, P7 bagging metric-space mismatch, P8 HP search on low-SNR validation. Future work bracketed: our 25,000 panel observations is below the threshold where ML helps; Gu-Kelly-Xiu's 1.5 million is above. The natural next study identifies the boundary. End of main slides; next two are backup.

---

# Slide 21: Backup -- HP inventory across the ladder

- Rungs 1a/1c: no hyperparameters (closed-form)
- Rungs 2a-d: alpha grid, l1_ratio grid, all swept
- Rung 3a: n_splines swept 3-20; Rung 3b: 9-grid + GKX
- Rung 4: 5 seeds + hidden sweep + HP search (all logged)

**Figure**: none (this slide is the condensed Appendix A Tab 4 -- HP inventory)

## Speaker Notes
Backup slide for "what was tested" questions. Rung 1 closed-form models have no HPs. Rung 2 regularization strengths are all swept on inner TimeSeriesSplit with Spearman-IC scoring matching the outer metric. Rung 3a GAM n_splines is swept 3 through 20 plus K-fold lambda vs GCV (negative result: K-fold did not beat GCV). Rung 3b XGBoost has the 9-config grid plus the GKX literature-canonical configuration. Rung 4 MLP has the seed sensitivity sweep, hidden-dim sweep, K=5 seed bagging (negative), inner-CV HP search (negative). All audit results frozen in `audits/` directory. Use this slide to deflect "did you try X" questions.

---

# Slide 22: Backup -- LASSO selection frequency confirms feature concentration

- Walk-forward LASSO selection across 58 outer folds
- IVOL 67%, Gross Profitability 62%, Beta 53% (top 3)
- Short Interest Ratio + Momentum 12-1 round out top 5
- Concentration consistent with GAM near-monotone finding

**Figure**: `fig_lasso_freq.pdf` -- Three features survive >50% of folds; signal concentrated in a small subset.

## Speaker Notes
Backup slide if asked about feature importance. LASSO selection frequency across 58 walk-forward folds is a model-free proxy for permutation importance, and it agrees with what GAM found: signal lives in a small subset of features. IVOL, Gross Profitability, and Beta survive in over half of folds; Short Interest Ratio and Momentum 12-1 round out the top 5. Permutation importance for the top-three models (paper Fig 11) ranks IVOL and GP highest across all three. The concentration of predictive signal in a small subset is consistent with GAM peaking at 1-2 internal knots: most features contribute linearly or not at all.

---

## Figure Usage Audit

Every PDF in `/Users/evanolott/Desktop/MA751-Project/paper/MA751-paper/figures/` is accounted for below:

| Figure | Status | Slide / Reason |
|---|---|---|
| `fig_ladder.pdf` | Used | Slide 3 -- the 5-rung complexity ladder framework |
| `fig_v2_vs_v3.pdf` | Used | Slide 4 -- justifies V3 (27-feature) panel choice over V2 |
| `fig_gam_n_splines.pdf` | Used | Slide 8 -- defaults 2x too complex; peak at n=4 |
| `fig_xgb_sensitivity.pdf` | Used | Slide 9 -- GKX-tuned config dominates 9-grid |
| `fig_mlp_seeds.pdf` | Used | Slide 10 -- all 5 seeds below Ridge floor |
| `fig_cpcv_sharpe.pdf` | Used | Slide 12 -- top-6 boxplot overlap (the punchline) |
| `fig_pareto_with_bounds.pdf` | Used | Slide 13 -- IC-Sharpe Pareto with CPCV +/-1sigma error bars |
| `fig_effective_df.pdf` | Used | Slide 18 -- 100x DoF range, identical Sharpe band |
| `fig_lasso_freq.pdf` | Used | Slide 22 (backup) -- feature concentration matches GAM finding |
| `fig_cumret.pdf` | Not used | Cumulative-return time series is a corroborating exhibit for slide 7's "models cluster" message but the CPCV boxplot (slide 12) carries that message statistically. Keeping the deck under 22 slides. Strong candidate to add if presentation expands to 25 slides or as a second backup. |
| `fig_correlation.pdf` | Not used | Feature correlation heatmap motivates regularized estimators, but slide 4 already justifies the V3 panel and slide 7 jumps directly to results. Could be added to slide 4 as a small inset if desired. |
| `fig_pca.pdf` | Not used | PCA scree + PC1xPC2 was an original-plan deliverable but the deck does not need feature-space exploration to make the headline argument. Reserve as a third backup if feature-structure questions arise. |
| `fig_perm_importance.pdf` | Not used | Permutation importance for top-3 models duplicates the LASSO-frequency message on slide 22. Keep slide 22 simpler with one chart; offer perm-importance as a fourth backup. |
| `fig_regime_timeline.pdf` | Not used | HMM K=3 regime posteriors over time is relevant only to Rung 5 MoE, which slide 17 covers verbally. The chart would force a separate slide on a Rung-5-specific architecture detail and the deck is already at 22 slides. Reserve as a deep-dive backup if a Rung 5 question emerges. |

**Total figures**: 14 PDFs (each with a PNG twin). **Used**: 9. **Reserved as backup / not strictly needed**: 5.

---

## Open TODOs for Eva before Google Slides import

1. **Slide 11 Rung 5 row**: fill exact Sharpe values from `output/results_5*_uw.parquet` (currently marked TODO).
2. **Slide 14 (P7 bagging)**: optional bagging-vs-non-bagging bar chart -- 5 single-seed Sharpe + bagged Sharpe = 6 bars. Easy to make from `output/rung4_seed_bagging_*.csv`.
3. **Slide 15 (P8 HP search)**: optional bimodal histogram of chosen HPs across folds. Data in `output/rung4_hp_search_*.csv`.
4. **Slide 5 / Slide 6**: optional schematic figures for walk-forward and CPCV protocols. TikZ or matplotlib hand-drawn.
5. **Slide 11**: trim Tab 1 to top half only for slide-friendly density. Full table goes in handout / appendix.
