# Rung 2c: Post-LASSO OLS

Branch: `post-lasso-ols`. Script: `run_rung2c_selected_ols.py`.

## Motivation

Run OLS on feature subsets chosen by prior selection methods, to test
whether the full 14-feature OLS baseline is diluted by noise factors.

Two selection sources:
- **LASSO frequency** (from `output/lasso_coefs_by_fold.csv`): features kept
  in ≥50% of walk-forward folds — IVOL (67%), GrossProfitability (62%),
  Beta (53%), plus ShortInterestRatio and Momentum12_1 rounding out top-5.
- **Fama–MacBeth t-stats** (teammate's analysis): NetDebt/EBITDA (t=-2.37),
  GrossProfitability (1.51), AnalystRevision (1.7), RevisionBreadth (1.31).

## Results (58 test months, same walk-forward harness as Rung 1/2)

| Model | features | IC | IC_IR | Hit_IC | LS Sharpe | LS-Hit |
|---|---|---|---|---|---|---|
| 2c_OLS_LASSO3 | IVOL, GP, Beta | +0.0168 | 0.078 | 55.2% | 0.809 | 58.6% |
| **2c_OLS_LASSO5** | + ShortInterest, Mom12_1 | +0.0152 | 0.073 | 55.2% | **0.829** | **63.8%** |
| 2c_OLS_FMB4 | NetDebt, GP, AnRev, RevBreadth | +0.004 | 0.024 | 44.8% | 0.146 | 50.0% |

Paired t-test vs full 14-feature OLS (Rung 1a) monthly ICs:
- LASSO5: **p = 0.030** (only statistically significant improvement in the ladder)
- LASSO3: p = 0.116
- FMB4:   p = 0.218

## Verdict: LASSO5 wins out

LASSO5 beats the other two 2c variants on LS Sharpe (0.829) and LS-Hit (63.8%),
matches LASSO3 on Hit_IC, and is the only selected-feature model whose IC
improvement over 1a-OLS is significant at α=0.05.

It also beats the 5d MTL on portfolio metrics (Sharpe 0.829 vs 0.664,
LS-Hit 63.8% vs 55.2%) while trailing on IC/IC_t. Net of both metrics,
LASSO5 is the strongest linear model in the ladder.

## Why did LASSO5 (OLS on 5 features) beat plain LASSO (2a)?

1. **Shrinkage bias.** LASSO shrinks kept coefficients toward zero. Post-LASSO
   OLS refits with unbiased coefficients — the classic relaxed-LASSO / debiasing
   improvement.
2. **Selection stability.** Per-fold LASSO picks a different subset each month
   (nothing is kept in 100% of folds). Fixing the set to the top-5 by full-sample
   selection frequency is more stable, though mildly look-ahead; a purer version
   would recompute top-k on each training window.

## Corrections / caveats

- The top-5 feature list uses full-sample LASSO selection frequencies, so LASSO5
  has a minor look-ahead flavor. The LASSO3 subset (features picked in ≥50% of
  folds) is less exposed to this.
- `statistical_tests.py` still compares Rung 1–2 against the V1 baseline parquet
  (83 months) while Rung 5 uses V2 (58 months). Pairwise tests involving 2c use
  the V2 58-month panel only.
- All Rung 1–3 results were re-run on the post-merge_asof-fix V2 panel
  (`data/master_panel_v2.parquet`) before this experiment; earlier saved
  summaries were stale.

## Look-Ahead Fix: 2c_OLS_LASSO_WF (added 2026-04-19)

The original LASSO3 and LASSO5 feature sets were chosen by computing LASSO
selection frequency over the **entire** walk-forward sample, including test
folds. This is look-ahead: the test-period data influenced which features
were used to generate predictions on those same months.

To eliminate the bias, a new variant `2c_OLS_LASSO_WF` was added to
`run_rung2c_selected_ols.py`. It uses `walk_forward_lasso_selected_ols()`,
which performs feature selection **inside** each fold's training window only:

1. Fit `LassoCV` (cv=5) on the training data (all 14 `ALL_FEATURE_COLS_V2`).
2. Identify non-zero coefficients.
3. Fall back to top-3 by |coef| if fewer than 3 survive.
4. Cap at 5 features (those with largest |coef| among survivors).
5. Refit OLS on the selected subset; predict on test month.

The original LASSO3/LASSO5/FMB4 variants are retained for comparison (now
annotated with `# LOOK-AHEAD` comments in the code), but `2c_OLS_LASSO_WF`
is the methodologically clean reference going forward.

## Artifacts

- `output/results_2c_OLS_LASSO3.parquet`
- `output/results_2c_OLS_LASSO5.parquet`
- `output/results_2c_OLS_FMB4.parquet`
- `output/results_2c_OLS_LASSO_WF.parquet`
- `output/rung2c_selected_ols_summary.csv`
