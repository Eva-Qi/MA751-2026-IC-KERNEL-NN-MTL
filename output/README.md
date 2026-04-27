# `output/` — Result Artifacts

Living results directory: walk-forward + CPCV + audit-cycle outputs. Total ~41 MB after 2026-04-27 cleanup. Cleaned up per code-council Part 12 §P9 (dead-code accumulation) — orphans deleted, logs moved to `archive/`.

## File-naming convention

- `monthly_<rung>_<model>_<panel>.csv` — per-month IC + LS portfolio for one model on one panel
- `results_<rung>_<model>_<panel>.parquet` — same but row-level (date, ticker, y_pred, y_true)
- `<rung>_<dimension>_summary.csv` — sweep/sensitivity summaries
- `<rung>_<study>_diag.csv` — per-fold diagnostics for one study
- `<group>_summary.csv` — cross-model summary tables

Panels:
- `_v2` = 14 firm features + 5 missingness flags (V2 panel)
- `_v3` = 22 firm features + 5 missingness flags (V3 panel) — used for paper

## Canonical paper-cited results

| Paper section | Source |
|---|---|
| §4 Tab 1 — walk-forward main results | `rung12_v3_summary.csv` |
| §5.1 Fig 3 — GAM `n_splines` sweep | `rung3a_n_splines_sweep.csv` + `rung3a_n_splines_finetune.csv` |
| §5.1 Fig 4 — XGBoost sensitivity | `rung3b_sensitivity_grid.csv`, `rung3b_gkx_tuned.csv` |
| §5.2 Fig 5 — MLP seed sensitivity | `rung5_seed_sensitivity.csv` |
| §5.3 — bagging negative result | `rung4_seed_bagging_*.csv` |
| §5.4 — HP search negative result | `rung4_hp_search_*.csv` |
| §5.1 — K-fold λ negative result | `rung3a_gam_kfold_lambda_*.csv` |
| §6 Tab 3 + Figs 6, 7 — CPCV | `cpcv_results.parquet` + `cpcv_summary.csv` |
| §4 Fig 8 — cumulative LS returns | `results_*_v3.parquet` (top 5 models) |

See `RESULTS.md` at project root for full file→paper mapping.

## Subdirectories

- `archive/` — run logs (cpcv_log, rung12_v2_rerun_log, audit-script logs) + early diagnostic plots. Not paper-cited; kept for reproducibility evidence.

## Cleanup history (2026-04-27)

**Deleted** (20 orphan files, ~3 MB):
- `interaction_features_test.csv`, `interaction_coefs.csv`, `interaction_summary.csv` — script `scripts/interaction_features_test.py` was deleted (one-off experiment, not paper-cited)
- `sector_adjusted_*.csv` (2 files) — `scripts/sector_adjusted_test.py` deleted
- `rolling_ic_*.csv/png` (4 files) — `scripts/rolling_ic.py` deleted
- `monthly_1d_CorrAdjIC_*.csv` (2), `results_1d_CorrAdjIC_*.parquet` (2) — old heuristic Barra. Paper cites True Barra (Grinold-Kahn) under `_1d_Barra_*` after the P0 #1 fix
- `results_2c_OLS_FMB4.parquet`, `results_2c_OLS_LASSO3.parquet`, `results_2c_OLS_LASSO5.parquet` — LOOK-AHEAD contaminated; producing script (`run_rung2c_selected_ols.py`) deleted
- `monthly_2_LASSO_v2.csv`, `results_2_LASSO_v2.parquet` — superseded by `_2a_LASSO_` naming convention
- `results_5a_plan.parquet`, `results_5a_uw.parquet` — relabeled to Rung 4; have `results_rung4_plan.parquet`, `results_rung4_uw.parquet`

**Moved to `archive/`** (12 files, ~1.5 MB):
- 11 `*_log.txt` files (cpcv, rung12 rerun, audit script logs)
- `gam_partial_dependence.png` (early GAM diagnostic, not in paper)
