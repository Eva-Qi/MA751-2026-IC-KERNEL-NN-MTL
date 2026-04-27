# `audits/` — Frozen Audit & Ablation Snapshots

This directory contains one-off scripts that ran during the audit cycle.
**Their results are baked into `output/*.csv` / `output/*.parquet` and cited
in the paper.** None of these scripts is imported by the active pipeline.

If you need to re-run any of them, note that `ROOT = Path(__file__).resolve().parent`
inside each script now points at `audits/` rather than the project root —
add `.parent` once more, or pre-pend the project root to `sys.path`
manually.

GitHub treats this directory as `linguist-generated` (see `.gitattributes`):
the files are not counted in language statistics and are collapsed by
default in pull-request diffs.

---

## Inventory

| File | Purpose | Output |
|---|---|---|
| `rung3_gam.py` | Original Rung 3a GAM driver (default `n_splines=10`); produces the "default config" baseline for Sec.~5.1 default-vs-tuned comparison | `output/results_3a_*.parquet`, `output/gam_summary.csv` |
| `rung3b_gbm.py` | Original Rung 3b XGBoost driver (default `depth=4, lr=0.05, n=200`); produces the "default config" baseline | `output/results_3b_*.parquet` |
| `rung3a_gam_audit.py` | n_splines coarse sweep $\{3, 5, 10, 15, 20\}$ | `output/rung3a_n_splines_sweep.csv` |
| `rung3a_gam_finetune.py` | n_splines fine-tune $\{4, 6, 7, 8, 9\}$ | `output/rung3a_n_splines_finetune.csv` |
| `rung3a_gam_kfold_lambda.py` | Replace pygam GCV with TimeSeriesSplit($K{=}5$) inner CV for $\lambda$ — negative result | `output/rung3a_gam_kfold_lambda_*.csv` |
| `rung3b_audit.py` | XGBoost 9-config sensitivity grid + Gu-Kelly-Xiu-tuned variant | `output/rung3b_sensitivity_grid.csv`, `output/rung3b_gkx_tuned.csv` |
| `rung4_hp_search.py` | Inner-CV HP grid search over hidden/lr/dropout — **negative result** (Sec.~5.4 P8 lesson) | `output/rung4_hp_search_*.csv` |
| `rung45_seed_bagging.py` | $K{=}5$ seed mean-prediction bagging — **negative result** (Sec.~5.3 P7 lesson) | `output/rung4_seed_bagging_*.csv` |
| `rung5_audit.py` | Rung 4 MLP seed sensitivity (5 seeds) + hidden-dim sweep + naive baselines | `output/rung5_seed_sensitivity.csv`, `output/rung5_hidden_sweep.csv` |
| `rung5_planned.py` | Rung 5 MTL ablation with rank + sector_rel auxiliary targets | `output/ablation_summary_planned.csv` |
| `rung5_combined.py` | Rung 5 MTL ablation across all aux-target combinations | `output/ablation_summary_combined.csv` |
| `lasso_v3_selection.py` | LASSO frequency analysis on V3 features — informed Enhanced MoE feature subset | `output/lasso_v3_selection_summary.csv`, `output/v3_feature_correlation.csv` |

---

## Active pipeline (lives in project root)

For reference, the actively-imported / actively-run files are:

- `config.py`, `load_data.py`, `metrics.py`, `walk_forward.py`, `statistical_tests.py`
- `run_rung12_v2.py` (Rung 1+2 driver)
- `scripts/rung12_v3.py` (V3 panel driver)
- `main.py` (Rung 4/5 MTL driver)
- `regmtl.py` (Regime MoE)
- `regmtl_enhanced.py` (Enhanced MoE)
- `cpcv_harness.py` + `run_cpcv_all_models.py` (CPCV)
- `src/models/mlp_audit.py` (consolidated MLP class)
