# MA751 Final Project — Cross-Sectional Stock Return Prediction

6-rung complexity ladder testing whether added model capacity improves cross-sectional monthly stock-return prediction on a 442-stock S&P~500 panel (2015--2024).

## Headline finding

At 442 stocks × 119 months × 27 features, **adding model complexity does not improve cross-sectional return prediction in a statistically meaningful way**. CPCV (Combinatorial Purged K-fold, 15 paths × 8 models) shows all top-7 models' 5-95% Sharpe bands overlap. **Fama-MacBeth and Grinold-Kahn Barra are the most robust point estimates** (CPCV mean Sharpe 1.071 / 1.058, both with $P(\text{Sharpe}{<}0)=0\%$).

See `RESULTS.md` for the canonical numbers and `PROGRESS.md` for the milestone timeline.

## Project structure

```
.
├── config.py                       # feature lists, panel constants
├── load_data.py                    # V2 panel assembly, NaN/winsorize
├── metrics.py                      # IC, IC-IR, Hit, LS Sharpe (canonical)
├── statistical_tests.py            # paired t-test, DM, BHY, power analysis
├── regime.py                       # HMM regime identification (Rung 6 MoE)
│
├── run_rung12_v2.py                # Rung 1+2 driver (8 model variants)
├── main.py                         # Rung 4/5 MLP/MTL driver
├── regmtl.py                       # Regime-Gated MoE (Rung 6)
├── regmtl_enhanced.py              # Enhanced MoE (Rung 6)
├── run_cpcv_all_models.py          # CPCV evaluation driver
│
├── pipeline/                       # Data-build scripts (run-once)
│   ├── download_wrds_tier12.py     # WRDS data download
│   ├── wrds_factor_builder.py      # Compustat → factor panel
│   ├── wrds_new_features.py        # IBES + Phase-2 features
│   └── compute_derived_features.py # Beta, IVOL, realized vol
│
├── scripts/                        # Analysis utilities
│   ├── rung12_v3.py                # V3 panel driver (paper Tab 1)
│   ├── paired_ttest_miss_flags.py  # Missingness-flag t-test
│   ├── ff3_alpha_test.py           # FF3 alpha regression
│   └── compute_turnover.py + compute_ls_hit_rate.py
│
├── src/                            # Shared modules
│   ├── harness/cpcv_harness.py     # CPCV harness (López de Prado 2018)
│   ├── models/mlp_audit.py         # Consolidated MLP class + train fn
│   ├── data_pipeline/              # taxonomy_map.py
│   └── factor_library/             # academic_factors.py
│
├── audits/                         # Frozen audit / ablation snapshots
│   └── (12 scripts + READMEs; linguist-generated in .gitattributes)
│
├── data/                           # Panel data (gitignored)
├── output/                         # Result artifacts (~76 files + archive/)
├── paper/                          # LaTeX paper draft (gitignored)
├── docs/                           # PROJECT_STORY.md + FIGURES_TODO.md + archive/
│
├── RESULTS.md                      # Canonical paper-cited numbers
├── PROGRESS.md                     # Milestone timeline (newest first)
└── README.md                       # this file
```

## 6-rung complexity ladder (12 model variants)

| Rung | Variants | Best (CPCV mean Sharpe) |
|---|---|---|
| 1 — Linear | OLS, IC-Ensemble, Fama-MacBeth, Barra (Grinold-Kahn) | **1.071 (FM)** |
| 2 — Regularized | LASSO, Ridge, Elastic Net, Adaptive LASSO | 1.037 (LASSO) |
| 3 — Smooth non-linear | GAM ($n_{\text{splines}}{=}4$), XGBoost (GKX-tuned) | **1.086 (XGB)** |
| 4 — Single-task MLP | hidden=64, ReLU, dropout 0.10, Huber loss | 0.412 (5-seed mean — all $<$ Ridge) |
| 5 — MTL | 5b/5c/5d MTL (shared encoder + per-task heads, Kendall-Gal-Cipolla UW) | dominated by linear |
| 6 — MoE | Regime-Gated MoE, Enhanced MoE (K=3 expert MLPs, HMM-gated softmax) | dominated by linear |

## Data

- **Universe**: 442 S&P~500 tickers (frozen membership snapshot)
- **Window**: 119 months, 2015-01 → 2024-11; test window 58 months (2020-02 → 2024-11)
- **Sources**: CRSP monthly + daily, Compustat annual + short interest, IBES, TFN 13F, FF3/5
- **Features**: 27 (= 22 V3 firm-level signals + 5 missingness flags). See `output/README.md` for the full feature list.
- **Walk-forward**: expanding window, 60-month minimum train, 1-month purge
- **Eval**: per-month Spearman IC + long-short quintile Sharpe (annualized)

## How to reproduce

```bash
# Active pipeline:
python run_rung12_v2.py            # produces V2 panel results
python scripts/rung12_v3.py        # produces V3 panel results (paper Tab 1)
python run_cpcv_all_models.py      # produces CPCV 15-path distribution (paper Tab 3)
python main.py                     # produces Rung 4/5 MLP/MTL results

# Audit / ablation snapshots (frozen, see audits/):
cd audits && python rung3a_gam_sweep.py
cd audits && python rung3b_audit.py
# ... etc — see audits/README.md
```

## Setup

```bash
pip install -r requirements.txt
```

WRDS access required for raw data download (`pipeline/download_wrds_tier12.py`); auth via `~/.pgpass` or `WRDS_USER`/`WRDS_PASS` env vars.

## See also

- `RESULTS.md` — canonical numbers, file→paper mapping
- `PROGRESS.md` — newest-first milestone timeline
- `docs/PROJECT_STORY.md` — full master narrative (~1,500 lines)
- `docs/FIGURES_TODO.md` — figure inventory
- `audits/README.md` — frozen audit-script inventory
- `output/README.md` — output-file naming + V2/V3 feature lists
