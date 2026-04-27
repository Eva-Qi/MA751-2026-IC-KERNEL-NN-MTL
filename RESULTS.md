# Final Results — MA751 Cross-Sectional Return Prediction

> Single source of truth for paper-cited numbers. Last updated 2026-04-27.
> All numbers reproducible from `output/*.csv` and `output/*.parquet`.

---

## Headline ranking (CPCV 15-path, V3 panel)

Source: `output/cpcv_summary.csv`

| Rank | Model | Sharpe mean | Sharpe std | 5\% | 95\% | P(Sharpe<0) | Final config |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | 3b XGBoost (GKX) | 1.086 | 0.766 | 0.05 | 2.15 | 6.7% | depth=1, lr=0.01, n=1000 |
| 2 | 1c Fama-MacBeth | 1.071 | **0.494** | 0.34 | 1.68 | **0%** | per-month CS OLS, time-avg |
| 3 | 1d Barra (Grinold-Kahn) | 1.058 | 0.544 | **0.43** | 1.86 | **0%** | Ledoit-Wolf shrinkage |
| 4 | 1a OLS | 1.040 | 0.618 | 0.28 | 1.94 | 0% | pooled cross-section |
| 5 | 2a LASSO | 1.037 | 0.618 | 0.28 | 1.94 | 0% | α via TimeSeriesSplit(K=5) |
| 6 | 2b Ridge | 1.008 | 0.622 | 0.20 | 1.91 | 0% | α via TimeSeriesSplit, scoring=Spearman |
| 7 | 2d Elastic Net | 0.997 | 0.592 | 0.28 | 1.94 | 0% | α + l1_ratio via TimeSeriesSplit |
| 8 | 1b IC-Ensemble | 0.840 | 0.559 | 0.10 | 1.64 | 0% | univariate, IC-weighted |

**Headline finding**: top-7 models' [5\%, 95\%] Sharpe bands all overlap. No model statistically dominates at our 119-month / 442-stock sample size.

---

## Walk-forward (single-path) ranking (V3 panel, 58 test months)

Source: `output/rung12_v3_summary.csv`

| Rung | Model | IC mean | IC-IR | Hit % | LS Sharpe | Final config |
|---|---|---:|---:|---:|---:|---|
| baseline | Zero predictor | +0.0002 | — | 53.4 | 0.262 | random noise |
| baseline | Ridge α=1 (linear floor) | +0.0164 | — | 53.4 | 0.925 | fixed α=1 |
| 1a | OLS | +0.0164 | 0.106 | 53.4 | 0.924 | — |
| 1b | IC-Ensemble | +0.0208 | 0.094 | 50.0 | 0.844 | — |
| 1c | Fama-MacBeth | +0.0259 | 0.165 | 63.8 | 0.959 | — |
| **1d** | **Barra (Grinold-Kahn)** | **+0.0283** | **0.229** | **60.3** | **1.134** | Ledoit-Wolf shrunk Σ_f |
| 2a | LASSO | +0.0183 | 0.118 | 53.4 | 1.000 | α grid logspace(-4,1,50) |
| 2b | Ridge | +0.0177 | 0.105 | 55.2 | 0.966 | α grid logspace(-4,4,50) |
| 2d | Elastic Net | +0.0179 | 0.116 | 53.4 | 0.997 | l1_ratio grid 7 points |
| 2e | Adaptive LASSO | +0.0164 | 0.106 | 53.4 | 0.924 | weights 1/\|β_OLS\| |
| 3a | GAM (tuned) | +0.0141 | 0.111 | 51.7 | 0.828 | **n_splines=4** |
| 3a | GAM (default) | +0.0068 | 0.066 | 53.4 | 0.555 | n_splines=10 |
| 3b | XGBoost (GKX-tuned) | +0.0177 | 0.146 | 62.1 | 0.984 | **depth=1, lr=0.01, n=1000** |
| 3b | XGBoost (default) | +0.0018 | 0.018 | 50.0 | 0.335 | depth=4, lr=0.05, n=200 |
| 4 | MLP single-task (seed=42) | +0.0007 | 0.007 | 51.7 | 0.347 | hidden=64, lr=1e-3 |
| 4 | MLP 5-seed mean | +0.0109 | — | — | 0.412 | bagging across 5 seeds |
| 5b† | MTL (ret + ret_3m) | −0.0036 | −0.028 | 50.0 | 0.229 | UW loss, neg-transfer sign-flip |
| 5c† | MTL (ret + vol) | +0.0061 | +0.043 | 48.3 | 0.391 | UW loss, vol head corr 0.50 |
| 5d† | MTL (ret + ret_3m + vol) | +0.0173 | +0.117 | 58.6 | 0.664 | UW loss, best Rung 5 |

†Rung 5 numbers are V2 14-feature panel; not retuned for V3 (paper §5.6 + Tab 1 footnote). Source: `output/ablation_summary_uncertainty.csv`. Regime-Gated MoE and Enhanced MoE follow the same dominated pattern; not summarized here.

---

## Sensitivity-study results

| Study | Result | Output | Audit script |
|---|---|---|---|
| GAM `n_splines` sweep (10 configs) | n=4 wins Sharpe (0.828); n=5 wins IC (0.0146); default n=10 over-fits | `output/rung3a_n_splines_sweep.csv` + `output/rung3a_n_splines_finetune.csv` | `audits/rung3a_gam_audit.py`, `audits/rung3a_gam_finetune.py` |
| GAM K-fold λ vs GCV | Negative — K=5 inner CV did NOT improve over GCV | `output/rung3a_gam_kfold_lambda_*.csv` | `audits/rung3a_gam_kfold_lambda.py` |
| XGBoost 9-config sensitivity grid | All 9 positive IC, sign stable; GKX-tuned wins | `output/rung3b_sensitivity_grid.csv`, `output/rung3b_gkx_tuned.csv` | `audits/rung3b_audit.py` |
| MLP 5-seed sensitivity | SNR=1.33; **all 5 seeds < Ridge floor** | `output/rung5_seed_sensitivity.csv` | `audits/rung5_audit.py` |
| MLP hidden_dim sweep | hidden=8 ≥ hidden=64 → over-parameterized | `output/rung5_hidden_sweep.csv` | `audits/rung5_audit.py` |
| K=5 seed bagging | Negative — Sharpe dropped 0.412→0.394 (P7 metric mismatch) | `output/rung4_seed_bagging_*.csv` | `audits/rung45_seed_bagging.py` |
| Inner-CV HP search | Negative — Sharpe collapsed 0.347→0.111 (P8 over-tune) | `output/rung4_hp_search_*.csv` | `audits/rung4_hp_search.py` |
| Rung 5 MTL planned aux | All combo_a–e negative IC | `output/ablation_summary_planned.csv` | `audits/rung5_planned.py` |
| Rung 5 MTL combined aux | All combo_a–e negative IC | `output/ablation_summary_combined.csv` | `audits/rung5_combined.py` |

---

## Code-bug fixes (mid-audit)

| Fix | Before | After |
|---|---|---|
| **P0 #1** Barra duplicate definition (line 368 was overriding line 299 True Barra) | V2 Sharpe 0.497, V3 Sharpe 0.746 | V2 0.982, V3 **1.134** |
| **P0 #2** `has_positive_earnings` post-filter bias | filter dropped EY<−1 rows entirely | sign captured pre-filter |
| **P1 #1** `compute_power(sigma=0.10)` hardcoded | only one σ supported | accepts `ic_std` kwarg |

---

## How to reproduce

```bash
# Active pipeline (in project root):
python run_rung12_v2.py            # produces V2 panel results
python scripts/rung12_v3.py        # produces V3 panel results (Tab 1 of paper)
python run_cpcv_all_models.py      # produces CPCV 15-path distribution (Tab 3)
python main.py                      # produces Rung 4/5 MLP/MTL results

# Audit snapshots (frozen in audits/):
cd audits && python rung3a_gam_audit.py        # GAM coarse sweep
cd audits && python rung3a_gam_finetune.py     # GAM fine-tune
cd audits && python rung3b_audit.py            # XGB sensitivity + GKX
# (etc — see audits/README.md for the full list)
```

---

## File-to-paper mapping

| Paper section | Source CSVs | Source scripts |
|---|---|---|
| §4 Tab 1 (walk-forward main results) | `output/rung12_v3_summary.csv` | `run_rung12_v2.py` + `scripts/rung12_v3.py` |
| §5.1 Fig 3 (GAM sweep) | `output/rung3a_n_splines_*.csv` | `audits/rung3a_gam_*.py` |
| §5.1 Fig 4 (XGB heatmap) | `output/rung3b_sensitivity_grid.csv` | `audits/rung3b_audit.py` |
| §5.2 Fig 5 (MLP seeds) | `output/rung5_seed_sensitivity.csv` | `audits/rung5_audit.py` |
| §6 Tab 3 + Figs 6,7 (CPCV) | `output/cpcv_results.parquet` + `output/cpcv_summary.csv` | `cpcv_harness.py` + `run_cpcv_all_models.py` |
