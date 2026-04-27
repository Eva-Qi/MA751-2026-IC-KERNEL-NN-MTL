# `output/` — Result Artifacts

Living results directory: walk-forward + CPCV + audit-cycle outputs. Total ~41 MB.

## File-naming convention

- `monthly_<rung>_<model>_<panel>.csv` — per-month IC + LS portfolio for one model on one panel
- `results_<rung>_<model>_<panel>.parquet` — same but row-level (date, ticker, y_pred, y_true)
- `<rung>_<dimension>_summary.csv` — sweep/sensitivity summaries
- `<rung>_<study>_diag.csv` — per-fold diagnostics for one study
- `<group>_summary.csv` — cross-model summary tables

## Feature panels

### V2 — 14 firm features + 5 missingness flags = 19 columns

**Classical factors (6)** — Compustat + CRSP-derived:
- `EarningsYield_zscore` (ib / mktcap)
- `GrossProfitability_zscore` (gp / at)
- `AssetGrowth_zscore` (Δassets / lag-assets)
- `Accruals_zscore` (working-capital change)
- `Momentum12_1_zscore` (12-1 momentum)
- `NetDebtEBITDA_zscore` ((debt − cash) / EBITDA)

**Analyst signals (4)** — IBES:
- `SUE_zscore` (standardized unexpected earnings)
- `AnalystRevision_zscore` (consensus revision)
- `AnalystDispersion_zscore` (analyst std)
- `RevisionBreadth_zscore` (up vs. down ratio)

**Risk (2)** — CRSP daily-derived:
- `Beta_zscore` (252-day market beta)
- `IVOL_zscore` (60-day CAPM residual std)

**Alt-data (2)** — Compustat short interest, TFN 13F:
- `ShortInterestRatio_zscore`
- `InstOwnershipChg_zscore`

**Missingness flags (5)** — paired with imputed features above:
- `has_sue`, `has_short_interest`, `has_inst_ownership`, `has_analyst_consensus`, `has_positive_earnings`

### V3 — 22 firm features + 5 missingness flags = 27 columns (paper panel)

V3 = V2 + 8 Phase-2 features (no deletions):

**Liquidity (+2)** — CRSP daily volume:
- `Turnover_zscore`
- `AmihudIlliquidity_zscore`

**Price patterns (+3)** — CRSP daily prices:
- `High52W_Proximity_zscore`
- `MaxDailyReturn_zscore`
- `ReturnSkewness_zscore`

**Growth (+2)** — IBES + Compustat quarterly:
- `ImpliedEPSGrowth_zscore`
- `QRevGrowthYoY_zscore`

**Coverage (+1)** — IBES:
- `AnalystCoverageChg_zscore`

(Same 5 missingness flags as V2.)

### V1 (deprecated) — yfinance + SEC XBRL

V1 used 4 firm features (`EarningsYield`, `AssetGrowth`, `Accruals`, `Momentum12_1`) + macro factors. Replaced entirely by V2 in 2026-04-18 due to split-contamination in EarningsYield and missing XBRL `dei` namespace. See `docs/archive/data_changelog.md` for migration table.

---



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

- `archive/` — run logs (cpcv_log, rung12_v2_rerun_log, audit-script logs), an early GAM diagnostic plot, and a few legacy result parquets still consumed by `statistical_tests.py`, `scripts/compute_turnover.py`, and `scripts/compute_ls_hit_rate.py`. Not paper-cited; kept for reproducibility evidence.
