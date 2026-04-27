# Figures Inventory & TODO

> Living list of paper figures: what we have, what's planned, what we should consider adding.
> Last updated 2026-04-27.

---

## ✅ Have (8 figures, all in `paper/figures/`)

| # | Figure | Section | Type | Source data |
|---|---|---|---|---|
| 1 | `fig_data_flow.pdf` | §2 Data | TikZ schematic | hand-drawn placeholder |
| 2 | `fig_ladder.pdf` | §3 Methodology | matplotlib FancyBboxPatch + arrows | `make_ladder_figure.py` (no data) |
| 3 | `fig_gam_n_splines.pdf` | §5.1 Default trap | dual-axis line | `output/rung3a_n_splines_sweep.csv` + `output/rung3a_n_splines_finetune.csv` |
| 4 | `fig_xgb_sensitivity.pdf` | §5.1 | 3×3 heatmap + GKX callout | `output/rung3b_sensitivity_grid.csv`, `output/rung3b_gkx_tuned.csv` |
| 5 | `fig_mlp_seeds.pdf` | §5.2 | bar + reference lines | `output/rung5_seed_sensitivity.csv` |
| 6 | `fig_cpcv_sharpe.pdf` | §6 CPCV | boxplot + swarm | `output/cpcv_results.parquet` |
| 7 | `fig_pareto_with_bounds.pdf` | §6 | scatter + errorbars | `output/cpcv_summary.csv` |
| 8 | `fig_cumret.pdf` | §4 Results | time-series multi-line | `output/results_*_v3.parquet` (top 5 models) |

Generation scripts (gitignored, local-only): `make_figures.py`, `make_ladder_figure.py`.

---

## 🎯 Planned but not yet implemented (3 — from original Week 4 plan)

These were on the original `docs/final-plan.pdf` deliverable list but never executed.

### 9. `fig_pca.pdf` — PCA scree + scatter

**Section**: §2.4 (proposed) — feature space dimensionality.
**Type**: Two-panel: top = scree plot (% variance per PC, cumulative), bottom = PC1 vs PC2 scatter.
**Color recommendation**: by **GICS sector** (11 colors) — interpretable cluster structure ("which features cluster cross-sector vs. within-sector"). Backup: by recent forward-return quartile.
**Source data**: `data/master_panel_v2.parquet`, take 27 V3 features, run sklearn `PCA`.
**Effort**: ~30 min, ~50 LOC.

### 10. `fig_perm_importance.pdf` — Permutation importance per top-3 model

**Section**: §5.3 (proposed) — feature contribution analysis.
**Type**: 3-panel bar chart (one per model: 1c FM, 1d Barra-GK, 3b XGB-GKX). x = feature, y = ΔIC when feature is shuffled in test set. Sorted descending.
**Source**: re-run each model with each feature shuffled (single fold or all folds).
**Effort**: ~45 min, ~80 LOC.

### 11. `fig_effective_df.pdf` — Effective degrees of freedom vs Sharpe

**Section**: §3.1 or §7 Discussion (proposed) — complexity vs. performance scatter.
**Type**: scatter, x = effective DoF (model parameter count after regularization), y = LS Sharpe. Points labeled by model. Reference line at Ridge floor.
**For each model**, effective DoF approximation:
- Linear (1a/1c/1d): p = 27 (feature count)
- LASSO/Ridge: p × (1 − shrinkage estimated from coefficient L2 norm)
- GAM: n_splines × p × (1 − GCV-based shrinkage)
- MLP: 3,073 × (1 − dropout)
**Source**: post-hoc analysis using fitted-model state from each rung.
**Effort**: ~1 hr (DoF approximation per rung is non-trivial), ~100 LOC.

---

## 💡 Suggested (4 — consider adding for paper polish)

### 12. `fig_lasso_freq.pdf` — LASSO selection frequency

**Why**: Shows which features survive walk-forward LASSO selection. Proxy for permutation importance, simpler.
**Type**: Horizontal bar chart, x = % of folds where feature has non-zero coefficient, y = feature.
**Section**: §5.1 (sidebar) or appendix.
**Source**: `output/lasso_v3_selection_summary.csv` already exists.
**Effort**: ~10 LOC, 5 min.

### 13. `fig_correlation_matrix.pdf` — V3 feature correlation heatmap

**Why**: Justifies Adaptive LASSO + Elastic Net (multicollinearity argument). Shows IBES-cluster (SUE/Revision/Dispersion correlated) and risk-cluster (Beta/IVOL correlated).
**Type**: 27×27 correlation heatmap, hierarchically clustered.
**Section**: §2.4 (data section) or appendix.
**Source**: `data/master_panel_v2.parquet`, compute `df[V3_features].corr()`.
**Effort**: ~15 LOC, 10 min.

### 14. `fig_regime_timeline.pdf` — HMM regime states over time

**Why**: Visualizes Rung 5 MoE gate input. Shows COVID/inflation/mag-7 regime shifts. Useful color for the §5.5 Rung 5 audit discussion.
**Type**: Stacked area chart, x = test month, y = posterior probability of each of 3 HMM states, colored.
**Section**: §5.5 or §6 (CPCV interpretation re: regime-conditional XGB tail risk).
**Source**: HMM posteriors per fold (would need to extract from `regmtl.py` runs or recompute).
**Effort**: ~30 LOC, 20 min if HMM data already saved; otherwise 1 hr.

### 15. `fig_v2_vs_v3.pdf` — V2 vs V3 ΔSharpe per model

**Why**: Shows which models benefited from Phase-2 features (V3 expansion) vs. plain V2.
**Type**: Grouped bar chart, x = model, y = Sharpe, two bars per model (V2 / V3), with delta annotated.
**Section**: §2 Data (motivates V3 expansion) or appendix.
**Source**: `output/rung12_v2_summary.csv` + `output/rung12_v3_summary.csv`.
**Effort**: ~15 LOC, 10 min.

---

## 🔬 Possible future work (post-deadline)

- `fig_per_seed_traces.pdf` — MLP training trajectories per seed (tie to §5.2)
- `fig_cpcv_paths_grid.pdf` — small-multiples of all 15 CPCV cumulative-return paths per model
- `fig_decile_spread.pdf` — decile spread per model (vs. quintile in `fig_cumret.pdf`)
- `fig_turnover_vs_sharpe.pdf` — turnover-Sharpe Pareto (cost-adjusted)
- `fig_rolling_ic.pdf` — per-model 12-month rolling IC over time (regime-conditional performance)

---

## Reproducibility

Add new figures to `make_figures.py` (gitignored). Run:
```bash
python make_figures.py        # regenerates Figs 3-8 + new ones
python make_ladder_figure.py  # regenerates Fig 2 only
```

Both scripts depend on `output/*.csv` + `output/*.parquet` files already present in the repo (since 2026-04-26 audit cycle).

---

## Total inventory

| Category | Count | Files |
|---|---|---|
| ✅ Have | 8 | Figs 1--8 |
| 🎯 Planned (Week-4 deliverable) | 3 | Figs 9--11 |
| 💡 Suggested polish | 4 | Figs 12--15 |
| 🔬 Future | 5 | unnumbered |
| **Total target** | **20** | |
