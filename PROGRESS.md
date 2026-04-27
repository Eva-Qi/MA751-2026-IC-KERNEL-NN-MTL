# Project Progress — Newest First

> Living progress snapshot. **Last line first**: read top → bottom = present → past.
> Detailed narrative + per-section sources in `docs/PROJECT_STORY.md`.

---

## 🎯 NOW (2026-04-27)

**Thesis (paper-final)**: At 442 stocks × 119 months × 27 features, **adding model complexity does not improve cross-sectional return prediction in a statistically meaningful way**. CPCV shows top-7 models' [5%, 95%] Sharpe bands all overlap.

**Headline Pareto** (CPCV 15-path mean ± std):

| Model | CPCV Sharpe | P(Sharpe<0) |
|---|---|---|
| 3b XGB GKX-tuned | 1.086 ± 0.766 | 6.7% |
| **1c Fama-MacBeth** | **1.071 ± 0.494** ⭐ most-robust | 0% |
| 1d Barra (Grinold-Kahn) | 1.058 ± 0.544 | 0% |
| 1a OLS / 2a LASSO / 2b Ridge / 2d ElNet | 1.00–1.04 ± 0.6 | 0% |
| 1b IC-Ensemble | 0.840 ± 0.559 | 0% |

(See `RESULTS.md` for canonical numbers. See `output/cpcv_summary.csv` for raw data.)

---

## 🔁 Pivots & milestones (newest first)

### 2026-04-27 (today): Paper Rung 5 fill + repo move + slides outline
- **Rung 5 added to paper body**: Tab 1 extended with 5b/5c/5d MTL rows (V2 panel, footnoted); new §5.6 "Rung 5 Negative Findings" documents negative transfer, expert collapse, HMM under-identification; §7 Discussion paragraph links Rung 5 failures to sample-size constraint; Limitations note for CPCV exclusion reconciled
- **Figure callouts strengthened**: `fig_cumret`, `fig_cpcv_sharpe`, `fig_pareto_with_bounds` now have interpretive body text instead of navigational refs
- **Bibliography**: added `yu_pcgrad` (NeurIPS 2020) and `shazeer_moe` (ICLR 2017)
- **Repo move**: `paper/MA751-paper/` → `MA751-paper/` at repo root; `paper/` stays gitignored (private scratch); LaTeX build-artifact ignores added for `MA751-paper/*` to keep `pdflatex` runs out of git status
- **Slides outline**: 22-slide deck (`MA751-paper/slides_outline.md`) with action-title format + 3-5 sentence speaker notes per slide, ready for Google Slides import
- **Pushed to GitHub origin/main** for teammate access (commit 55529f5)

### 2026-04-27 (earlier): Repo cleanup + paper finalization
- Codebase 30+ → 16 active `.py` (audits/, src/ consolidated; 5,000+ LOC dead-code purged)
- Docs 10 scattered MDs → 1 master `PROJECT_STORY.md` + `RESULTS.md` + `FIGURES_TODO.md` + `archive/`
- Output: 130 → 96 active files; 2 outdated summaries archived (`gam_summary.csv` for default n=10; `rung2c_selected_ols_summary.csv` for look-ahead variants)
- Branches: `audit-fixes-2026-04-24` + 5 old branches deleted; only `main` remains
- 14 paper figures generated (Figs 1–8 paper-cited + Figs 9–15 polish)

### 2026-04-26: Final audit cycle
- **CPCV** (15-path × 8 models): top-7 [5%, 95%] Sharpe bands overlap; FM most robust
- **Rung 3a GAM `n_splines` sweep**: default n=10 over-fits → tuned n=4 (Sharpe 0.828) / n=5 (IC 0.0146); both materially > default 0.555
- **Rung 4/5 negatives**: K=5 seed bagging hurt (P7 metric mismatch); inner-CV HP search hurt (P8 over-tune); 4 independent attempts confirm MLP < linear

### 2026-04-24: Code-council audit + fixes
- **P0 #1** Barra duplicate definition — silently overrode True Grinold-Kahn. Fix: V3 1d Barra Sharpe 0.746 → **1.134** (new V3 #1)
- **P0 #2** `has_positive_earnings` post-filter bias — captured pre-filter sign
- **5 new models**: 1c Fama-MacBeth, 1d True Barra, 2d ElasticNet, 2e Adaptive LASSO, 3b XGBoost (default + GKX-tuned)
- **Methodology hardening**: TimeSeriesSplit replaces random K-fold; Ridge `scoring=Spearman`; per-fold seeds; Huber loss in MLP family

### 2026-04-20: Rung 2c look-ahead disclosure
- `2c_OLS_LASSO3/5` feature subsets chosen from full-sample LASSO frequency → forward leak
- Paper §4 demotes these results to "look-ahead caveat" footnote
- Producing script (`run_rung2c_selected_ols.py`) deleted from repo on 2026-04-26

### 2026-04-18: Data pipeline V2 (WRDS migration)
- V1 (yfinance + SEC XBRL) → V2 (WRDS: CRSP/Compustat/IBES/13F/short)
- Eliminates split-contamination in EarningsYield (AAPL 4.3× undercount, GOOG 20×)
- Phase-2 features: 14 → 22 firm-level + 5 missingness flags = **27 total**
- 16 parquet files, 19.7M rows, 251 MB

### 2026-04-10: Pipeline fixes (cross-sectional z-score, NaN semantics)
- C1: Z-score temporal leakage (pooled global → per-month CS)
- C2: EarningsYield split contamination (V1 → V2 fixes)
- C3: Type-A NaN handling (sector-undefined fundamentals kept NaN, not zero-filled)

### 2026-04-09: V1 baseline pipeline
- 4 features: EarningsYield, AssetGrowth, Accruals, Momentum 12-1
- Baseline LASSO + IC-Ensemble + OLS
- Now superseded by V2 — script preserved in `data/build_baseline_dataset.py` for legacy reproducibility

### 2026-03-31: Original project plan
- 5-rung complexity ladder (Rung 1 linear → Rung 5 MTL+MoE)
- 13 features, ~500 S&P 500 stocks, 4-week timeline
- Original Week 4 deliverable: PCA + permutation importance + effective-df plot (delivered 2026-04-27 as Figs 9, 10, 11)

---

## 🛠️ Methodology lessons logged to `code-council` skill

| # | Pattern | Source incident |
|---|---|---|
| P5 | HP default trap (untuned ≠ evaluated) | Rung 3a + 3b GAM/XGB defaults |
| P7 | Bagging metric-space match (rank vs Pearson) | Rung 4/5 K=5 seed bagging |
| P8 | HP search corrupts on low-SNR validation | Rung 4 inner-CV HP search |
| P9 | Dead-code accumulation in research codebases | this same project, 2nd-pass audit |
| P10 | Audit scope blindness (.py-only sweep, GitHub Languages) | 2 dead notebooks + 2 V1 dev tools missed |
| P11 | Producer-side deletion blindness (consumer outlives producer) | output/ cleanup catch |
| Part 14 | Project documentation coalescence (5-rule pattern) | docs/ 10-MDs reorg |

---

## Where to read more

- `RESULTS.md` (project root) — canonical numbers, file→paper mapping
- `docs/PROJECT_STORY.md` — full master narrative (~1,500 lines, newest first)
- `docs/FIGURES_TODO.md` — figure inventory (8 have / 3 planned / 4 suggested / 5 future)
- `docs/archive/` — 10 original MDs preserved
- `audits/README.md` — frozen audit-script inventory
- `output/README.md` — output file-naming + paper mapping

---

*Maintained as a one-page synopsis. For depth, see `docs/PROJECT_STORY.md`.*
