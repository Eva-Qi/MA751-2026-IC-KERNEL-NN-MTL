# MA751 Project — Audit Update (2026-04-24)

> Snapshot of what's changed since commit `0623ff7` (pre-audit baseline).
> Companion to commit `b30a002` on branch `audit-fixes-2026-04-24`.

---

## New Models Added

| Rung | Model | File | Method |
|---|---|---|---|
| **1c** | Fama-MacBeth | `run_rung12_v2.py` | Per-month cross-sectional OLS via `np.linalg.lstsq`, time-series β mean |
| **1d** | True Barra | `run_rung12_v2.py` | Grinold-Kahn (2000): per-month `f_m`, Ledoit-Wolf `Σ_f`, `w* = Σ_f⁻¹ μ_f` |
| **2d** | ElasticNet | `run_rung12_v2.py` | `ElasticNetCV`, refined `l1_ratio = [0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99]` |
| **2e** | Adaptive LASSO | `run_rung12_v2.py` | Zou (2006) oracle property, `1/|β_OLS|` adaptive weights |
| **3b** | XGBoost (MSE + Quantile) | `rung3b_gbm.py` (new) | Gradient boosted trees, two variants |
| **4** | MLP single-task (relabel) | `main.py` | Previously mislabeled as `5a` inside Rung 5 ablation |

---

## Audit Fixes Applied

### Data layer (`config.py`, `load_data.py`)
- Wired `MISSINGNESS_INDICATORS` into default feature set (`ALL_FEATURE_COLS_V2_WITH_MISS`)
- Added 2 new indicators: `has_analyst_consensus`, `has_positive_earnings`
- Beta/IVOL NaN handling: zero-fill → sector cross-sectional median
- V2 panel rebuilt to include new flag columns

### Model layer (`run_rung12_v2.py`, `scripts/rung12_v3.py`)
- `cv=5` random KFold → **month-aware TimeSeriesSplit** (gap = 1 month, not 1 row)
- Ridge α selection: R² → **Spearman IC** (align with evaluation metric)
- LASSO α grid: reverted to `logspace(-4, 1, 50)` after extended grid caused plateau
- ElasticNet `l1_ratio` grid refined (added 0.01 / 0.95 / 0.99 for Ridge/LASSO-end resolution)
- IC-Ensemble + Barra recent window: **by DATE** not `12*400-row` proxy
- FM uses SVD `lstsq` instead of `LinearRegression` for rank-deficiency safety
- LASSO/ElasticNet/AdaptiveLASSO marked `[unreliable_SNR]` in docstrings

### MLP/MTL layer (`main.py`, `regmtl.py`, `regmtl_enhanced.py`)
- MSE → Huber (`F.smooth_l1_loss`) with **β dynamic = 0.5 × std(y)** per task
  - Fixes previous β=0.01 which was ~1/8 std(y) → degenerated to MAE
- Per-fold `torch.manual_seed(fold_idx)` + `np.random.seed(fold_idx)` for reproducibility
- `--data` default → `master_panel_v2.parquet` (avoids V1 dei-namespace gaps)

### Other files (`statistical_tests.py`, `rung3_gam.py`, `rung5_planned.py`, `rung5_combined.py`)
- Rung 4 / `5a` relabeling propagated to statistical-test comparisons
- `--data` default updates

---

## Latest Results

### V2 Panel (19 features = 14 V2 + 5 miss flags), 58 test months

| Model | IC | IC-IR | Hit | Sharpe |
|---|---|---|---|---|
| 1a OLS | 0.0118 | 0.071 | 56.9% | 0.810 |
| 1b IC-Ensemble | 0.0159 | 0.076 | 55.2% | 0.771 |
| **1c FamaMacBeth** | **0.0244** | **0.139** | **62.1%** | **1.008** ← V2 Pareto-best |
| 1d Barra | 0.0070 | 0.036 | 53.4% | 0.497 |
| 2a LASSO | 0.0118 | 0.071 | 56.9% | 0.810 |
| 2b Ridge | 0.0126 | 0.071 | 55.2% | 0.800 |
| 2d ElasticNet | 0.0118 | 0.071 | 56.9% | 0.810 |
| 2e AdaptiveLASSO | 0.0118 | 0.071 | 56.9% | 0.810 |

**V2 observations**: LASSO/EN/AdaptiveLASSO identical to OLS → 100% fallback (ElasticNet diag: `alpha=10.0, n_nonzero=0.0`). Confirms "low SNR at IC≈0.01 → L1 collapse" finding.

### V3 Panel (27 features = 22 V3 + 5 miss flags), 58 test months

| Model | IC | IC-IR | Hit | Sharpe |
|---|---|---|---|---|
| 1a OLS | 0.0164 | 0.106 | 53.4% | 0.924 |
| 1b IC-Ensemble | 0.0208 | 0.094 | 50.0% | 0.844 |
| **1c FamaMacBeth** | **0.0259** | **0.165** | **63.8%** | 0.959 |
| 1d Barra | 0.0183 | 0.105 | 62.1% | 0.746 |
| **2a LASSO** | 0.0183 | 0.118 | 53.4% | **1.000** ← V3 Pareto-best |
| 2b Ridge | 0.0177 | 0.105 | 55.2% | 0.966 |
| **2d ElasticNet** | 0.0179 | 0.116 | 53.4% | 0.997 |
| 2e AdaptiveLASSO | 0.0164 | 0.106 | 53.4% | 0.924 |

**V3 observations**:
- LASSO/EN fallback no longer happens (EN diag: `alpha=8.29, n_nonzero=1.5`)
- 5 of 8 models achieve Sharpe ≥ 0.92
- FM IC-IR 0.165 = strongest signal consistency
- Barra V2→V3 biggest Δ (+0.249 Sharpe) — more features → stable `Σ_f`

### Rung 3b XGBoost (V3 only)

| Model | IC | Sharpe |
|---|---|---|
| 3b GBM (MSE) | 0.0018 | 0.335 |
| 3b GBM (Quantile) | -0.0029 | 0.064 |

**Confirms**: non-linear tree ensembles also fail on this panel. Supports "simple linear dominates" thesis.

---

## Pareto Frontier — Three Points

```
           | FM V2       | LASSO V3    | FM V3
-----------|-------------|-------------|-------------
Features   | 14 + 5 flag | 22 + 5 flag | 22 + 5 flag
Sharpe     | 1.008 ⭐    | 1.000 ⭐    | 0.959
IC         | 0.0244      | 0.0183      | 0.0259 ⭐
IC-IR      | 0.139       | 0.118       | 0.165 ⭐
Hit Rate   | 62.1%       | 53.4%       | 63.8% ⭐
Narrative  | parsimony   | regularized | consistency
```

**Interpretation**:
- **FM V2 wins Sharpe** — parsimony wins for per-month cross-sectional OLS averaging (fewer features → more stable per-month β)
- **LASSO V3 wins Sharpe (tied)** — explicit regularization lets extra V3 features contribute without overfitting
- **FM V3 wins IC / IC-IR / Hit Rate** — strongest month-to-month signal consistency

---

## Statistical Significance

Paired t-tests over 58 test months across all comparisons (flag vs no-flag, V2 vs V3, pairs of linear models): **none significant at p < 0.05**. Best p-value observed: 0.208 (Ridge V3 LS return vs baseline).

Consistent with Prof. Carvalho's prior expectation that distinguishable alpha is unlikely at this sample size. Under-powered to detect small effects.

---

## Audit Extension (2026-04-26)

Following the 2026-04-24 audit, ran 4 deeper studies + 1 final harness migration. Results:

### 1. Rung 3a GAM n_splines sweep + fine-tune (`docs/rung3a_findings.md`)
- 9 configs tested (n_splines ∈ {3, 4, 5, 6, 7, 8, 9, 10, 15, 20}). pygam default n=10 was 2× too complex
- **n=4: Sharpe 0.828 (Pareto-best Sharpe), n=5: IC 0.0146 (Pareto-best IC)**
- Forced retraction #2 of audit cycle: ~~"GAM additive non-linearity insufficient"~~ → "GAM with proper n_splines reaches 86% of Ridge floor"
- K-fold λ vs GCV baseline: negative result (K=5 inner CV did not improve over single-shot GCV; documented as P5/K-default trap in code-council Part 12)

### 2. Rung 4/5 seed bagging — NEGATIVE (`docs/rung45_seed_bagging_findings.md`)
- K=5 seed bagging on Rung 4 MLP gave Sharpe 0.394 vs single-seed mean 0.412 — **bagging hurt**
- Root cause: averaged raw predictions (Pearson space) but evaluated on Spearman rank IC. Mean-averaging compresses extremes
- Lesson logged: **code-council Part 12 P7** — bagging operator must match metric space (rank-aware vs value-aware)

### 3. Rung 4 inner-CV HP search — NEGATIVE (`docs/rung4_hp_search_findings.md`)
- TimeSeriesSplit(K=3) inner CV over 27 HP configs → Sharpe collapsed from baseline 0.347 to 0.111
- Root cause: at IC≈0.01 SNR, validation-noise SE ≈ true signal magnitude → HP search overfits to validation noise
- Lesson logged: **code-council Part 12 P8** — HP search degrades outer performance when signal ≤ 2× val-noise SE

### 4. CPCV (Combinatorial Purged K-fold) — Final analysis (`docs/cpcv_findings.md`)
Built `cpcv_harness.py`. Ran 15-path CPCV on top 8 models:

| Rank | Model | Sharpe (15-path) | std | [5%, 95%] | P(<0) |
|---|---|---|---|---|---|
| 1 | 3b XGB GKX-tuned | 1.086 | 0.766 | [0.05, 2.15] | 6.7% |
| 2 | **1c FamaMacBeth** | **1.071** | **0.494** ⭐ | [0.34, 1.68] ⭐ | **0%** |
| 3-6 | OLS / LASSO / Ridge / ElasticNet | 0.997-1.040 | 0.59-0.62 | overlapping | 0% |
| 7 | 1b IC-Ensemble | 0.840 | 0.559 | [0.10, 1.64] | 0% |
| 8 | 1d Barra | 0.671 | 0.425 | [0.12, 1.35] | 6.7% |

**Big finding**: All 8 models' 5-95% Sharpe bands **overlap**. Single-path walk-forward Pareto comparisons are within path noise. **FM is most robust** point estimate; XGB has highest mean but largest variance. **No model is statistically dominant** at our sample size.

**Thesis update**: "Adding complexity does not improve cross-sectional return prediction *in a statistically meaningful way at our sample size*."

---

## Outstanding Items

### Discussed but pending
- **[P0 confirmed]** `torch.manual_seed` in all Rung 5 → done
- **[P1 pending]** MoE gate entropy + load-balancing regularizer (Shazeer 2017)
- **[P1 pending]** Rung 5 `planned` auxiliary: `rank` → `fwd_sector_return` (genuinely independent task)
- **[P2 pending]** Re-select Enhanced MoE features on pre-2020 folds (fix look-ahead)
- **[P2 pending]** Pairwise ranking loss for MLP/MTL (address MSE/IC mismatch)

### Skip per friend's critique
- Drop LASSO/EN/AdaLASSO standalone → we chose `keep + [unreliable_SNR] flag + disclose`
- PCGrad gradient surgery — document Yu et al. 2020 finding, don't implement
- Differentiable Spearman (torchsort) — defer to future work
- Excess return — skip (IC invariant, Sharpe is constant offset)

### Open questions (pending verification / discussion)
- Do we want to run the **full Rung 5** ablation (MTL / Regime MoE / Enhanced MoE) with the new Huber β + seed fixes?
- Report writing — status?

---

## Files Changed (this session)

**Code (committed in `b30a002`)**:
- `config.py`, `load_data.py`, `main.py`, `regmtl.py`, `regmtl_enhanced.py`
- `run_rung12_v2.py`, `rung3_gam.py`, `rung5_combined.py`, `rung5_planned.py`
- `scripts/rung12_v3.py`, `statistical_tests.py`
- `rung3b_gbm.py` (new), `scripts/paired_ttest_miss_flags.py` (new)

**Output artifacts**: ~30 parquet + CSV files in `output/` (new models × V2/V3)

**NOT committed** (per user instruction): `docs/*`, `data/*`

---

*Auto-generated 2026-04-24. Companion commits: `b30a002`.*
