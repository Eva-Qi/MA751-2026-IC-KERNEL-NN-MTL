# MA751 Project — Master Document

> **Living history** of the Cross-Sectional Stock Return Prediction project (S&P~500, 2016--2024).
>
> Read top-down for current state. Scroll down for older entries — each dated section is a snapshot of what was true at that point. **Last line first**: most recent at top.
>
> *Maintained as a merge of 10 prior Markdown docs (now archived under `docs/archive/`). Source for each section is annotated.*

---

## 🎯 TL;DR — Where we are now (2026-04-27)

**Thesis (final)**: At the scale of 442 S&P~500 stocks × 119 months × 27 features, **adding model complexity does not improve cross-sectional return prediction in a statistically meaningful way**. Combinatorial Purged K-fold (CPCV) shows that all top-7 models' 5--95% Sharpe bands overlap. **Fama-MacBeth and Grinold-Kahn Barra are the most robust point estimates** (CPCV mean Sharpe 1.071 / 1.058, both with $P(\text{Sharpe}{<}0)=0\%$).

**5-rung complexity ladder** (12 model variants):

| Rung | Variants | Top result (CPCV mean Sharpe) |
|---|---|---|
| 1 — Linear | OLS, IC-Ensemble, Fama-MacBeth, Barra (Grinold-Kahn) | **1.071** (FM) |
| 2 — Regularized | LASSO, Ridge, Elastic Net, Adaptive LASSO | 1.037 (LASSO) |
| 3 — Smooth non-linear | GAM ($n_{\text{splines}}{=}4$), XGBoost (GKX-tuned) | **1.086** (XGB) |
| 4 — Single-task MLP | hidden=64, ReLU, dropout 0.10 | 0.412 (5-seed mean, all $<$ Ridge) |
| 5 — Multi-task / MoE | 5b/5c/5d MTL, Regime-MoE, Enhanced MoE | dominated by linear |

**Active code structure (post-cleanup)**:
- 16 active `.py` in project root (driver scripts) + `scripts/` (analysis utilities) + `src/` (shared modules)
- 12 frozen audit / ablation snapshots in `audits/` (linguist-generated, GitHub stats exclude)
- All paper-cited numbers traceable from `RESULTS.md` to `output/*.csv` to specific scripts

**Single source of truth for paper numbers**: see `RESULTS.md` at project root.

---

# 📅 Timeline — Newest first

---

## 2026-04-27: Repo cleanup + paper finalization

**Deleted dead code (today)**:
- `analysis/linear.ipynb` (493 KB), `analysis/lasso.ipynb` (146 KB) — early prototype notebooks superseded by `run_rung12_v2.py`
- `data/diagnose_tickers.py`, `data/validate_data_legitimacy.py` — V1-era debug tools
- `walk_forward.py` — 113 LOC of "shared framework" with zero importers (every rung implemented walk-forward inline)

**Total today**: −2,277 LOC + 640 KB JSON. Project root .py count 30+ → 18 → 16. GitHub Languages now $\sim$100% Python (was 61% Jupyter / 39% Python).

**Branch consolidation**: deleted `audit-fixes-2026-04-24` + 5 older branches (`data-pipeline`, `mtl`, `post-lasso-ols`, `reg`, `rung-1`). Repo is now `main` only. PR #5 closed (squash-merged locally as commit `7fe8047`).

**Lessons logged to code-council Part 12 P9 + P10**: dead-code accumulation patterns + audit-scope blindness (extension/directory-blacklist + GitHub-Languages cross-check).

---

## 2026-04-26: Final audit cycle — CPCV + sensitivity studies

### 2026-04-26 (cpcv_findings.md): CPCV path-uncertainty analysis

> Source: `docs/archive/cpcv_findings.md` · Last updated 2026-04-26 · From `run_cpcv_all_models.py` + `cpcv_harness.py`.

**TL;DR**: 15-path Combinatorial Purged K-fold (López de Prado 2018, ch.~12) on 8 top models. **All top-7 models' 5--95% Sharpe bands overlap** — no model statistically dominates at our sample size.

**CPCV setup**:
- Panel: 119 months × 442 stocks × 27 features
- $N_{\text{blocks}}{=}6$ contiguous time blocks (sizes [20, 20, 20, 20, 20, 19] months)
- $k_{\text{test}}{=}2$ blocks per path; embargo = 1 month adjacent
- $\binom{6}{2}{=}15$ paths per model

**Cross-model ranking** (`output/cpcv_summary.csv`):

| Rank | Model | Sharpe mean | Std | [5\%, 95\%] | P($<$0) | WF Sharpe | IC mean |
|---|---|---|---|---|---|---|---|
| 1 | 3b XGB GKX-tuned | **1.086** | 0.766 | [0.05, 2.15] | 6.7% | n/a | +0.0280 |
| 2 | 1c FamaMacBeth | 1.071 | **0.494** ⭐ | [0.34, 1.68] ⭐ | 0% | 0.959 | +0.0347 |
| 3 | 1d Barra (G-K, after fix) | 1.058 | 0.544 | **[0.43, 1.86]** | 0% | **1.134** | +0.0295 |
| 4 | 1a OLS | 1.040 | 0.618 | [0.28, 1.94] | 0% | 0.924 | +0.0315 |
| 5 | 2a LASSO | 1.037 | 0.618 | [0.28, 1.94] | 0% | 1.000 | +0.0308 |
| 6 | 2b Ridge | 1.008 | 0.622 | [0.20, 1.91] | 0% | 0.966 | +0.0310 |
| 7 | 2d ElasticNet | 0.997 | 0.592 | [0.28, 1.94] | 0% | 0.997 | +0.0299 |
| 8 | 1b IC-Ensemble | 0.840 | 0.559 | [0.10, 1.64] | 0% | 0.844 | +0.0263 |

**Key observations**:
- **A**. Single-path WF and CPCV-mean broadly agree, but FM was *under-rated* by single-path (WF 0.959 → CPCV 1.071) and Barra was *over-rated* (WF 1.134 → CPCV 1.058 due to 1 unfortunate test block partition).
- **B**. **No model dominates under uncertainty**. Top-6 models cluster within mean Sharpe [0.99, 1.09]; 5--95% bands all overlap heavily. Single-path "+0.05 Sharpe" deltas are path noise.
- **C**. **Risk-adjusted**: FM has the lowest std (0.494) and narrowest 5--95% band — the **most deployable** result. XGB has the highest mean but widest tail.
- **D**. **6 of 8 models have P(Sharpe$<$0) = 0%** — predictive signal is real (positive in every path); only XGB and Barra each have one negative-Sharpe path out of 15.

**Why CPCV matters**: walk-forward gives one historical realization (COVID + 2022 inflation + 2023 mag-7). CPCV picks alternative train/test partitions, isolating model robustness from regime luck. The Pareto comparisons we used to make (FM 0.959 vs LASSO 1.000) **all fit inside one CPCV distribution** — they're realizations of the same model class on this panel, not different models.

**Practical recommendation**: deploy by robustness (lowest std, lowest P($<$0)) → 1c Fama-MacBeth. For peak performance hunt: 3b XGBoost GKX-tuned, but accept higher variance.

**Skipped from CPCV**: Rung 3a GAM (too slow), Rung 4 MLP (already dominated under WF), Rung 5 MTL/MoE (same).

---

### 2026-04-26 (rung3a_findings.md): Rung 3a GAM — `n_splines` sensitivity sweep

> Source: `docs/archive/rung3a_findings.md` · Last updated 2026-04-26 · From `audits/rung3a_gam_sweep.py` (consolidates earlier `rung3a_gam_audit.py` + `rung3a_gam_finetune.py`).

**TL;DR**: pygam's default `n_splines=10` was over-fitting our IC≈0.01 SNR. Combined sweep over `n_splines ∈ {3, 4, 5, 6, 7, 8, 9, 10, 15, 20}` shows IC peaks at $n=5$ (+0.0146), Sharpe peaks at $n=4$ (0.828). Default (n=10) gave only 0.555. **Hyperparameter Default Trap (P5) confirmed.**

**Combined sweep results** (from `output/rung3a_n_splines_sweep.csv` + `output/rung3a_n_splines_finetune.csv`):

| n_splines | IC mean | IC-IR | Hit % | LS Sharpe |
|---|---|---|---|---|
| 3 | NaN (pygam: needs n>spline_order=3) | — | — | — |
| **4** | +0.0141 | 0.111 | 51.7 | **0.828** ⭐ Sharpe |
| **5** | **+0.0146** ⭐ IC | **0.126** ⭐ IC-IR | 50.0 | 0.792 |
| 6 | +0.0115 | 0.101 | 55.2 | 0.654 |
| 7 | +0.0095 | 0.087 | 55.2 | 0.561 |
| 8 | +0.0057 | 0.053 | 48.3 | 0.464 |
| 9 | +0.0047 | 0.046 | 46.6 | 0.468 |
| 10 (default) | +0.0068 | 0.066 | 53.4 | 0.555 |
| 15 | +0.0062 | 0.065 | 53.4 | 0.505 |
| 20 | +0.0071 | 0.089 | 55.2 | 0.406 |

**Pareto choice**: n=4 wins LS Sharpe by +0.036; n=5 wins IC + IC-IR by negligible margins (within fold noise). **Recommendation**: use $n_{\text{splines}}=4$ for deployment, $n_{\text{splines}}=5$ for paper-style ranking metrics. Both are radically better than pygam's default.

**Pattern interpretation**: $n_{\text{splines}} = N + \text{spline\_order}$ controls internal knot count (≈ N − 3 for cubic basis). Knots = piecewise complexity points. In IC≈0.01 SNR:
- $n=4 →$ 1 internal knot: monotone with one curvature point (broad direction-of-effect financial factors typically have)
- $n=5 →$ 2 internal knots: allows mild S-shape — slight rank-correlation benefit
- $n=6+ →$ 3+ knots: model fits noise; each extra knot reduces out-of-sample IC and Sharpe

Per-feature relationships in cross-sectional equity (Beta → return, Momentum → return, GP/A → return) are typically near-monotone with mild concavity. 1--2 internal knots is the right complexity. pygam's default of 10 was designed for general regression problems with much higher SNR.

**K-fold λ vs GCV (negative result)**: replacing pygam's single-shot GCV with TimeSeriesSplit($K{=}5$, gap=1) inner CV did NOT improve over GCV. ΔIC = −0.002 (within noise), ΔSharpe = −0.029. λ-selection diagnostic: median 0.25, range 0.016 → 63 (4 orders of magnitude across folds — high variance, suggests $K=5$ inner CV may have been too aggressive for our 60-month minimum train). Per López de Prado 2018, prefer CPCV over standard K-fold for financial time series.

---

### 2026-04-26 (rung45_seed_bagging_findings.md): Rung 4/5 Seed Bagging — NEGATIVE result

> Source: `docs/archive/rung45_seed_bagging_findings.md` · Last updated 2026-04-26 · From `audits/rung45_seed_bagging.py`.

**TL;DR**: K=5 seed bagging gave bagged Sharpe **0.394**, *lower* than 5-seed single-seed mean Sharpe of 0.412. Bagging hurt.

**Diagnosis**: bagging variance-reduction theorem $\mathrm{Var}(\bar{y}_K) \approx \sigma^2(1+(K-1)\rho)/K$ holds in **Pearson/MSE space**. Our evaluation metric is **Spearman rank** correlation. Mean-averaging predictions creates a rank-space compromise that pulls extreme rankings toward consensus middle. LS quintile Sharpe depends on extreme rankings; consensus-middle rankings produce weaker tails — the opposite of what bagging promises.

**Rank-level fix (not implemented for deadline)**:
```python
# WRONG (what we did): average raw predictions
y_bagged = np.mean([pred_k for k in seeds], axis=0)
ic = spearmanr(y_te, y_bagged)

# RIGHT for rank metrics: average ranks
ranks = np.array([rankdata(pred_k) for k in seeds])
y_bagged_rank = np.mean(ranks, axis=0)

# OR: median-prediction (rank-aware via robustness)
y_bagged_med = np.median([pred_k for k in seeds], axis=0)
```

**Lesson** (logged as code-council Part 12 P7): match the ensemble operator to the evaluation metric's space. For rank metrics: rank-level averaging or median. For Pearson metrics: mean-prediction.

**Decision**: do NOT re-run with rank-level bagging. Even with rank bagging, expected Sharpe lift is modest (probably 0.05--0.15 at best); even an optimistic bagged Sharpe of $\sim$0.5 is still well below Ridge 0.925. Bagging cannot rescue MLP from being dominated by linear baselines on this panel.

---

### 2026-04-26 (rung4_hp_search_findings.md): Rung 4 HP Search — NEGATIVE result

> Source: `docs/archive/rung4_hp_search_findings.md` · Last updated 2026-04-26 · From `audits/rung4_hp_search.py`.

**TL;DR**: Inner-CV hyperparameter search over 27 configs (hidden $\in\{16,32,64\}$ × lr $\in\{10^{-4},10^{-3},10^{-2}\}$ × dropout $\in\{0,0.1,0.2\}$) **collapsed Rung 4 Sharpe from 0.347 to 0.111**. HP search hurt by 0.236.

**Result table**:

| Configuration | IC mean | IC-IR | Hit % | LS Sharpe |
|---|---|---|---|---|
| Hardcoded (seed=42, h=64, lr=1e-3, drop=0.10) | +0.0007 | 0.007 | 51.7 | **0.347** |
| **Inner-CV HP-tuned** | **+0.0016** | **0.012** | 48.2 | **0.111** ❌ |
| Single-seed mean (5-seed audit) | +0.0109 | — | — | 0.412 |
| Ridge α=1 (linear floor) | +0.0164 | — | 53.4 | 0.925 |

**Diagnosis**: at IC≈0.01 SNR with $\sim$10,000 inner-validation observations, per-config Spearman IC standard error under $H_0$ is $1/\sqrt{10^4}\approx 0.011$ — **comparable to true signal magnitude**. Grid search picks "best" HP based on validation noise, not true signal. Outer test sees pure overfit-to-validation.

**Diagnostic signature** (the smoking gun for noise-driven HP search): chosen HP distribution across folds was **bimodal**:
- `hidden_dim`: 16 (31%) / 32 (7%) / **64 (62%)** — spread between two modes
- `lr`: 1e-4 (3%) / **1e-3 (47%) / 1e-2 (50%)** — bimodal
- `dropout`: 0 (53%) / 0.1 (29%) / 0.2 (17%) — usually 0

A signal-driven HP search converges to a single modal choice; ours did not. **Bimodal HP selection across folds is the smoking gun for over-tuning on validation noise.**

**Lesson** (logged as code-council Part 12 P8): before HP grid search on noisy panels, estimate validation-noise SE via $1/\sqrt{n_{\text{val}}}$. If signal magnitude ≤ 2× SE, grid search is contraindicated; prefer literature defaults or a small ($\le$5-config) screen.

**Decision**: do NOT re-run with smaller search space. The Rung 5 audit conclusion ("MLP < linear, complexity dominated") is now strongly confirmed by 4 independent attempts to improve MLP — single-seed sweep, architecture sweep, K=5 bagging, inner-CV HP search — all failed to beat linear. Remaining MLP improvement work (e.g., rank-level bagging, narrower HP screen) would not change the conclusion at the noise floor.

---

## 2026-04-24: Rung 5 MTL + MoE Deep Audit

> Source: `docs/archive/audit.md` · Last updated 2026-04-24. 简体中文 + English jargon. Companion plan was `~/.claude-eva/plans/i-want-you-to-imperative-badger.md`.

**Sections**: Kendall MTL loss → Negative transfer → Regime MoE architecture → Why MoE didn't work (Shazeer 2017) → HMM under-identification → Hyperparameter inventory → Fix priority.

### 1. Kendall Uncertainty-Weighted MTL Loss

Defined in `main.py:154-246` as `UncertaintyMTLLoss`. For each active task $k$:

$$\mathcal{L}_k = \tfrac{1}{2} e^{-s_k} \cdot \frac{\text{MSE}_k}{\text{Var}(y_k^{\text{train}})} + \tfrac{1}{2} s_k$$

- $s_k = \log\sigma_k^2$ is a learnable `nn.Parameter` (init 0)
- $e^{-s_k}$ = precision (inverse variance) = task weight
- $\text{Var}(y_k)$ computed once per fold from training subset, constant during training
- $0.5 s_k$ regularizer prevents $s_k\to-\infty$
- Multi-task: **sum** (not mean) over tasks

**Reference**: Kendall, Gal, Cipolla (CVPR 2018), "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics."

**Critical degeneracy at `n_tasks=1`** (Rung 4, formerly labeled `5a`): with single task, the loss reduces to $\mathcal{L}^* = 0.5 + 0.5\log(L_{\text{ret}})$ — a monotone transform of MSE. **Shared-layer gradient direction is identical to pure MSE.** Uncertainty mechanism is designed for multi-task balance; single-task it adds a dead `s_ret` parameter and nothing else. This should be acknowledged honestly in the report rather than implying single-task uncertainty has any meaning.

**Variance normalization regime-sensitivity**: `var_ret = float(y_ret_tr.var())` per fold (`main.py:203`). High-vol fold (e.g., 2020 COVID training) → large `var_ret` → loss shrunk → small gradient → slow updates. Low-vol fold → opposite. This is the *opposite* of what volatility targeting suggests. Mitigation candidates: rolling/EWMA variance, or long-term average variance.

### 2. Negative Transfer — Why Uncertainty Weighting Cannot Fix It

**Mechanism**: uncertainty weighting only adjusts loss *magnitude*, not gradient *direction*. For shared encoder parameters $\theta$:
- $g_{\text{ret}} = \nabla_\theta\mathcal{L}_{\text{ret}}$
- $g_{\text{3m}} = \nabla_\theta\mathcal{L}_{\text{3m}}$
- Combined: $\nabla_\theta\mathcal{L}_{\text{total}} = w_{\text{ret}}g_{\text{ret}} + w_{\text{3m}}g_{\text{3m}}$ (where $w$ is uncertainty weight)

If $g_{\text{ret}}$ and $g_{\text{3m}}$ point in opposing directions (negative cosine similarity), combining them is partial cancellation. No choice of $w$ flips the conflict — only attenuates one side. Shared representation learns the **compromise space**, which is mediocre for both.

**Reference**: Yu, Kumar, Gupta, Levine, Hausman, Finn (NeurIPS 2020), "Gradient Surgery for Multi-Task Learning" introduces PCGrad (project each task gradient onto the normal plane of conflicting tasks).

**5b empirical confirmation**:

| Variant | Active tasks | IC mean | IC t-stat |
|---|---|---|---|
| Rung 4 (single-task ret) | {ret} | +0.0048 | 0.254 |
| **5b** | {ret, ret3m} | **−0.0036** | −0.211 |
| 5c | {ret, vol} | +0.0061 | 0.324 |
| 5d | {ret, ret3m, vol} | +0.0173 | 0.892 |

**5b is worse than 4 with sign-flipped IC** — textbook negative transfer. 3-month and 1-month forward returns conflict during 2022 regime shift (long-term continuation vs. short-term reversal).

5d > 5b/5c suggests adding the third task (vol) **dilutes** the ret-vs-ret3m conflict because vol prediction forces orthogonal risk representation. This is a side effect, not a designed feature.

**Decision**: skip PCGrad implementation (40+ LOC, second-order gradients, deadline risk). Discuss 5b in report as the negative-transfer empirical example.

### 3. Regime-Gated MoE Architecture

**Plain MoE (`regmtl.py`)**:
```
Encoder: Linear(14→64) → ReLU → Dropout → Linear(64→32) → ReLU → Dropout
Gate:    Linear(3→16) → ReLU → Linear(16→K)   [K=3 experts]
Heads:   K Linear(32→1) per active task (nn.ModuleList)
```

Gate input: 3-dim HMM regime posteriors (`regime_p0/p1/p2`), recomputed per fold. Mixing: $\hat{y}^{(k)}_{\text{mixed}} = \sum_j g_j\hat{y}_{\text{expert}_j}^{(k)}$, $g = \text{softmax}(\text{gate}(r))$. **Softmax temperature hardcoded T=1.0** — no annealing.

Loss applied only on mixed output (`UncertaintyMTLLoss(preds, yr, y3m, yv)` on `y_mixed`). **No per-expert loss, no gate entropy regularizer, no load-balancing penalty.**

**Enhanced MoE (`regmtl_enhanced.py`) differences**:

| Component | plain MoE | Enhanced MoE |
|---|---|---|
| Stock features | 14 (V2) | 11 (7 LASSO + 4 interactions) |
| Encoder hidden1 | 64 | 32 |
| Encoder hidden2 | 32 | 16 |
| Gate input dim | 3 (regime only) | 6 (3 regime + 3 macro) |
| Gate hidden | 16 | 12 |
| Macro scaling | N/A | StandardScaler per-fold |

**Look-ahead contamination**: `ENHANCED_MOE_FEATURE_COLS` (7 features) chosen from LASSO frequency on **all 58 folds**, including test window — same look-ahead bug as `2c_OLS_LASSO3/5`. Fix planned (Tier C5): pre-2020 LASSO selection.

### 4. Why MoE Didn't Work — Shazeer 2017 Warning

**Reference**: Shazeer et al. (ICLR 2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."

**Failure modes for soft-gate MoE without explicit constraints**:
1. **Expert collapse** — all experts converge to similar function (homogeneous), $K{=}3$ degenerates to $K{=}1$
2. **Gate collapse** — gate always picks same expert, others idle
3. **Load imbalance** — one expert handles most samples, training unstable

**Shazeer's defenses**:
- *Importance regularizer*: encourage gate entropy to prevent always-pick-same-expert
- *Load regularizer*: penalize $\text{Var}(\text{gate.mean}(0))$ across batches

**MA751 empirical** (`results_5d_enhanced_moe.parquet`):

| Metric | Value |
|---|---|
| Expert 0 mean gate weight | 0.33 |
| Expert 1 mean gate weight | 0.33 |
| Expert 2 mean gate weight | 0.34 |
| Gate entropy std | 0.08-0.12 |

**Nearly uniform distribution.** Experts did not specialize, gate did not learn regime → expert mapping. **K=3 MoE is empirically equivalent to K=1 MTL** — extra experts are dead parameters.

**Fix candidate** (Tier C1, ~10 LOC):
```python
gate_w = preds["gate_weights"]  # [B, K]
entropy_per_sample = -(gate_w * torch.log(gate_w + 1e-8)).sum(dim=1)
load_var = torch.var(gate_w.mean(dim=0))
loss = mtl_loss - lambda_ent * entropy_per_sample.mean() + lambda_lb * load_var
# lambda_ent=0.01, lambda_lb=0.01 initial values
```

### 5. HMM Under-Identification

**HMM setup** (`regime.py`): K=3 states, 5-dim observation (`mkt_ret_1m, mkt_rv_1m, VIXCLS, T10Y2Y, BAMLH0A0HYM2`), `covariance_type='full'`.

**Free parameters** (per Hamilton 1989):
- Initial state distribution: K−1 = 2
- Transition matrix: K(K−1) = 6
- Means: K × 5 = 15
- Covariances: K × 5(5+1)/2 = K × 15 = 45
- **Total**: 2 + 6 + 15 + 45 = **68 free parameters**

**Early fold under-identification**: minimum 60 months training → 60 obs / 68 params → underidentified. Mid folds (80--90 obs) marginal; latest folds ($\sim$118 obs) only 1.7× parameter count, still tight.

**Consequence**: early-fold regime posteriors are noisy → gate input unstable → MoE learns inconsistent gate-mapping between early and late folds (structural break).

**Fix candidates**:
- Short-term: `covariance_type='diag'` (cov params 45 → 15, total params 68 → 38)
- Medium-term: K=2 (bull/bear binary)
- Long-term: HMM input → 3-dim (VIX + T10Y2Y + market return)

### 6. Hyperparameter Inventory (all Rung 5 variants)

**Architecture**:

| Param | MTL (5b/c/d) | plain MoE | Enhanced MoE |
|---|---|---|---|
| n_factors | 14 (V2+miss=19) | 14 | 11 |
| hidden1 | 64 | 64 | **32** |
| hidden2 | 32 | 32 | **16** |
| dropout | 0.10 | 0.10 | 0.10 |
| K (experts) | N/A | **3** | **3** |
| Gate hidden | N/A | 16 | 12 |
| Gate input | N/A | 3 (regime) | 6 (3 regime + 3 macro) |

**Training**:

| Param | Value | File:line |
|---|---|---|
| Optimizer | Adam | `main.py:294` |
| lr | 1e-3 | `main.py:265` |
| weight_decay | 1e-5 | `main.py:266` |
| batch_size | 512 | `main.py:262` |
| Scheduler | ReduceLROnPlateau, patience=8, factor=0.5 | `main.py:300` |
| max_epochs | 150 | `main.py:262` |
| early-stop patience | 20 | `main.py:267` |
| val_frac | 0.10 (chronological tail) | `main.py:268` |
| Softmax temp (MoE) | 1.0 hardcoded | `regmtl.py:126` |
| Gate entropy reg | **none** | — |
| Load-balance reg | **none** | — |
| log_sigma init | 0.0 per task | `main.py:183-187` |

**Reproducibility audit** (pre-fix):

| File | torch.manual_seed | np.random.seed |
|---|---|---|
| `main.py` | none | none |
| `regmtl.py` | none | none |
| `regmtl_enhanced.py` | none | none |
| `audits/rung5_planned.py` | none | none |
| `audits/rung5_combined.py` | none | none |

**Fix applied (B3)**: per-fold `torch.manual_seed(fold_idx)` + `np.random.seed(fold_idx)` in all Rung 5 code.

### 7. Fix Priority Ranking

| Priority | Item | Tier | Status |
|---|---|---|---|
| **High** | `torch.manual_seed(fold_idx)` all Rung 5 | B3 | ✅ Done (Day 2) |
| **High** | MSE → Huber (`smooth_l1_loss`) in MTL | B2 | ✅ Done |
| **High** | Target winsorize 1%/99% | B4 | ✅ Done |
| Medium | MoE gate entropy + load balance reg | C1 | not implemented |
| Medium | Rung 5 planned: rank → `fwd_sector_return` | C2 | not implemented |
| Medium | Pairwise ranking loss in MLP/MTL | C3 | not implemented |
| Medium | Clean feature selection (pre-2020 LASSO) | C5 | not implemented |
| Low | PCGrad gradient surgery | C4 | SKIP / document |
| Low | ListNet / ApproxNDCG | D1 | SKIP |
| Low | torchsort Spearman | D2 | SKIP |
| Low | Gate temperature annealing | D3 | gate on C1 |
| Low | HMM `covariance_type='diag'` + K=2 | — | not implemented |

**References**: Kendall, Gal, Cipolla 2018 (CVPR); Yu, Kumar, Gupta, Levine, Hausman, Finn 2020 (NeurIPS, PCGrad); Shazeer et al. 2017 (ICLR, MoE); Caruana 1997 (Multitask Learning); Ruder 2017 (MTL survey, arXiv:1706.05098); Jacobs, Jordan, Nowlan, Hinton 1991 (Adaptive Mixtures); Gorishniy et al. 2021 (Tabular DL); Diebold & Mariano 1995 (DM test); Fama & MacBeth 1973 (JPE).

---

## 2026-04-20: Rung 2c Look-Ahead Disclosure

> Source: `docs/archive/rung2c_notes.md` · Last touched 2026-04-20. Branch: `post-lasso-ols` (now deleted). Script: `audits/run_rung2c_selected_ols.py` (later deleted as confirmed look-ahead contaminated).

**Motivation**: run OLS on feature subsets chosen by prior selection, to test whether the full 14-feature OLS baseline is diluted by noise factors.

**Two selection sources**:
- *LASSO frequency* (from `output/lasso_coefs_by_fold.csv`): features kept in ≥50% of folds — IVOL (67%), GrossProfitability (62%), Beta (53%), plus ShortInterestRatio and Momentum12_1 in top-5
- *Fama–MacBeth t-stats*: NetDebt/EBITDA (t=−2.37), GrossProfitability (1.51), AnalystRevision (1.7), RevisionBreadth (1.31)

**Results** (58 test months):

| Model | Features | IC | IC-IR | Hit % | LS Sharpe | LS Hit |
|---|---|---|---|---|---|---|
| 2c_OLS_LASSO3 | IVOL, GP, Beta | +0.0168 | 0.078 | 55.2 | 0.809 | 58.6 |
| **2c_OLS_LASSO5** | + ShortInt, Mom12_1 | +0.0152 | 0.073 | 55.2 | **0.829** | **63.8** |
| 2c_OLS_FMB4 | NetDebt, GP, AnRev, RevBreadth | +0.004 | 0.024 | 44.8 | 0.146 | 50.0 |

**Look-ahead disclosure**: LASSO3 and LASSO5 feature sets were chosen by full-sample LASSO frequency including the test window — this is a forward-information leak that makes the comparison invalid. We later wrote `2c_OLS_LASSO_WF` (clean walk-forward selection) to address this, but the original LASSO3/LASSO5 results were still reported in earlier summary tables.

**Decision (post-audit, 2026-04-26)**: paper §4 demotes 2c results to "look-ahead caveat" footnote. Script `run_rung2c_selected_ols.py` deleted from repo on 2026-04-26 — was the only true dead-code script in the codebase.

---

## 2026-04-18: Data Pipeline V2 — WRDS Migration

### 2026-04-18 (data_changelog.md): V1 → V2 source migration

> Source: `docs/archive/data_changelog.md` · 2026-04-18.

**Why migrate**: V1 (yfinance + SEC XBRL) had stock-split contamination in EarningsYield (54 tickers with major splits, AAPL undercount 4.3×, GOOG 20×). XBRL parser also missed the `dei` namespace, causing 22.4% of GP/A to be NaN. V2 moves everything to WRDS for institutional-grade data.

| Component | Before (V1) | After (V2) | Why |
|---|---|---|---|
| Prices / Market Cap | yfinance (split-adjusted) | CRSP monthly (`prc`, `shrout`) | Eliminates stock-split contamination in EY |
| Fundamentals | SEC XBRL (parsed) | Compustat annual (`gp`, `at`, `ib`, etc.) | GP/A NaN: 22.4% → 0% |
| Forward Returns | yfinance price changes | CRSP monthly `ret` | No split/dividend ambiguity |
| Realized Vol | yfinance 21d rolling | CRSP daily 21d rolling | 11.5M daily return rows |
| Analyst Data | none | IBES (consensus + surprises) | SUE, revision, dispersion, breadth |
| Risk Factors | none | CRSP daily (computed) | 252-day beta, 60-day IVOL |
| Short Interest | none | Compustat short interest | Biweekly positions |
| Institutional | none | Thomson Reuters 13F | Quarterly ownership |
| Fama-French | none | WRDS FF library | 5 factors + momentum |
| CCM Link | none | CRSP-Compustat merged | PERMNO ↔ GVKEY |

**Total V2 WRDS data**: 16 parquet files, 19.7M rows, 251 MB.

### 2026-04-18 (wrds_data_summary.md): WRDS Data Inventory

> Source: `docs/archive/wrds_data_summary.md` · 2026-04-18 · Source: WRDS, BU subscription.

**Tier 0 — Core (Downloaded earlier)**:

| File | Rows | Description |
|---|---|---|
| `crsp_monthly.parquet` | 505,604 | CRSP monthly stock file — unadjusted price, returns, shares outstanding, market cap |
| `compustat_annual.parquet` | 117,570 | Compustat annual fundamentals — revenue, GP, assets, earnings, cash flow, CapEx |
| `compustat_quarterly.parquet` | 471,559 | Compustat quarterly fundamentals |
| `ccm_link.parquet` | 33,324 | CRSP-Compustat Merged link table |

**Tier 1 — Analyst & Factor Data**:

| File | Rows | Description |
|---|---|---|
| `ibes_consensus.parquet` | 1,353,848 | IBES consensus EPS — mean, median, std, analyst count, up/down revisions |
| `ibes_surprises.parquet` | 642,013 | IBES earnings surprises — actual vs. consensus, standardized surprise |
| `ibes_crsp_link.parquet` | 30,080 | IBES ticker → CRSP PERMNO mapping |
| `ff5_factors_monthly.parquet` | 144 | Fama-French 5 factors + momentum (MktRF, SMB, HML, RMW, CMA, UMD, RF) |
| `ff_factors_monthly.parquet` | 144 | Fama-French 3 factors + momentum |

**Tier 2 — Alternative & Risk Data**:

Short interest, 13F institutional ownership, CRSP daily returns (for beta + IVOL computation).

(Full inventory in `docs/archive/wrds_data_summary.md`.)

---

## 2026-04-10 to 2026-04-27: Pipeline Fixes (rolling)

> Source: `docs/archive/FIXES.md` · Last updated 2026-04-27.

Rolling document of code fixes applied during the audit cycle. Organized by severity tier.

### C-tier (Critical)

**C1: Z-Score Temporal Leakage** (`load_data.py:100-106`)
- Bug: pooled global mean/std across all dates and tickers (introduces look-ahead via stats)
- Fix: cross-sectional z-score per `signal_date`: `groupby("signal_date")[raw_col].transform(_cs_zscore)`
- Impact: affects all Rung 4-5 results. master_panel regenerated.

**C2: EarningsYield Split Contamination** (`load_data.py:96`)
- Bug: yfinance returns split-adjusted prices, SEC XBRL reports raw shares → `mktcap = adj_price × raw_shares` is undercount (AAPL 4.3×, GOOG 20×; 54 tickers affected)
- Fix v1 (reverted): blanket NaN for any ticker with a historical split (wiped 65% of EY data)
- Fix v2 (current): NaN only for periods where split timing creates the bias
- Impact: V1 EY results invalid; V2 (WRDS-based) bypasses the issue entirely

**C3: Type-A NaN Handling**
- Bug: economic measures undefined for certain sectors (e.g., GP/A for banks, Accruals for financials) were zero-filled, treating "undefined" as "average"
- Fix: keep NaN for sector-undefined; sector-cross-section median for new listings; missingness flags paired with imputation

### I-tier (Important)

**I1: Beta/IVOL Zero-Fill on New Listings**
- Bug: 252-day rolling beta is undefined for stocks with <252 trading days of history; same for IVOL
- Fix: sector-cross-sectional median fill on the affected periods, with `has_short_interest` style indicator considered (though no separate flag for beta — risk feature)

### D-tier (Mid-audit corrections)

**D1: Duplicate `corr_adj_ic_ensemble_model`** (`run_rung12_v2.py:368`)
- Bug: two function definitions, the second silently overrode the first (true Grinold-Kahn Barra). Paper labeled "True Barra" but ran simplified Corr(X)⁻¹·IC heuristic
- Fix: deleted line 368-410 duplicate; rerun V2/V3 with True Barra
- Impact: V3 1d Barra Sharpe 0.746 → **1.134** (new V3 #1); V2 0.497 → 0.982. CPCV reran.

**D2: `has_positive_earnings` Filter Bias** (`load_data.py:712`)
- Bug: indicator computed AFTER `|EY|≤1` plausibility filter that drops deeply-loss firms entirely → flag systematically biased toward profitable firms
- Fix: `EarningsYield_sign_raw = np.sign(EY)` captured before filter; flag derived from raw sign

**D3: Power Analysis sigma Hardcoded** (`statistical_tests.py:538`)
- Bug: `sigma = 0.10` hardcoded in `compute_power()`
- Fix: `compute_power(ic_std=...)` accepts observed std

(Full content in `docs/archive/FIXES.md`.)

---

## 2026-04-09: V1 Baseline Pipeline (now superseded)

> Source: `docs/archive/data-pipeline.md` · 2026-04-09 · Historical reference.

**V1 layout**:
- Raw: `data/raw/factor_panel.parquet`, `data/raw/xbrl_df.parquet`, ...
- Final: `data/baseline/model_dataset.parquet`

**V1 goal**: a complete supervised dataset $T = (X, y)$ with no missing predictor entries, no missing target entries, cross-sectionally standardized predictors and target.

**V1 features** (4 + macro):
- 4 firm-level: EarningsYield, AssetGrowth, Accruals, Momentum12_1
- + macro factors

**V1 status (today)**: superseded by V2 WRDS pipeline. `data/baseline/model_dataset.parquet` still exists and is consumed by `statistical_tests.py` for some legacy comparisons. V1 paths still callable via `--v1` CLI flag in `load_data.py` but not used for V2 results.

---

## 2026-03-31: Original Project Plan

> Source: `docs/final-plan.pdf` (118 KB, retained as the only PDF). 4-week plan from March 31.

**Original premise**: 5-rung complexity ladder on cross-sectional stock return prediction, walking from simplest theory-driven model to multi-task neural network. **13 pre-validated fundamental factors** (the original V1 set, before V2 expansion to 27) across $\sim$500 S&P 500 stocks.

**Original ladder definition**:

| Rung | Model | Function space | Regularization | Learned params | Key question |
|---|---|---|---|---|---|
| 1 | IC-weighted linear ensemble | Linear, fixed weights | None (theory-driven) | 0 | Baseline: how far does theory alone get us? |
| 2 | Lasso | Linear, learned weights | L1 sparsity | $\sim$13 | Does data-driven selection beat theory? |
| 3 | GAM with natural splines | Additive nonlinear | Smoothness penalty | $\sim$13 × knots | Does smooth nonlinearity help? |
| 4 | Single-task MLP | Nonlinear, unconstrained | L2 + dropout | $\sim$2,912 | Does free-form nonlinearity beat splines? |
| 5 | Multi-task MLP | Nonlinear + MTL | Shared layers (implicit) | $\sim$3,100 | Does aux supervision add value? |

**Original 4-week timeline**:
- Week 1 (Mar 31--Apr 6): Data pipeline + Rung 1 + Rung 2
- Week 2 (Apr 7--Apr 13): Rung 3 (GAM) + Rung 4 (single-task MLP) + statistical testing
- Week 3 (Apr 14--Apr 20): Rung 5 (MTL) + ablation
- Week 4 (Apr 21--Apr 27): **PCA visualization, permutation importance, effective-df plot. Report writing. Final PDF.**

**Original 3-person work split**:
- Person A (Eva): Architecture, Rung 4 + Rung 5, ablation, PCA visualization
- Person B: Rung 1 + Rung 2, walk-forward validation, Sharpe / risk metrics
- Person C: Rung 3 (GAM with splines), permutation importance, effective-df

**What changed in execution** (vs. original plan):
- Feature count: 13 → **27** (V3 expansion: Phase-2 features + missingness flags)
- "5a" originally meant single-task MLP within Rung 5 ablation; renamed to **Rung 4** for ladder consistency
- Added **CPCV evaluation** (López de Prado 2018) on top of walk-forward
- Added 5 sensitivity studies (audit cycle): GAM `n_splines` sweep, K-fold $\lambda$, XGBoost grid + GKX-tuned, MLP seed sensitivity, MLP HP search
- Final visualizations from original plan **partially delivered**: GAM `n_splines` sweep is effectively the effective-df plot; LASSO selection frequency is approximate permutation importance; **PCA never done** (still in TODO)

---

# Appendix A: How to Reproduce

```bash
# Active pipeline (project root):
python run_rung12_v2.py            # produces V2 panel results
python scripts/rung12_v3.py        # produces V3 panel results (paper Tab 1)
python run_cpcv_all_models.py      # produces CPCV 15-path distribution (paper Tab 3)
python main.py                     # produces Rung 4/5 MLP/MTL results

# Audit / ablation snapshots (frozen in audits/):
cd audits && python rung3a_gam_sweep.py       # GAM n_splines sweep
cd audits && python rung3b_audit.py           # XGB sensitivity grid + GKX-tuned
cd audits && python rung3a_gam_kfold_lambda.py # K-fold λ vs GCV (negative)
cd audits && python rung4_hp_search.py        # MLP HP search (negative)
cd audits && python rung45_seed_bagging.py    # Seed bagging (negative)
cd audits && python rung5_audit.py            # MLP seed sensitivity + hidden_dim sweep
```

See `RESULTS.md` at project root for the canonical mapping from paper-cited numbers to source CSVs and producing scripts.

---

# Appendix B: Active Code & File Structure

**Project root** (16 active `.py` after cleanup):

| File | Role |
|---|---|
| `config.py` | Feature lists, missingness indicators, V1/V2/V3 splits |
| `load_data.py` | V1/V2 panel assembly, NaN/winsorize logic |
| `metrics.py` | IC, IC-IR, Hit Rate, Long-Short Sharpe (canonical) |
| `statistical_tests.py` | Paired t-test, Diebold-Mariano, BHY correction, power analysis |
| `compute_derived_features.py` | CRSP daily → Beta (252d), IVOL (60d), realized vol |
| `download_wrds_tier12.py` | Tier 1+2 WRDS download (CRSP/Compustat/IBES/FF/short/13F) |
| `wrds_factor_builder.py` | Compustat → EarningsYield, GP/A, Accruals, etc. |
| `wrds_new_features.py` | IBES → SUE, AnalystRevision, Dispersion, Breadth |
| `regime.py` | HMM regime identification (Rung 5 MoE) |
| `run_rung12_v2.py` | Rung 1+2 driver (V2 panel) |
| `main.py` | Rung 4/5 MLP/MTL driver |
| `regmtl.py` | Regime-Gated MoE (Rung 5) |
| `regmtl_enhanced.py` | Enhanced MoE (Rung 5) |
| `cpcv_harness.py` | Combinatorial Purged K-fold harness (López de Prado 2018) |
| `run_cpcv_all_models.py` | CPCV driver (8 models × 15 paths) |
| `RESULTS.md` | Single source of truth for paper-cited numbers |

**`scripts/`** (analysis utilities, all run from CLI):

| File | Output |
|---|---|
| `rung12_v3.py` | V3 panel driver |
| `paired_ttest_miss_flags.py` | `output/paired_ttest_miss_flags.csv` |
| `ff3_alpha_test.py` | `output/ff3_alpha_test.csv` |
| `compute_turnover.py` | `output/turnover_*.csv` |
| `compute_ls_hit_rate.py` | `output/ls_hit_rate_*.csv` |
| `download_wrds_*.py` | Additional WRDS downloads |

**`audits/`** (12 frozen audit/ablation snapshots; `linguist-generated`):

| File | Purpose |
|---|---|
| `rung3_gam.py` | Original GAM driver (default n=10) |
| `rung3b_gbm.py` | Original GBM driver (default config) |
| `rung3a_gam_sweep.py` | n_splines sweep (consolidated coarse + fine-tune) |
| `rung3a_gam_kfold_lambda.py` | K-fold λ vs GCV |
| `rung3b_audit.py` | XGB 9-config grid + GKX-tuned |
| `rung4_hp_search.py` | Inner-CV HP search (negative) |
| `rung45_seed_bagging.py` | K=5 bagging (negative) |
| `rung5_audit.py` | MLP seed sensitivity + hidden_dim sweep |
| `rung5_planned.py` | MTL planned-aux ablation |
| `rung5_combined.py` | MTL combo-aux ablation |
| `lasso_v3_selection.py` | LASSO frequency analysis |
| `rung3a_audit_procedure.md` | Iterative-audit narrative |

**`src/`**:

| Path | Purpose |
|---|---|
| `src/models/mlp_audit.py` | Consolidated MLP class + train_mlp_fold (used by 3 audit scripts) |
| `src/data_pipeline/taxonomy_map.py` | Sector taxonomy (XBRL → GICS mapping) |
| `src/factor_library/academic_factors.py` | Academic factor definitions (Fama-MacBeth-style) |

---

# Appendix C: Figures Inventory (have / planned / suggested)

## Figures we have (in paper)

| # | Source data → figure | Section | Status |
|---|---|---|---|
| 1 | `fig_data_flow.pdf` (TikZ schematic) | §2 Data | placeholder, hand-drawn |
| 2 | `fig_ladder.pdf` (matplotlib FancyBboxPatch + arrows) | §3 Methodology | ✅ generated by `make_ladder_figure.py` |
| 3 | `fig_gam_n_splines.pdf` (dual-axis line) | §5.1 Default trap | ✅ |
| 4 | `fig_xgb_sensitivity.pdf` (3×3 heatmap) | §5.1 | ✅ |
| 5 | `fig_mlp_seeds.pdf` (bar + reference lines) | §5.2 | ✅ |
| 6 | `fig_cpcv_sharpe.pdf` (boxplot) | §6 CPCV | ✅ |
| 7 | `fig_pareto_with_bounds.pdf` (scatter + errorbars) | §6 | ✅ |
| 8 | `fig_cumret.pdf` (time-series multi-line) | §4 Results | ✅ |

## Figures planned but not yet implemented

| # | Figure | Source | Target section | Reason |
|---|---|---|---|---|
| 9 | `fig_pca.pdf` — PCA scree + PC1×PC2 scatter | 27 V3 features | §2.4 (proposed) | **Original plan Week 4 deliverable**; never done |
| 10 | `fig_perm_importance.pdf` — permutation importance per top-3 model | 1c FM, 1d Barra-GK, 3b XGB-GKX | §5.3 (proposed) | **Original plan Week 4 deliverable**; never done |
| 11 | `fig_effective_df.pdf` — DoF vs Sharpe scatter | All rungs | §3.1 or §7 (proposed) | **Original plan Week 4 deliverable**; partially via Fig 3 |

**Note for Fig 9 (PCA)**: recommend coloring by GICS sector (11 colors, interpretable cluster structure) rather than by return quintile. Sector clusters directly answer "which features cluster cross-sector vs. within-sector"; return-quintile coloring is a regression test only.

## Suggested additional figures

| # | Figure | Why useful | Effort |
|---|---|---|---|
| 12 | `fig_lasso_freq.pdf` — LASSO selection frequency bar chart | Shows which features survive walk-forward selection (proxy for permutation importance) | ~10 LOC, data already in `output/lasso_v3_selection_summary.csv` |
| 13 | `fig_correlation_matrix.pdf` — V3 feature correlation heatmap | Justifies Adaptive LASSO and Elastic Net by showing feature collinearity | ~15 LOC |
| 14 | `fig_regime_timeline.pdf` — HMM regime states over time | Visualizes Rung 5 MoE gate input; shows COVID/inflation regime shifts | ~30 LOC |
| 15 | `fig_v2_vs_v3.pdf` — V2 vs V3 ΔSharpe per model | Shows which Phase-2 features added vs. V2 baseline | ~15 LOC |

## Reproducibility

Active figure generation: `make_figures.py` + `make_ladder_figure.py` (gitignored, local-only). Both should be expanded to add Figs 9-15 above.

---

*End of document. Original sources archived under `docs/archive/`.*
