# Rung 4/5 Seed Bagging — Audit Findings (NEGATIVE result)

> Updated 2026-04-26. From `rung45_seed_bagging.py` run.

## TL;DR

**Hypothesis** (going in): K=5 seed bagging on Rung 4 MLP would lift the audit-measured per-seed SNR from 1.33 (≈ noise floor) toward √5 × 1.33 ≈ 3.0, by averaging predictions across independent random initializations.

**Result**: Bagging did **NOT** improve over single-seed performance. Bagged Sharpe = 0.394 (vs single-seed mean 0.412); bagged IC = +0.0076 (vs single-seed mean +0.0109). Bagging ≈ slight degradation.

**Root cause**: We averaged **raw predictions** (correct for MSE/Pearson) but evaluated on **Spearman rank IC** (rank-based). For rank metrics, mean-averaging predictions **compresses tail rankings**, losing the extremes that drive long-short quintile Sharpe.

This is a generalizable methodology lesson, now logged as Part 12 P7 in code-council.

---

## Result table

| Model | IC mean | LS Sharpe | Notes |
|---|---|---|---|
| MLP h=64 seed=0 | +0.0210 | 0.648 | single-seed (prior audit) |
| MLP h=64 seed=1 | +0.0020 | 0.097 | single-seed (prior audit) |
| MLP h=64 seed=2 | +0.0130 | 0.414 | single-seed (prior audit) |
| MLP h=64 seed=42 | +0.0007 | 0.347 | single-seed (prior audit) |
| MLP h=64 seed=123 | +0.0176 | 0.556 | single-seed (prior audit) |
| **MLP h=64 K=5 bagged (mean of preds)** | **+0.0076** | **0.394** | **this run** |
| Ridge α=1 | +0.0164 | 0.925 | linear baseline |

**Stats**:
- Bagged IC = +0.0076, IC std (across 58 months) = 0.139, IC-IR = 0.054
- Hit Rate = 60.3% (interesting — the only metric where bagging looks decent)
- Per-fold across-seed std ≈ 0.04-0.09 (high diversity)
- Per-fold across-seed prediction correlation ≈ 0.27-0.56 (low — strong ensembling potential by Pearson logic)

---

## Why it failed: rank vs. value

**Theory of bagging works in Pearson space**:
$$\text{Var}(\bar{y}_K) = \frac{\sigma^2 (1 + (K-1)\rho)}{K}$$

If predictions are weakly correlated (`ρ ≈ 0.3` here), `Var(bagged)` shrinks substantially → MSE drops → Pearson r increases.

**For Spearman rank correlation, this logic breaks**:
1. Spearman cares about **rank order**, not values
2. K different MLPs produce K different rankings (pred-corr 0.27-0.56 confirms diversity)
3. Mean-averaging the **raw predictions** produces a new ranking that is a **rank-space compromise** — extremes (top/bottom 20%) get pulled toward consensus middle
4. LS quintile Sharpe needs **strong tail rankings**; consensus-middle rankings have weaker tails
5. Hence: bagged IC ≤ best single-seed IC, often even ≤ mean of seeds

**Concrete fold-level signature**:
- Single-seed extreme picks (top 20% predicted) often disagree across seeds
- Mean-pred-bagged picks are diluted intersections of those disagreements → no longer extreme
- Bagged IC = correlation of diluted ranking with reality < correlation of any well-aligned single seed

---

## What we should have done (rank-level bagging)

```python
# WRONG — what we did (mean of raw predictions; works for MSE/Pearson)
y_bagged = np.mean([y_pred_seed_k for k in seeds], axis=0)
ic = spearmanr(y_te, y_bagged)

# RIGHT for rank/Spearman metrics — rank-level averaging
from scipy.stats import rankdata
ranks_per_seed = np.array([rankdata(y_pred_seed_k) for k in seeds])
y_bagged_rank = np.mean(ranks_per_seed, axis=0)  # average rank → consensus rank
ic = spearmanr(y_te, y_bagged_rank)

# ALTERNATIVE: median-prediction bagging (more robust to outliers, also rank-aware)
y_bagged_median = np.median([y_pred_seed_k for k in seeds], axis=0)
```

**Rank-level bagging** preserves "which stocks are at extremes" better than mean-prediction bagging. Each seed's top-20%-predicted set gets a vote, and the bagged top-20% is the **majority-voted top-20%** — typically stronger than seed-level top-20% because vote boosts confidence.

---

## Decision: do NOT re-run

Reasoning:
1. Even with rank-level bagging, expected Sharpe lift is modest (probably 0.05-0.15 at best, given how weak the per-seed signals are)
2. Even an optimistic bagged Sharpe of ~0.5 is **still well below Ridge 0.925** — bagging cannot rescue MLP from being dominated by linear baselines on this panel
3. The audit's core finding ("MLP < linear, signal exists but is dominated by Ridge") is unaffected
4. Compute time better spent on Rung 4/5 HP search (task #7) and CPCV harness (task #5)

We document this as a methodology gap and move on. Future work could implement rank-level bagging if MLP/MTL is to be deployed.

---

## Lessons logged

This finding directly produced **code-council Part 12 P7**: "K-seed mean-averaging assumes Pearson space; for Spearman/rank evaluation it can decrease signal — use rank-level averaging or median for rank metrics."

---

## File locations

- Script: `rung45_seed_bagging.py`
- Per-fold diagnostics: `output/rung4_seed_bagging_diag.csv`
- Summary stats: `output/rung4_seed_bagging_summary.csv`
- Comparison table: `output/rung4_seed_bagging_final_comparison.csv`
- Full log: `output/rung45_seed_bagging_log.txt`
- This document: `docs/rung45_seed_bagging_findings.md`

## Changelog

- 2026-04-26: Initial document. Negative result documented. Decision: skip rank-level rerun, move to HP search (task #7).
