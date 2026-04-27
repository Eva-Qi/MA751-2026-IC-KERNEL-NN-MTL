# Rung 4 HP Search — Audit Findings (NEGATIVE)

> Updated 2026-04-26. From `rung4_hp_search.py`.

## TL;DR

We added inner TimeSeriesSplit(K=3) hyperparameter search over `hidden_dim ∈ {16, 32, 64}`, `lr ∈ {1e-4, 1e-3, 1e-2}`, `dropout ∈ {0.0, 0.1, 0.2}` (27 configs × 3 inner splits × 58 outer folds = 4698 MLP fits, ~58 min wall clock). 

**Result**: HP-tuned MLP gives **IC=+0.0016 / Sharpe=0.111**, which is **WORSE** than the hardcoded `seed=42, h=64, lr=1e-3, dropout=0.10` baseline (IC=+0.0007 / Sharpe=0.347). Sharpe degradation = -0.236.

**Cause**: Inner-CV signal is noise-dominated at our SNR (~0.01 IC). Grid-searching HP over noisy inner CV picks HP that fits validation noise → outer test sees over-fit to noise → degraded performance.

This is a **classic over-tuning trap on low-SNR validation**. The "more search → better tuning" intuition is wrong when the validation signal is below the noise floor.

---

## Result table

| Configuration | IC mean | IC std | IC-IR | Hit | LS Sharpe |
|---|---|---|---|---|---|
| Hardcoded (seed=42, h=64, lr=1e-3, drop=0.10) | +0.0007 | 0.103 | 0.007 | 51.7% | **0.347** |
| **Inner-CV HP-tuned (this run)** | **+0.0016** | **0.128** | **0.012** | **48.2%** | **0.111** ❌ |
| Single-seed mean (5-seed audit) | +0.0109 | (across-seed: 0.008) | — | — | 0.412 |
| K=5 raw bagging | +0.0076 | 0.139 | 0.054 | 60.3% | 0.394 |
| Ridge α=1 (linear floor) | +0.0164 | — | — | 53.4% | 0.925 |

---

## Chosen HP distribution (58 folds)

| HP | Count | Note |
|---|---|---|
| `hidden_dim=16` | 18 (31%) | smaller, suggests low-SNR penalty |
| `hidden_dim=32` | 4 (7%) | rarely chosen |
| `hidden_dim=64` | 36 (62%) | original default, often picked |
| `lr=1e-4` | 2 (3%) | rarely chosen |
| `lr=1e-3` | 27 (47%) | original default |
| `lr=1e-2` | 29 (50%) | aggressive, often chosen |
| `dropout=0.0` | 31 (53%) | most common — no dropout |
| `dropout=0.1` | 17 (29%) | original default |
| `dropout=0.2` | 10 (17%) | aggressive |

**Bimodal selection** in `hidden_dim` (16 vs 64) and `lr` (1e-3 vs 1e-2) is the smoking gun: HP choice is **unstable across folds**, indicating the inner CV is responding to noise rather than to a stable HP-vs-IC curve.

---

## Why it failed: over-tuning on noisy validation

**The setup**:
- Outer fold: 60+ training months, 1 test month, predict ~440 stocks' returns
- Inner CV: K=3 TimeSeriesSplit on training months → each inner-val fold is ~10-20 months × ~440 stocks
- Inner IC scoring on Spearman rank correlation (matches outer eval)

**The math that breaks**:

For `n` test obs, the standard error of Spearman correlation under H0 is approximately `1/sqrt(n)`. With 20 months × 440 stocks = 8800 obs, **SE ≈ 0.011**. Our true signal IC ≈ 0.01-0.02. **Signal ≈ noise floor**.

When 27 HP configs are evaluated on this noise floor, the "best" HP is **whichever config got luckiest with the inner-val noise**, not whichever has highest true IC. Result: HP overfits to inner-val realization, generalizes poorly to outer test.

**Concrete signature**: HP distribution should converge if the search is finding signal. Ours bimodal-spread → noise-driven, not signal-driven.

---

## Bug noted: NaN predictions at high lr + high dropout

Folds 25-26 produced `IC = NaN` when chosen HP was `(64, 1e-2, 0.2)`. The combination of high lr (0.01) + high dropout (0.2) appears to make the MLP predictions degenerate (likely all-zero or all-NaN due to gradient explosion or dead neurons). Final IC mean computed over 56 valid folds, not 58.

**Fix for future**: Add `if np.std(y_pred) < 1e-10: fallback to baseline HP` guard inside the HP search loop.

---

## Decision: do NOT re-run with smaller search space

Reasoning:
1. Reducing search to 9 configs (3 hidden × 3 lr × 1 dropout) might recover some Sharpe — but is unlikely to beat the hardcoded baseline, much less Ridge 0.925
2. The Rung 5 audit conclusion ("MLP < linear, complexity dominated by simple linear") is now **strongly confirmed by 4 independent attempts to improve MLP**:
   - Single-seed sweep → all 5 seeds < Ridge
   - Architecture sweep (hidden=8,16,32,64) → all < Ridge
   - K=5 raw bagging → worse than single-seed (P7 finding)
   - Inner-CV HP search → much worse than hardcoded (P8 finding)
3. Compute time better spent on CPCV harness (task #5), which produces report-grade path-uncertainty bounds

We document this finding and move on. Future work could implement narrower HP search (e.g., 9 configs) with explicit signal-vs-noise floor comparison.

---

## Lessons logged

This finding directly produced **code-council Part 12 P8**: "Inner-CV HP search degrades outer performance when validation signal ≈ noise floor. Larger search space → more over-fit to validation noise."

---

## File locations

- Script: `rung4_hp_search.py`
- Per-fold summary: `output/rung4_hp_search_summary.csv`
- Per-fold per-HP-config diagnostics: `output/rung4_hp_search_diag.csv`
- Full log: `output/rung4_hp_search_log.txt`
- This document: `docs/rung4_hp_search_findings.md`

## Changelog

- 2026-04-26: Initial document. Negative result. Decision: skip retry, move to CPCV (task #5).
