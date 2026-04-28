# HMM forward-filter rerun — Enhanced MoE 5a (ret-only)

Branch: `moe-hmm-forward-filter`
Run script: `run_6a_filtered.py`
Result: `output/results_5a_enhanced_moe_filtered.parquet`
Log: `output/run_5a_filtered_log.txt`
Wall: 17.2 min, 58 folds, CPU

---

## Hypothesis

Original `regime.py` propagated regime posteriors after `train_end` via
prior-only `current_probs @ T`, ignoring test-month market observations.
This decays toward the HMM stationary distribution and likely caused
the gate to receive a weak / uniform regime signal — producing
expert collapse.

Fix: standard HMM forward filtering on test-month market covariates
(VIX, term spread, credit spread, market return, market RV) using the
**train-fit scaler + train-fit HMM** only. No alpha-target leakage.

---

## Result

| Metric | V1 (prior-only) | **V2 (forward-filter)** | Δ |
|---|---:|---:|---:|
| LS Sharpe | 0.481 | **0.458** | −0.023 |
| IC mean | 0.0322 | **0.0264** | −0.006 |
| IC-IR | 0.138 | 0.123 | −0.015 |
| n_months | 58 | 57 | −1 |

**Sharpe difference within fold-level noise** (~5%). The Enhanced MoE
ret-only result is robust to this methodological choice.

---

## Gate distribution diagnostic

Per-month mean gate weights (across 57 test folds):

```
              mean    std     range          
expert 0:     0.321   0.120   [0.044, 0.709]
expert 1:     0.334   0.106   [0.145, 0.586]
expert 2:     0.345   0.116   [0.090, 0.595]

Avg per-month max-min spread: 0.229
```

**Interpretation**:
- Gate is no longer constant 1/3 1/3 1/3 — single-month spread reaches 0.7
- But cross-month average remains near uniform — specialization is
  transient, not systematic
- Forward-filter resolved the technical collapse but did NOT
  produce sustained expert specialization

---

## Why Sharpe didn't improve

Likely root causes (ranked):

1. **Experts learned similar functions**. Even with non-uniform gate,
   if expert_k(X) ≈ expert_j(X) for all (k, j), the gating choice is
   irrelevant. This is "soft collapse" — gate distinguishes states
   but experts don't.

2. **HMM regime signal is weak**. 3-state HMM on macro features may
   not distinguish meaningful return regimes — dominant state covers
   most months, posterior shifts only at transitions.

3. **SNR ceiling**. At monthly cross-sectional IC ≈ 0.02, no MoE
   architecture (with ~9.5K weights) is going to beat the linear
   ceiling Sharpe 1.13 set by Barra. The bottleneck is signal, not
   gate mechanics.

---

## Implication for paper

This is a **stronger null-result writeup**, not a weaker one:

> "We tested two HMM regime-propagation schemes (prior-only and
> forward-filtered) and observed Enhanced MoE LS Sharpe in the band
> [0.46, 0.48] — well below the linear ceiling 1.13. The null result
> is robust to this methodological choice."

The original concern (expert collapse from weak gate) is partially
real but does not change the verdict. The fundamental constraint is
SNR, not implementation.

---

## Reproduce

```bash
git checkout moe-hmm-forward-filter
python run_6a_filtered.py        # ~17 min on M-series CPU
```

To rerun **with reduced HMM features** (e.g. ret + vol only), edit
`config.py:89` `HMM_FEATURE_COLS` and rerun.

---

## Possible next-step experiments

To address the underlying expert collapse (not the regime signal),
consider on a follow-up branch:

1. **Load-balancing aux loss** (Switch Transformer):
   `loss += α · Σ_i f_i · P_i`, α ∈ [0.01, 0.1]
2. **Top-k hard gating** with k=1 or k=2
3. **Different seeds per expert** at init (cheapest)
4. **Gate entropy regularizer** (negative): `loss += −β · H(gate)`
5. **Reduced HMM_FEATURE_COLS** to `[mkt_ret_1m, mkt_rv_1m]`
   (pure Hamilton 1989 setup)

None of these are expected to clear Sharpe 1.13, but they would
verify that *if* Enhanced MoE could specialize, the SNR ceiling
would still bind.
