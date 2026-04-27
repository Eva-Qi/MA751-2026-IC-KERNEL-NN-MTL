# Rung 3a GAM Audit — Procedure & Findings

> Documents the iterative two-stage `n_splines` audit so we don't have to keep
> two near-identical scripts side by side.

## Why this audit

`rung3_gam.py` (the original Rung 3a driver) used `pygam.LinearGAM` with the
package's default `n_splines=10`. The audit asked: **is 10 the right
complexity for our IC≈0.01 SNR regime, or is it over-fitting?**

## Procedure

We performed the audit in **two stages**, motivated by the typical
sensitivity-sweep workflow: first locate the rough peak with a coarse grid,
then bracket it with a fine grid.

### Stage 1 — Coarse sweep (April 24)

- Grid: `n_splines ∈ {3, 5, 10, 15, 20}`
- Output: `output/rung3a_n_splines_sweep.csv`
- Originally implemented in `audits/rung3a_gam_audit.py` (now consolidated)
- Result: peak Sharpe at `n=5` (0.792); default `n=10` gave 0.555 — a
  material gap. Default was over-fitting.

### Stage 2 — Fine-tune (April 25)

- Grid: `n_splines ∈ {4, 6, 7, 8, 9}` (bracketing the Stage-1 winner `n=5`)
- Output: `output/rung3a_n_splines_finetune.csv`
- Originally implemented in `audits/rung3a_gam_finetune.py` (now consolidated)
- Result: Sharpe peaks shifted to `n=4` (0.828); IC peak stays at `n=5`
  (0.0146); monotone decay for `n ≥ 6`.

### Stage 3 — Combined sweep (today, post-cleanup)

To avoid duplicated code, both stage scripts were merged into a single
`audits/rung3a_gam_sweep.py` covering the full grid
`{3, 4, 5, 6, 7, 8, 9, 10, 15, 20}` in one execution. This is the
canonical script if you ever need to re-run the audit.

The two original CSVs (`rung3a_n_splines_sweep.csv`,
`rung3a_n_splines_finetune.csv`) remain in `output/` because the paper
figures and tables cite them directly.

## Findings (combined)

| n_splines | IC mean | IC-IR | LS Sharpe | Hit % |
|---|---:|---:|---:|---:|
| 3 | NaN (pygam constraint: n_splines > spline_order) | — | — | — |
| **4** | +0.0141 | 0.111 | **0.828** ⭐ | 51.7 |
| **5** | **+0.0146** ⭐ | **0.126** ⭐ | 0.792 | 50.0 |
| 6 | +0.0115 | 0.101 | 0.654 | 55.2 |
| 7 | +0.0095 | 0.087 | 0.561 | 55.2 |
| 8 | +0.0057 | 0.053 | 0.464 | 48.3 |
| 9 | +0.0047 | 0.046 | 0.468 | 46.6 |
| 10 (default) | +0.0068 | 0.066 | 0.555 | 53.4 |
| 15 | +0.0062 | 0.065 | 0.505 | 53.4 |
| 20 | +0.0071 | 0.089 | 0.406 | 55.2 |

**Pareto choice between `n=4` and `n=5`**:
- `n=4` wins LS Sharpe by +0.036
- `n=5` wins IC + IC-IR by negligible margins
- IC delta is within fold noise; Sharpe delta is meaningful

**Practical recommendation**: `n_splines=4` for deployment (best risk-adjusted
return); `n_splines=5` for paper-style ranking metrics. Both materially better
than `pygam`'s default `n=10`.

## Lessons

This is a textbook **Hyperparameter Default Trap** (P5 in `code-council` Part 12):
framework defaults target moderate-SNR generic-ML benchmarks. At our IC≈0.01
SNR, defaults overshoot optimal complexity. Always sensitivity-sweep before
declaring a method "failed."

The two-stage audit (coarse → fine) is the right workflow but doesn't justify
keeping two scripts in the repo permanently — they're 95\% identical. The
consolidated `rung3a_gam_sweep.py` reproduces both stages in one run.

See also `code-council` skill, Part 12 §P9 (dead-code accumulation in research
codebases) for the broader meta-lesson.
