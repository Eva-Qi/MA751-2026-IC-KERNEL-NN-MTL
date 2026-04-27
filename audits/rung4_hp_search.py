"""
Rung 4 MLP — TimeSeriesSplit Inner-CV Hyperparameter Search

MOTIVATION:
  rung5_audit.py showed MLP with hardcoded hidden=64, lr=1e-3, dropout=0.10 has:
    - 5-seed mean IC = +0.0108, std = 0.0082 → SNR = 1.33 (barely above init noise)
    - All 5 seeds Sharpe < Ridge baseline 0.925
  Hidden-dim sweep showed hidden=8 ≥ hidden=64, suggesting over-parameterization.
  Seed-bagging (rung45_seed_bagging.py) failed: bagged Sharpe 0.394 < mean 0.412
  due to rank-vs-Pearson space mismatch in ensemble averaging.

APPROACH:
  Per-fold HP selection via inner TimeSeriesSplit(K=3, gap=1).
  K=3 chosen (not K=5) because our 60-month minimum outer-train window gives
  ~20 months per inner segment — K=5 would leave <15 months inner-train in early folds.

GRID:
  hidden_dim ∈ [16, 32, 64]
  lr ∈ [1e-4, 1e-3, 1e-2]
  dropout ∈ [0.0, 0.1, 0.2]
  → 27 configs per outer fold

INNER SCORING: Spearman IC (not MSE) — matches outer evaluation metric.
  This was a known bug in earlier scripts, fixed in run_rung12_v2.py.

Total MLP fits: 27 configs × 3 inner splits × 58 outer folds ≈ 4698
Expected runtime: 6-12 hours (background, nohup)

SANITY CHECKS (from task spec):
  Final HP-tuned IC should land in [+0.005, +0.025]
  Final HP-tuned Sharpe should land in [+0.3, +0.9]
  Most common hidden: probably 16 or 32 (over-param penalty for 64 at low SNR)
  Most common lr: probably 1e-3
  Most common dropout: 0.1
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate
from src.models.mlp_audit import MLP, train_mlp_fold  # Category C + D(Group2) consolidation

FEATURES = ALL_FEATURE_COLS_V3_WITH_MISS
DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

# ── HP Grid ──────────────────────────────────────────────────────────────────
HP_GRID = list(itertools.product(
    [16, 32, 64],       # hidden_dim
    [1e-4, 1e-3, 1e-2], # lr
    [0.0, 0.1, 0.2],    # dropout
))
# 27 configs total: (hidden, lr, dropout)

# Inner CV: K=3, gap=1 month.
# K=3 chosen over K=5: with 60-month min outer-train, K=3 gives ~20 months per segment.
# K=5 would squeeze inner-train to ~15 months in early outer folds — too thin.
INNER_K = 3
INNER_GAP_MONTHS = 1

# Outer walk-forward: 60-month min train + 1-month purge (mirrors rung5_audit)
OUTER_SEED = 42


# MLP class and train_mlp_fold removed — imported from src.models.mlp_audit (Category C + D consolidation)


# ── Inner CV — month-aware K=3 splitter ──────────────────────────────────────

def _inner_month_splits(train_dates, n_splits=INNER_K, gap_months=INNER_GAP_MONTHS):
    """Return list of (train_idx, val_idx) with month-aligned inner splits.

    Adapted from run_rung12_v2.py _month_aware_cv with K=3.
    Expanding-window style: fold k trains on months [0..k*seg], validates on
    the next segment (after gap_months purge). This respects time order —
    no random K-fold which would leak future data.

    Returns list of (train_idx, val_idx) arrays, or None if not enough data.
    """
    unique_months = np.array(sorted(np.unique(train_dates)))
    n_months = len(unique_months)
    seg = max(1, n_months // (n_splits + 1))
    splits = []
    for k in range(n_splits):
        train_end_idx = (k + 1) * seg - 1
        val_start_idx = train_end_idx + 1 + gap_months
        val_end_idx = min(val_start_idx + seg - 1, n_months - 1)
        if val_start_idx > val_end_idx or val_start_idx >= n_months:
            continue
        train_end_month = unique_months[train_end_idx]
        val_start_month = unique_months[val_start_idx]
        val_end_month = unique_months[val_end_idx]
        tr_idx = np.where(train_dates <= train_end_month)[0]
        va_idx = np.where(
            (train_dates >= val_start_month) & (train_dates <= val_end_month)
        )[0]
        # Minimum sanity: at least 30 train obs, 10 val obs
        if len(tr_idx) < 30 or len(va_idx) < 10:
            continue
        splits.append((tr_idx, va_idx))
    return splits if splits else None


def _inner_cv_score(X_outer_tr, y_outer_tr, dates_outer_tr,
                    hidden, lr, dropout, seed):
    """Run K=3 inner TimeSeriesSplit for one HP config; return mean Spearman IC.

    Scaler is fit on each inner-train independently (no leakage from inner-val).
    Returns np.nan if inner splits cannot be constructed.
    """
    splits = _inner_month_splits(dates_outer_tr)
    if splits is None:
        return np.nan

    ics = []
    for split_idx, (tr_idx, va_idx) in enumerate(splits):
        X_itr = np.nan_to_num(X_outer_tr[tr_idx], nan=0.0)
        y_itr = y_outer_tr[tr_idx]
        X_iva = np.nan_to_num(X_outer_tr[va_idx], nan=0.0)
        y_iva = y_outer_tr[va_idx]

        if len(np.unique(y_iva)) < 5:
            continue  # can't compute meaningful Spearman

        # Scaler fit on inner-train only (no leakage)
        sc = StandardScaler()
        X_itr_s = sc.fit_transform(X_itr)
        X_iva_s = sc.transform(X_iva)

        # Use inner-val as both the "val" (early stop) and "test" (IC scoring).
        # We pass a dummy X_te = X_iva so train_mlp_fold returns inner-val predictions.
        inner_seed = seed * 10000 + split_idx
        y_pred_inner = train_mlp_fold(
            X_itr_s, y_itr, X_iva_s, y_iva, X_iva_s,
            hidden=hidden, seed=inner_seed, lr=lr, dropout=dropout,
        )
        ic = spearmanr(y_iva, y_pred_inner).statistic
        if not np.isnan(ic):
            ics.append(float(ic))

    return float(np.mean(ics)) if ics else np.nan


# ── Main walk-forward with HP search ─────────────────────────────────────────

def walk_forward_hp_search(df):
    """
    Outer walk-forward: for each test month
      1. Inner K=3 TimeSeriesSplit HP search on outer train
      2. Pick best HP by mean Spearman IC
      3. Refit on full outer train at chosen HP (seed=42)
      4. Predict on outer test month

    Returns:
      monthly_results   — list of per-fold result dicts
      diag_rows         — list of per-fold per-HP inner IC dicts
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS
    n_folds = len(months) - start

    monthly_results = []
    diag_rows = []
    t_wall = time.time()

    print(f"HP grid: {len(HP_GRID)} configs × K={INNER_K} inner splits × {n_folds} folds "
          f"= {len(HP_GRID) * INNER_K * n_folds} MLP fits (upper bound)")
    print(f"Features: {len(FEATURES)}")
    print("=" * 100, flush=True)

    for fold_idx, (i, test_month) in enumerate(
        enumerate(months[start:], start=start), start=1
    ):
        t_fold = time.time()

        # Outer train / test split (same as rung5_audit.py)
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]

        if len(df_te) < 10:
            print(f"  fold {fold_idx:3d}/{n_folds}  test_month={test_month.date()}  "
                  f"SKIP: df_te < 10 rows", flush=True)
            continue

        # Use last 10% tail of outer-train as the early-stop val for final refit
        # (mirrors rung5_audit.py — same pattern for consistency)
        tr_months_list = sorted(df_tr[DATE_COL].unique())
        n_val_m = max(1, int(len(tr_months_list) * 0.10))
        val_months_set = set(tr_months_list[-n_val_m:])
        df_tr_main = df_tr[~df_tr[DATE_COL].isin(val_months_set)]
        df_val = df_tr[df_tr[DATE_COL].isin(val_months_set)]

        if len(df_val) < 10 or len(df_tr_main) < 100:
            print(f"  fold {fold_idx:3d}/{n_folds}  test_month={test_month.date()}  "
                  f"SKIP: val < 10 or train < 100", flush=True)
            continue

        # Raw features for inner CV (NaN → 0 done inside _inner_cv_score per split)
        X_outer_tr_raw = df_tr_main[FEATURES].values
        y_outer_tr = df_tr_main[TARGET_COL].values
        dates_outer_tr = df_tr_main[DATE_COL].values

        # ── Inner HP search ──────────────────────────────────────────────────
        best_hp = None
        best_inner_ic = -np.inf
        hp_diag = {}

        for hp_idx, (hidden, lr, dropout) in enumerate(HP_GRID):
            inner_ic = _inner_cv_score(
                X_outer_tr_raw, y_outer_tr, dates_outer_tr,
                hidden=hidden, lr=lr, dropout=dropout, seed=OUTER_SEED,
            )
            hp_key = (hidden, lr, dropout)
            hp_diag[hp_key] = inner_ic

            if not np.isnan(inner_ic) and inner_ic > best_inner_ic:
                best_inner_ic = inner_ic
                best_hp = hp_key

        # Fallback if all inner ICs are NaN
        if best_hp is None:
            best_hp = (32, 1e-3, 0.1)
            best_inner_ic = np.nan
            print(f"  fold {fold_idx:3d}/{n_folds}  WARN: all inner ICs NaN, "
                  f"using fallback HP {best_hp}", flush=True)

        best_hidden, best_lr, best_dropout = best_hp

        # ── Final refit on full outer train at chosen HP ─────────────────────
        X_tr_raw = np.nan_to_num(df_tr_main[FEATURES].values, nan=0.0)
        y_tr = df_tr_main[TARGET_COL].values
        X_val_raw = np.nan_to_num(df_val[FEATURES].values, nan=0.0)
        y_val = df_val[TARGET_COL].values
        X_te_raw = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[TARGET_COL].values

        sc = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr_raw)
        X_val_s = sc.transform(X_val_raw)
        X_te_s  = sc.transform(X_te_raw)

        fold_seed = OUTER_SEED * 1000 + i
        y_pred = train_mlp_fold(
            X_tr_s, y_tr, X_val_s, y_val, X_te_s,
            hidden=best_hidden, seed=fold_seed,
            lr=best_lr, dropout=best_dropout,
        )

        ic = float(spearmanr(y_te, y_pred).statistic)
        elapsed_fold = time.time() - t_fold
        elapsed_total = time.time() - t_wall

        print(
            f"  fold {fold_idx:3d}/{n_folds}  [{elapsed_fold:.1f}s]  "
            f"test={test_month.date()}  IC={ic:+.4f}  "
            f"hp=({best_hidden},{best_lr:.0e},{best_dropout})  "
            f"inner_IC={best_inner_ic:+.4f}  "
            f"total={elapsed_total/60:.1f}min",
            flush=True,
        )

        # Monthly result (for Sharpe aggregation)
        monthly_results.append({
            DATE_COL:       test_month,
            "IC":           ic,
            "best_hidden":  best_hidden,
            "best_lr":      best_lr,
            "best_dropout": best_dropout,
            "best_inner_ic": best_inner_ic,
            "pred_std":     float(np.std(y_pred)),
            "pred_mean":    float(np.mean(y_pred)),
            "y_pred_list":  y_pred.tolist(),
            "y_true_list":  y_te.tolist(),
            "tickers":      df_te[STOCK_COL].values.tolist(),
        })

        # Per-HP diagnostic row
        base_diag = {
            "fold":        fold_idx,
            "test_month":  test_month,
            "best_hidden": best_hidden,
            "best_lr":     best_lr,
            "best_dropout": best_dropout,
            "best_inner_ic": best_inner_ic,
            "outer_ic":    ic,
        }
        for (h, l, d), ic_val in hp_diag.items():
            col = f"h{h}_lr{l:.0e}_dr{d}"
            base_diag[col] = ic_val
        diag_rows.append(base_diag)

    return monthly_results, diag_rows


# ── Aggregation helpers ───────────────────────────────────────────────────────

def aggregate_results(monthly_results):
    """Compute IC/Sharpe/Hit from monthly results list."""
    monthly_df = pd.DataFrame(monthly_results)
    if len(monthly_df) == 0:
        return {}, monthly_df

    ic_series = monthly_df["IC"].dropna()

    rows = []
    for _, r in monthly_df.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t,
                         "y_pred": yp, "y_true": yt})
    expanded = pd.DataFrame(rows)
    sharpe = compute_long_short_sharpe(expanded)

    stats = {
        "IC_mean":   float(ic_series.mean()),
        "IC_std":    float(ic_series.std()),
        "IC_IR":     float(compute_ic_ir(ic_series)),
        "Hit_Rate":  float(compute_hit_rate(ic_series)),
        "LS_Sharpe": float(sharpe),
        "n_months":  int(len(ic_series)),
    }
    return stats, monthly_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("RUNG 4 MLP — TimeSeriesSplit Inner-CV Hyperparameter Search")
    print("=" * 100)
    print(f"Grid: hidden∈{[h for h,_,_ in HP_GRID[:3]]}...  "
          f"lr∈{sorted(set(l for _,l,_ in HP_GRID))}  "
          f"dropout∈{sorted(set(d for _,_,d in HP_GRID))}")
    print(f"Inner K={INNER_K}, gap={INNER_GAP_MONTHS}m | Outer: "
          f"min_train={DEFAULT_MIN_TRAIN_MONTHS}m, purge={DEFAULT_PURGE_MONTHS}m | "
          f"seed={OUTER_SEED}")
    print(f"Started: {pd.Timestamp.now()}", flush=True)

    print("\nLoading V3 panel...", flush=True)
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, "
          f"{len(FEATURES)} features", flush=True)

    # ── Walk-forward HP search ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("WALK-FORWARD: Inner-CV HP search per fold")
    print("=" * 100, flush=True)

    t0 = time.time()
    monthly_results, diag_rows = walk_forward_hp_search(df)
    elapsed = time.time() - t0

    print(f"\nWalk-forward complete: {elapsed/60:.1f} min", flush=True)

    if len(monthly_results) == 0:
        print("ERROR: No folds returned results. Aborting.", flush=True)
        return

    # ── Aggregate ──────────────────────────────────────────────────────────
    stats, monthly_df = aggregate_results(monthly_results)

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"  n_months  : {stats['n_months']}")
    print(f"  IC mean   : {stats['IC_mean']:+.4f}")
    print(f"  IC std    : {stats['IC_std']:.4f}")
    print(f"  IC IR     : {stats['IC_IR']:.3f}")
    print(f"  Hit Rate  : {stats['Hit_Rate']:.1%}")
    print(f"  LS Sharpe : {stats['LS_Sharpe']:.3f}", flush=True)

    # ── HP frequency analysis ───────────────────────────────────────────────
    print("\n  === Chosen HP Distribution ===")
    hp_counts_h = monthly_df["best_hidden"].value_counts().sort_index()
    hp_counts_l = monthly_df["best_lr"].value_counts().sort_index()
    hp_counts_d = monthly_df["best_dropout"].value_counts().sort_index()
    print(f"  hidden_dim: {dict(hp_counts_h)}")
    print(f"  lr:         {dict(hp_counts_l)}")
    print(f"  dropout:    {dict(hp_counts_d)}", flush=True)

    # ── Comparison vs baselines ─────────────────────────────────────────────
    BASELINE_IC_S42     = +0.0007
    BASELINE_SHARPE_S42 = 0.347
    BASELINE_RIDGE      = 0.925

    print("\n  === Comparison vs Baselines ===")
    print(f"  MLP h64 lr1e-3 seed42 (hardcoded):  IC={BASELINE_IC_S42:+.4f}, "
          f"Sharpe={BASELINE_SHARPE_S42:.3f}")
    print(f"  Ridge α=1 (floor):                  Sharpe={BASELINE_RIDGE:.3f}")
    print(f"  HP-tuned MLP (this run):             IC={stats['IC_mean']:+.4f}, "
          f"Sharpe={stats['LS_Sharpe']:.3f}")

    ic_delta = stats["IC_mean"] - BASELINE_IC_S42
    sharpe_delta = stats["LS_Sharpe"] - BASELINE_SHARPE_S42
    print(f"  Delta vs seed42 baseline:  IC={ic_delta:+.4f}, Sharpe={sharpe_delta:+.3f}")

    # ── Sanity checks ──────────────────────────────────────────────────────
    print("\n  === Sanity Checks ===")
    ic_ok     = 0.005 <= stats["IC_mean"] <= 0.025
    sharpe_ok = 0.3   <= stats["LS_Sharpe"] <= 0.9
    print(f"  IC in [+0.005, +0.025]:    {'PASS' if ic_ok     else 'FAIL *** OUT OF RANGE ***'}")
    print(f"  Sharpe in [+0.3, +0.9]:    {'PASS' if sharpe_ok else 'FAIL *** OUT OF RANGE ***'}")
    if not ic_ok or not sharpe_ok:
        print("  WARN: Sanity check failed — check for data or implementation bug.")

    # ── Save outputs ────────────────────────────────────────────────────────
    summary_row = {
        "label": "rung4_hp_tuned_inner_cv",
        "inner_K": INNER_K,
        "inner_gap_months": INNER_GAP_MONTHS,
        "outer_seed": OUTER_SEED,
        "hp_grid_size": len(HP_GRID),
        **stats,
        "most_common_hidden": int(hp_counts_h.idxmax()),
        "most_common_lr": float(hp_counts_l.idxmax()),
        "most_common_dropout": float(hp_counts_d.idxmax()),
        "baseline_ic_s42": BASELINE_IC_S42,
        "baseline_sharpe_s42": BASELINE_SHARPE_S42,
        "ridge_sharpe": BASELINE_RIDGE,
        "ic_delta_vs_s42": ic_delta,
        "sharpe_delta_vs_s42": sharpe_delta,
    }
    pd.DataFrame([summary_row]).to_csv(
        OUTPUT / "rung4_hp_search_summary.csv", index=False
    )
    print("\n  Saved: output/rung4_hp_search_summary.csv", flush=True)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUTPUT / "rung4_hp_search_diag.csv", index=False)
    print("  Saved: output/rung4_hp_search_diag.csv", flush=True)

    print("\n" + "=" * 100)
    print("DONE.")
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
