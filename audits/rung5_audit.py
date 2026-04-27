"""
Rung 5 Audit — seed + arch sensitivity per teammate's PDE audit meta-learnings.

STAGE 0 — Naive baselines (rule 7, outsider eyes)
    Zero predictor (IC should ≈ 0) + Ridge α=1 linear baseline.
    Grounds MLP/MTL claims against 2 achievable floors.

STAGE 1 — rung4 MLP seed sensitivity (rule 3)
    5 seeds × 58 folds. If IC sign reverses across seeds → MLP result is
    noise in a low-SNR task. Mean/std ratio < 1 → indistinguishable from init.

STAGE 2 — MLP hidden_dim sweep (rule 3, rule 7)
    hidden ∈ {8, 16, 32, 64} at seed=42.
    If smaller arch performs equal or better, 3073-param default is
    over-parameterized for IC~0.005 task — supports finding that
    complexity DIDN'T help in Rung 4/5.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

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

# MLP class and train_mlp_fold removed — imported from src.models.mlp_audit (Category C + D consolidation)
# dropout=0.10 default in canonical version produces IDENTICAL results to original hardcoded Dropout(0.10)


def _wf_common_prep(df):
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS
    return df, months, start


def _aggregate(monthly_df, label, extra=None):
    ic_series = monthly_df["IC"].dropna()
    rows = []
    for _, r in monthly_df.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t, "y_pred": yp, "y_true": yt})
    expanded = pd.DataFrame(rows)
    sharpe = compute_long_short_sharpe(expanded)
    out = {
        "label": label,
        "IC_mean": float(ic_series.mean()),
        "IC_std": float(ic_series.std()),
        "IC_IR": float(compute_ic_ir(ic_series)),
        "Hit_Rate": float(compute_hit_rate(ic_series)),
        "LS_Sharpe": float(sharpe),
        "n_months": int(len(ic_series)),
    }
    if extra:
        out.update(extra)
    return out


def walk_forward_mlp(df, hidden, seed, label):
    df, months, start = _wf_common_prep(df)
    results = []
    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]
        if len(df_te) < 10:
            continue

        tr_months = sorted(df_tr[DATE_COL].unique())
        n_val_months = max(1, int(len(tr_months) * 0.1))
        val_months = set(tr_months[-n_val_months:])
        df_tr_main = df_tr[~df_tr[DATE_COL].isin(val_months)]
        df_val = df_tr[df_tr[DATE_COL].isin(val_months)]
        if len(df_val) < 10 or len(df_tr_main) < 100:
            continue

        X_tr = np.nan_to_num(df_tr_main[FEATURES].values, nan=0.0)
        y_tr = df_tr_main[TARGET_COL].values
        X_val = np.nan_to_num(df_val[FEATURES].values, nan=0.0)
        y_val = df_val[TARGET_COL].values
        X_te = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[TARGET_COL].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)

        fold_seed = seed * 1000 + i
        y_pred = train_mlp_fold(X_tr_s, y_tr, X_val_s, y_val, X_te_s,
                                hidden=hidden, seed=fold_seed)
        ic = spearmanr(y_te, y_pred).statistic

        results.append({
            DATE_COL: test_month, "IC": ic,
            "pred_std": float(np.std(y_pred)),
            "pred_mean": float(np.mean(y_pred)),
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
        })

    monthly = pd.DataFrame(results)
    extra = {
        "hidden": hidden, "seed": seed,
        "pred_std_mean": float(monthly["pred_std"].mean()) if len(monthly) else 0.0,
    }
    return _aggregate(monthly, label, extra=extra)


def walk_forward_ridge(df, alpha, label):
    df, months, start = _wf_common_prep(df)
    results = []
    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]
        if len(df_te) < 10:
            continue
        X_tr = np.nan_to_num(df_tr[FEATURES].values, nan=0.0)
        y_tr = df_tr[TARGET_COL].values
        X_te = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[TARGET_COL].values
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        m = Ridge(alpha=alpha)
        m.fit(X_tr_s, y_tr)
        y_pred = m.predict(X_te_s)
        ic = spearmanr(y_te, y_pred).statistic
        results.append({
            DATE_COL: test_month, "IC": ic,
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
        })
    return _aggregate(pd.DataFrame(results), label)


def walk_forward_zero(df, label):
    df, months, start = _wf_common_prep(df)
    results = []
    rng = np.random.default_rng(0)
    for i, test_month in enumerate(months[start:], start=start):
        df_te = df[df[DATE_COL] == test_month]
        if len(df_te) < 10:
            continue
        y_te = df_te[TARGET_COL].values
        y_pred = rng.standard_normal(len(y_te)) * 1e-9
        ic = spearmanr(y_te, y_pred).statistic
        results.append({
            DATE_COL: test_month, "IC": ic,
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
        })
    return _aggregate(pd.DataFrame(results), label)


def main():
    print("=" * 80)
    print("RUNG 5 AUDIT — Baselines + Seed + Hidden Dim")
    print("=" * 80, flush=True)

    print("Loading V3 panel...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, {len(FEATURES)} features", flush=True)

    # ─────────────────────────────────────────────────────────────
    # STAGE 0: Naive baselines
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 0: Naive baselines (zero predictor + ridge linear)")
    print("=" * 80, flush=True)

    t0 = time.time()
    zero_res = walk_forward_zero(df, "zero_predictor")
    print(f"  zero predictor [{time.time()-t0:.1f}s]  "
          f"IC={zero_res['IC_mean']:+.4f}, Sharpe={zero_res['LS_Sharpe']:.3f}, "
          f"Hit={zero_res['Hit_Rate']:.1%}", flush=True)

    t0 = time.time()
    ridge_res = walk_forward_ridge(df, 1.0, "ridge_alpha1.0")
    print(f"  ridge α=1    [{time.time()-t0:.1f}s]  "
          f"IC={ridge_res['IC_mean']:+.4f}, Sharpe={ridge_res['LS_Sharpe']:.3f}, "
          f"Hit={ridge_res['Hit_Rate']:.1%}", flush=True)

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: rung4 MLP seed sensitivity (hidden=64)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 1: rung4 MLP seed sensitivity (hidden=64, 5 seeds)")
    print("=" * 80, flush=True)
    seeds = [0, 1, 2, 42, 123]
    seed_results = []
    for seed in seeds:
        t0 = time.time()
        res = walk_forward_mlp(df, hidden=64, seed=seed, label=f"mlp_h64_s{seed}")
        elapsed = time.time() - t0
        print(f"  seed={seed} [{elapsed:.1f}s]  "
              f"IC={res['IC_mean']:+.4f}, Sharpe={res['LS_Sharpe']:.3f}, "
              f"pred_std={res['pred_std_mean']:.5f}, Hit={res['Hit_Rate']:.1%}", flush=True)
        seed_results.append(res)

    seed_df = pd.DataFrame(seed_results)
    ic_vals = seed_df["IC_mean"].values
    n_pos = int((ic_vals > 0).sum())
    n_neg = int((ic_vals < 0).sum())
    ic_range = ic_vals.max() - ic_vals.min()
    ic_mean = ic_vals.mean()
    ic_std = ic_vals.std()
    print(f"\n  [SEED SIGN]  positive IC: {n_pos}/{len(ic_vals)}, negative: {n_neg}/{len(ic_vals)}")
    print(f"  [SEED RANGE] {ic_vals.min():+.4f} to {ic_vals.max():+.4f}  (Δ={ic_range:.4f})")
    print(f"  [SEED STATS] mean IC = {ic_mean:+.4f}, std across seeds = {ic_std:.4f}")
    snr = abs(ic_mean) / max(ic_std, 1e-8)
    print(f"  [SNR]        |mean|/std = {snr:.2f}")
    if n_pos > 0 and n_neg > 0:
        print("  🚨 SIGN REVERSAL ACROSS SEEDS — MLP IC is init-noise, not signal")
    elif snr < 1.0:
        print("  🚨 |mean|/std < 1 — effect indistinguishable from init noise")
    elif snr < 2.0:
        print("  ⚠️  |mean|/std in [1,2) — weak signal, barely above init noise")
    else:
        print("  ✓ sign stable + |mean|/std ≥ 2 — signal distinguishable from init noise")
    seed_df.to_csv(OUTPUT / "rung5_seed_sensitivity.csv", index=False)

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Hidden dim sweep (seed=42 fixed)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 2: MLP hidden_dim sweep (seed=42, hidden ∈ {8, 16, 32, 64})")
    print("=" * 80, flush=True)
    hd_results = []
    for hd in [8, 16, 32, 64]:
        t0 = time.time()
        res = walk_forward_mlp(df, hidden=hd, seed=42, label=f"mlp_h{hd}_s42")
        elapsed = time.time() - t0
        print(f"  hidden={hd:>2} [{elapsed:.1f}s]  "
              f"IC={res['IC_mean']:+.4f}, Sharpe={res['LS_Sharpe']:.3f}, "
              f"pred_std={res['pred_std_mean']:.5f}", flush=True)
        hd_results.append(res)

    hd_df = pd.DataFrame(hd_results)
    print("\n  === Hidden dim summary ===")
    print(hd_df[["hidden", "IC_mean", "IC_IR", "LS_Sharpe", "pred_std_mean"]].to_string(index=False))
    hd_df.to_csv(OUTPUT / "rung5_hidden_sweep.csv", index=False)

    hd_range = hd_df["IC_mean"].max() - hd_df["IC_mean"].min()
    best_hd = hd_df.loc[hd_df["IC_mean"].idxmax()]
    print(f"\n  Best hidden: {int(best_hd['hidden'])} (IC={best_hd['IC_mean']:+.4f}, Sharpe={best_hd['LS_Sharpe']:.3f})")
    print(f"  Hidden-dim IC range: {hd_range:.4f}")
    if hd_df.loc[hd_df["hidden"] == 8, "IC_mean"].values[0] >= hd_df.loc[hd_df["hidden"] == 64, "IC_mean"].values[0]:
        print("  ⚠️  hidden=8 ≥ hidden=64 — our 64-32 default is over-parameterized")

    # ─────────────────────────────────────────────────────────────
    # Final verdict
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"  Zero predictor: IC={zero_res['IC_mean']:+.4f}, Sharpe={zero_res['LS_Sharpe']:.3f}")
    print(f"  Ridge α=1:      IC={ridge_res['IC_mean']:+.4f}, Sharpe={ridge_res['LS_Sharpe']:.3f}")
    print(f"  MLP h64 seeds:  mean IC={ic_mean:+.4f} ± {ic_std:.4f} (n=5, SNR={snr:.2f})")
    print(f"  MLP h sweep:    IC range {hd_df['IC_mean'].min():+.4f} to {hd_df['IC_mean'].max():+.4f} "
          f"(hidden {int(hd_df.loc[hd_df['IC_mean'].idxmin(),'hidden'])} → "
          f"{int(hd_df.loc[hd_df['IC_mean'].idxmax(),'hidden'])})")
    print("=" * 80)
    print("Saved: output/rung5_seed_sensitivity.csv, output/rung5_hidden_sweep.csv")


if __name__ == "__main__":
    main()
