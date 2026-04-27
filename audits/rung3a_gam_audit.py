"""
Rung 3a GAM Audit — n_splines hyperparameter sensitivity sweep.

Tests whether pygam.LinearGAM n_splines=10 default is optimal for
cross-sectional return prediction (IC≈0 low-SNR domain).

Sweep: n_splines ∈ [3, 5, 10, 15, 20]
Walk-forward: expanding window, min 60 months train, 1-month purge, 58 test months.
λ auto-tuned per fold via gridsearch (matches rung3_gam.py exactly).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

FEATURES = ALL_FEATURE_COLS_V3_WITH_MISS
DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

LAM_GRID = np.logspace(-3, 3, 11)


def run_gam_nsplines(df, n_splines, label):
    """
    Full walk-forward with LinearGAM using given n_splines.
    Returns summary dict + list of per-fold IC dicts.
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS

    n_features = len(FEATURES)
    fold_records = []    # {date, y_true list, y_pred list, tickers, ic}
    ic_list = []

    total_folds = len(months) - start
    t_run_start = time.time()

    for fold_idx, i in enumerate(range(start, len(months))):
        test_month = months[i]
        train_end  = months[i - DEFAULT_PURGE_MONTHS - 1]

        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]

        if len(df_te) < 10:
            continue

        # Features — NaN → 0 (mirrors rung3_gam.py)
        X_tr = np.nan_to_num(df_tr[FEATURES].values.astype(np.float64), nan=0.0)
        y_tr = df_tr[TARGET_COL].values.astype(np.float64)
        X_te = np.nan_to_num(df_te[FEATURES].values.astype(np.float64), nan=0.0)
        y_te = df_te[TARGET_COL].values.astype(np.float64)

        # StandardScaler fit on train only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        # Build GAM terms
        terms = s(0, n_splines=n_splines)
        for j in range(1, n_features):
            terms += s(j, n_splines=n_splines)

        gam = LinearGAM(terms)
        gam.gridsearch(X_tr_s, y_tr, lam=LAM_GRID, progress=False)

        y_pred = gam.predict(X_te_s)
        ic = float(spearmanr(y_te, y_pred).statistic)
        ic_list.append(ic)

        fold_records.append({
            DATE_COL:   test_month,
            STOCK_COL:  df_te[STOCK_COL].values.tolist(),
            "y_true":   y_te.tolist(),
            "y_pred":   y_pred.tolist(),
            "ic":       ic,
        })

        elapsed = time.time() - t_run_start
        print(
            f"  [{label}] fold {fold_idx+1:3d}/{total_folds} | "
            f"{str(test_month)[:7]} | n_train={len(df_tr):6,d} | "
            f"n_test={len(df_te):4d} | IC={ic:+.4f} | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    # Build expanded frame for Sharpe
    rows = []
    for r in fold_records:
        for t, yp, yt in zip(r[STOCK_COL], r["y_pred"], r["y_true"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t,
                         "y_pred": yp, "y_true": yt})
    expanded = pd.DataFrame(rows)

    ic_series = pd.Series(ic_list, dtype=float).dropna()
    sharpe    = compute_long_short_sharpe(expanded) if len(expanded) else float("nan")

    summary = {
        "n_splines":  n_splines,
        "label":      label,
        "IC_mean":    round(float(ic_series.mean()), 4) if len(ic_series) else float("nan"),
        "IC_std":     round(float(ic_series.std(ddof=1)), 4) if len(ic_series) > 1 else float("nan"),
        "IC_IR":      round(compute_ic_ir(ic_series), 4),
        "Hit_Rate":   round(compute_hit_rate(ic_series), 3),
        "LS_Sharpe":  round(sharpe, 3),
        "n_months":   len(ic_series),
    }
    return summary


def sanity_check(summary):
    """Flag values outside expected ranges."""
    ic = summary["IC_mean"]
    sharpe = summary["LS_Sharpe"]
    flags = []
    if not (float("nan") == ic) and not (-0.02 <= ic <= 0.05):
        flags.append(f"IC={ic:.4f} OUTSIDE [-0.02, +0.05] RANGE — POSSIBLE BUG")
    if not (float("nan") == sharpe) and not (-0.5 <= sharpe <= 1.5):
        flags.append(f"Sharpe={sharpe:.3f} OUTSIDE [-0.5, +1.5] RANGE — POSSIBLE BUG")
    return flags


def main():
    print("=" * 80)
    print("RUNG 3A GAM AUDIT — n_splines hyperparameter sensitivity sweep")
    print("=" * 80, flush=True)

    print(f"Loading panel: {DATA}")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, "
          f"{df[STOCK_COL].nunique()} tickers, {len(FEATURES)} features", flush=True)

    n_splines_grid = [3, 5, 10, 15, 20]
    all_results = []

    for ns in n_splines_grid:
        print(f"\n{'=' * 80}")
        print(f"n_splines = {ns}")
        print(f"{'=' * 80}", flush=True)

        t0 = time.time()
        label = f"gam_ns{ns}"
        try:
            summary = run_gam_nsplines(df, n_splines=ns, label=label)
            elapsed = time.time() - t0
            summary["wall_sec"] = round(elapsed, 1)

            flags = sanity_check(summary)
            summary["flags"] = " | ".join(flags) if flags else "OK"

            print(f"\n  DONE n_splines={ns}: IC={summary['IC_mean']:+.4f}, "
                  f"IC_IR={summary['IC_IR']:.4f}, Sharpe={summary['LS_Sharpe']:.3f}, "
                  f"Hit={summary['Hit_Rate']:.1%}, elapsed={elapsed:.1f}s", flush=True)
            if flags:
                for f in flags:
                    print(f"  *** SANITY FLAG: {f} ***", flush=True)

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR for n_splines={ns}: {exc}", flush=True)
            summary = {
                "n_splines": ns, "label": label,
                "IC_mean": float("nan"), "IC_std": float("nan"),
                "IC_IR": float("nan"), "Hit_Rate": float("nan"),
                "LS_Sharpe": float("nan"), "n_months": 0,
                "wall_sec": round(elapsed, 1), "flags": f"ERROR: {exc}",
            }

        all_results.append(summary)

    # ── Save CSV ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    out_csv = OUTPUT / "rung3a_n_splines_sweep.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved results -> {out_csv}", flush=True)

    # ── Final summary table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)
    cols = ["n_splines", "IC_mean", "IC_std", "IC_IR", "Hit_Rate", "LS_Sharpe", "n_months", "wall_sec"]
    print(results_df[cols].to_string(index=False))

    # ── Verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    valid = results_df.dropna(subset=["IC_mean"])
    if len(valid) == 0:
        print("  No valid results — all configurations failed.")
    else:
        best_row  = valid.loc[valid["IC_mean"].idxmax()]
        ns10_row  = valid[valid["n_splines"] == 10]

        print(f"  Best n_splines by IC_mean:  {int(best_row['n_splines'])} "
              f"(IC={best_row['IC_mean']:+.4f}, Sharpe={best_row['LS_Sharpe']:.3f})")

        if len(ns10_row):
            ns10 = ns10_row.iloc[0]
            print(f"  Default n_splines=10:       IC={ns10['IC_mean']:+.4f}, "
                  f"Sharpe={ns10['LS_Sharpe']:.3f}")
            delta_ic = float(best_row["IC_mean"]) - float(ns10["IC_mean"])
            if int(best_row["n_splines"]) == 10:
                print("  n_splines=10 IS optimal — no gain from changing knot count.")
            elif abs(delta_ic) < 0.003:
                print(f"  n_splines={int(best_row['n_splines'])} marginally beats default "
                      f"(ΔIC={delta_ic:+.4f}) — within noise range, no actionable change.")
            else:
                print(f"  n_splines={int(best_row['n_splines'])} materially beats default "
                      f"(ΔIC={delta_ic:+.4f}) — HYPERPARAMETER DEFAULT TRAP CONFIRMED.")

        ic_vals = valid["IC_mean"].values
        n_pos = (ic_vals > 0).sum()
        n_neg = (ic_vals < 0).sum()
        print(f"\n  Sign check: {n_pos}/{len(valid)} positive IC, {n_neg}/{len(valid)} negative IC")
        if n_pos > 0 and n_neg > 0:
            print("  SIGN REVERSAL detected — IC sign is unstable across n_splines.")
        else:
            print("  No sign reversal — IC sign is stable across all tested n_splines values.")

        ic_range = float(ic_vals.max() - ic_vals.min())
        print(f"  IC range: {ic_vals.min():+.4f} to {ic_vals.max():+.4f} (Δ={ic_range:.4f})")
        if ic_range < 0.005:
            print("  Small IC range — GAM is robust to n_splines choice in this regime.")
        elif ic_range < 0.01:
            print("  Moderate IC range — some sensitivity but not dramatic.")
        else:
            print("  Large IC range — n_splines materially affects performance.")

    print("=" * 80)
    print(f"Outputs: {out_csv}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
