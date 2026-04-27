"""
Rung 3a GAM — Combined n_splines hyperparameter sweep.

Single sweep covering grid {3, 4, 5, 6, 7, 8, 9, 10, 15, 20} for cross-sectional
return prediction (IC≈0 low-SNR domain). pygam default is n_splines=10; this
sweep tests whether the default is optimal.

Walk-forward: expanding window, min 60 months train, 1-month purge, 58 test months.
λ auto-tuned per fold via pygam.gridsearch (matches the original rung3_gam.py).

History
-------
Originally split across two scripts:
  rung3a_gam_audit.py       — coarse grid {3, 5, 10, 15, 20}
  rung3a_gam_finetune.py    — fine-grain {4, 6, 7, 8, 9}, after coarse identified
                              n=5 as Pareto-best
These are merged here. Their outputs (rung3a_n_splines_sweep.csv +
rung3a_n_splines_finetune.csv) remain in output/ and are paper-cited.

This combined script produces a single output:
  output/rung3a_n_splines_sweep_full.csv   (all 10 configs)

Re-run only if you need to regenerate. Default invocation: ~60–120 min depending
on n_splines (n=20 is slowest because GCV gridsearch evaluates more knots).

See audits/rung3a_audit_procedure.md for the iterative-audit narrative.
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

# After audits/ relocation, ROOT must point at project root (parent of this file)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

FEATURES = ALL_FEATURE_COLS_V3_WITH_MISS
DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# Full sweep grid (combines original coarse + finetune)
N_SPLINES_GRID = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
LAM_GRID = np.logspace(-3, 3, 11)


def run_gam_nsplines(df, n_splines, label):
    """Full walk-forward with LinearGAM at given n_splines.
    Returns summary dict with IC / Sharpe / Hit / IC-IR / wall_sec."""
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS

    n_features = len(FEATURES)
    fold_records = []
    ic_list = []

    total_folds = len(months) - start
    t_run_start = time.time()

    for fold_idx, i in enumerate(range(start, len(months))):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == months[i]]
        if len(df_te) < 10:
            continue

        X_tr = np.nan_to_num(df_tr[FEATURES].values, nan=0.0)
        y_tr = df_tr[TARGET_COL].values
        X_te = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[TARGET_COL].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Build GAM with n_splines per feature
        gam_terms = s(0, n_splines=n_splines)
        for j in range(1, n_features):
            gam_terms += s(j, n_splines=n_splines)
        gam = LinearGAM(gam_terms)

        gam.gridsearch(X_tr_s, y_tr, lam=LAM_GRID, progress=False)
        y_pred = gam.predict(X_te_s)
        ic = spearmanr(y_te, y_pred).statistic

        fold_records.append({
            DATE_COL: months[i],
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
            "IC": ic,
        })
        if not np.isnan(ic):
            ic_list.append(ic)

        if (fold_idx + 1) % 12 == 0 or fold_idx == total_folds - 1:
            elapsed = time.time() - t_run_start
            print(f"  [{label}] fold {fold_idx+1:3d}/{total_folds} | "
                  f"{months[i].strftime('%Y-%m')} | IC={ic:+.4f} | "
                  f"elapsed={elapsed:.1f}s", flush=True)

    monthly = pd.DataFrame(fold_records)
    expanded_rows = []
    for _, r in monthly.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            expanded_rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t,
                                  "y_pred": yp, "y_true": yt})
    expanded = pd.DataFrame(expanded_rows)

    ic_arr = np.array(ic_list)
    summary = {
        "n_splines": n_splines,
        "label": label,
        "IC_mean": float(ic_arr.mean()) if len(ic_arr) else float("nan"),
        "IC_std":  float(ic_arr.std())  if len(ic_arr) else float("nan"),
        "IC_IR":   compute_ic_ir(pd.Series(ic_arr)) if len(ic_arr) else float("nan"),
        "Hit_Rate": compute_hit_rate(pd.Series(ic_arr)) if len(ic_arr) else float("nan"),
        "LS_Sharpe": compute_long_short_sharpe(expanded) if len(expanded) else float("nan"),
        "n_months": len(ic_arr),
    }
    return summary


def sanity_check(summary):
    flags = []
    ic = summary["IC_mean"]
    sh = summary["LS_Sharpe"]
    if not np.isnan(ic) and (ic < -0.05 or ic > 0.05):
        flags.append(f"IC_mean={ic:+.4f} outside expected [-0.05, +0.05]")
    if not np.isnan(sh) and (sh < -1.0 or sh > 2.0):
        flags.append(f"LS_Sharpe={sh:.2f} outside expected [-1.0, +2.0]")
    return flags


def main():
    print("=" * 80)
    print("RUNG 3A GAM — Full n_splines sweep")
    print(f"Grid: {N_SPLINES_GRID}")
    print("=" * 80, flush=True)

    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"Panel: {len(df):,} rows, {df[DATE_COL].nunique()} months, "
          f"{df[STOCK_COL].nunique()} tickers, {len(FEATURES)} features",
          flush=True)

    all_results = []

    for ns in N_SPLINES_GRID:
        print(f"\n{'=' * 80}\nn_splines = {ns}\n{'=' * 80}", flush=True)
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

    results_df = pd.DataFrame(all_results)
    out_csv = OUTPUT / "rung3a_n_splines_sweep_full.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}", flush=True)

    print("\n" + "=" * 80 + "\nFINAL SUMMARY\n" + "=" * 80)
    cols = ["n_splines", "IC_mean", "IC_IR", "Hit_Rate", "LS_Sharpe", "n_months", "wall_sec"]
    print(results_df[cols].to_string(index=False))

    valid = results_df.dropna(subset=["IC_mean"])
    if len(valid):
        best = valid.loc[valid["LS_Sharpe"].idxmax()]
        print(f"\nBest LS Sharpe: n_splines={int(best['n_splines'])} "
              f"(Sharpe={best['LS_Sharpe']:.3f}, IC={best['IC_mean']:+.4f})")
        if "n_splines" in valid.columns and 10 in valid["n_splines"].values:
            ns10 = valid[valid["n_splines"] == 10].iloc[0]
            d_sharpe = float(best["LS_Sharpe"]) - float(ns10["LS_Sharpe"])
            print(f"vs default n_splines=10: ΔSharpe = {d_sharpe:+.3f}")
            if d_sharpe > 0.05:
                print("HYPERPARAMETER DEFAULT TRAP CONFIRMED — "
                      "tuning n_splines materially improves over default.")


if __name__ == "__main__":
    main()
