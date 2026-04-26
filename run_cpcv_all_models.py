"""
CPCV driver: apply Combinatorial Purged K-fold CV to all top linear + GBM models.

Models:
  1a OLS            run_rung12_v2.ols_model
  1b IC-Ensemble    run_rung12_v2.ic_ensemble_model
  1c Fama-MacBeth   run_rung12_v2.fama_macbeth_model
  1d Barra          run_rung12_v2.corr_adj_ic_ensemble_model
  2a LASSO          run_rung12_v2.lasso_model
  2b Ridge          run_rung12_v2.ridge_model
  2d ElasticNet     run_rung12_v2.elastic_net_model
  3b XGBoost GKX    inline (depth=1, n=1000, lr=0.01)

Config: N_blocks=6, k_test=2 → C(6,2)=15 paths per model.

Outputs:
  output/cpcv_results.parquet  — long-format: model × path_id × IC_mean × Sharpe × …
  output/cpcv_summary.csv      — per-model: 15-path percentiles

Run:
  nohup python run_cpcv_all_models.py > output/cpcv_log.txt 2>&1 &
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS,
    TARGET_COL, DATE_COL, STOCK_COL,
)
from metrics import compute_long_short_sharpe

# Import all linear model functions from the un-modified V2 runner
from run_rung12_v2 import (
    ols_model,
    ic_ensemble_model,
    fama_macbeth_model,
    corr_adj_ic_ensemble_model,
    lasso_model,
    ridge_model,
    elastic_net_model,
)

from cpcv_harness import cpcv_paths, summarise_paths, paths_to_long_df

DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

FEATURES = ALL_FEATURE_COLS_V3_WITH_MISS

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost GKX adapter — matches model_fn(X_tr, y_tr, X_te, features, **kwargs)
# ─────────────────────────────────────────────────────────────────────────────

def xgb_gkx_model(X_tr, y_tr, X_te, features, **kwargs):
    """3b XGBoost GKX-tuned: depth=1 stumps, n=1000, lr=0.01 (Gu-Kelly-Xiu 2020)."""
    model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=1,
        n_estimators=1000,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    ("1a_OLS",          ols_model),
    ("1b_IC_Ensemble",  ic_ensemble_model),
    ("1c_FamaMacBeth",  fama_macbeth_model),
    ("1d_Barra",        corr_adj_ic_ensemble_model),
    ("2a_LASSO",        lasso_model),
    ("2b_Ridge",        ridge_model),
    ("2d_ElasticNet",   elastic_net_model),
    ("3b_XGB_GKX",      xgb_gkx_model),
]

# CPCV config
N_BLOCKS = 6
K_TEST = 2
EMBARGO_MONTHS = 1
N_PATHS = 15  # C(6,2)

# Walk-forward reference Sharpe values (from rung12_v3_summary.csv)
WF_SHARPE_REF = {
    "1a_OLS":         0.924,
    "1b_IC_Ensemble": 0.844,
    "1c_FamaMacBeth": 0.959,
    "1d_Barra":       0.746,
    "2a_LASSO":       1.000,
    "2b_Ridge":       0.966,
    "2d_ElasticNet":  0.997,
    "3b_XGB_GKX":     None,   # not in V3 summary (GKX-tuned; Rung3b had ~0.7)
}


def load_wf_ref():
    """Try to load walk-forward reference from CSV; fallback to hardcoded dict."""
    ref_path = OUTPUT / "rung12_v3_summary.csv"
    mapping = {}
    if ref_path.exists():
        try:
            df = pd.read_csv(ref_path)
            for _, row in df.iterrows():
                name = row["Model"]
                # Strip _v3 suffix and normalise
                key = (name.replace("_v3", "")
                           .replace("1a_OLS", "1a_OLS")
                           .replace("1b_IC_Ensemble", "1b_IC_Ensemble")
                           .replace("1c_FamaMacBeth", "1c_FamaMacBeth")
                           .replace("1d_Barra", "1d_Barra")
                           .replace("2a_LASSO", "2a_LASSO")
                           .replace("2b_Ridge", "2b_Ridge")
                           .replace("2d_ElasticNet", "2d_ElasticNet"))
                mapping[key] = row["LS_Sharpe"]
        except Exception:
            pass
    # Merge with hardcoded fallback
    for k, v in WF_SHARPE_REF.items():
        if k not in mapping:
            mapping[k] = v
    return mapping


def print_path_report(label, paths, wf_sharpe_ref):
    """Print per-model CPCV summary."""
    sharpes = np.array([p["Sharpe"] for p in paths if pd.notna(p["Sharpe"])])
    ics = np.array([p["IC_mean"] for p in paths if pd.notna(p["IC_mean"])])
    wf_s = wf_sharpe_ref.get(label, None)

    print(f"\n{'='*70}")
    print(f"Model: {label}  (n_valid_paths={len(sharpes)}/{len(paths)})")
    print(f"{'='*70}")

    if len(sharpes) == 0:
        print("  No valid paths!")
        return

    print(f"  15-path Sharpe:")
    print(f"    mean={np.mean(sharpes):+.3f}  std={np.std(sharpes, ddof=1):.3f}")
    print(f"    5th={np.percentile(sharpes, 5):+.3f}  "
          f"25th={np.percentile(sharpes, 25):+.3f}  "
          f"50th={np.percentile(sharpes, 50):+.3f}  "
          f"75th={np.percentile(sharpes, 75):+.3f}  "
          f"95th={np.percentile(sharpes, 95):+.3f}")
    print(f"  P(Sharpe < 0)  = {(sharpes < 0).mean():.1%}")

    if len(ics):
        print(f"  15-path IC:    mean={np.mean(ics):+.4f}  std={np.std(ics, ddof=1):.4f}")

    if wf_s is not None:
        print(f"  Walk-forward Sharpe (reference): {wf_s:.3f}")
        delta = np.mean(sharpes) - wf_s
        print(f"  CPCV mean vs WF delta: {delta:+.3f}")

    # Sanity checks
    issues = []
    if not (-2 < np.mean(sharpes) < 2):
        issues.append(f"mean Sharpe {np.mean(sharpes):.3f} outside [-2,+2]")
    if len(sharpes) > 1 and np.std(sharpes, ddof=1) < 1e-4:
        issues.append("all paths have same Sharpe (degenerate CPCV)")
    if issues:
        for iss in issues:
            print(f"  [WARN] {iss}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 70)
    print("CPCV ALL MODELS — Lopez de Prado ch.12")
    print(f"Config: N_blocks={N_BLOCKS}, k_test={K_TEST}, C({N_BLOCKS},{K_TEST})={N_PATHS} paths, "
          f"embargo={EMBARGO_MONTHS}mo")
    print("=" * 70, flush=True)

    # Load panel
    print("\nLoading panel...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    missing_feats = [c for c in FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing features in panel: {missing_feats}")
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, {len(FEATURES)} features", flush=True)

    wf_ref = load_wf_ref()

    all_long_rows = []
    all_summaries = []

    for model_label, model_fn in MODELS:
        t0 = time.time()
        print(f"\n{'─'*70}")
        print(f"Running CPCV: {model_label}", flush=True)

        try:
            paths = cpcv_paths(
                df=df,
                model_fn=model_fn,
                features=FEATURES,
                N_blocks=N_BLOCKS,
                k_test=K_TEST,
                embargo_months=EMBARGO_MONTHS,
                verbose=True,
            )
        except Exception as e:
            print(f"  ERROR in {model_label}: {e}", flush=True)
            continue

        elapsed = time.time() - t0

        # Per-model report
        print_path_report(model_label, paths, wf_ref)
        print(f"  [time: {elapsed:.1f}s]", flush=True)

        # Accumulate for output files
        long_df = paths_to_long_df(paths, model_label)
        all_long_rows.append(long_df)

        summary = summarise_paths(paths, label=model_label)
        # Attach walk-forward reference for comparison
        summary["WF_Sharpe_ref"] = wf_ref.get(model_label, np.nan)
        all_summaries.append(summary)

    # ─────────────────────────────────────────────────────────────
    # Save outputs
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    if all_long_rows:
        cpcv_long = pd.concat(all_long_rows, ignore_index=True)
        out_parquet = OUTPUT / "cpcv_results.parquet"
        cpcv_long.to_parquet(out_parquet, index=False)
        print(f"  Saved: {out_parquet}  ({len(cpcv_long)} rows)")
    else:
        print("  No results to save (all models failed).")
        cpcv_long = pd.DataFrame()

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        # Reorder columns sensibly
        col_order = [
            "label", "n_paths", "n_valid_paths",
            "Sharpe_mean", "Sharpe_std", "Sharpe_p5", "Sharpe_p25", "Sharpe_p50", "Sharpe_p75", "Sharpe_p95",
            "Prob_neg_Sharpe",
            "IC_mean", "IC_std", "IC_p5", "IC_p95",
            "WF_Sharpe_ref",
        ]
        col_order = [c for c in col_order if c in summary_df.columns]
        summary_df = summary_df[col_order]
        out_csv = OUTPUT / "cpcv_summary.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        # ─────────────────────────────────────────────────────────
        # Final cross-model comparison (rank by 15-path mean Sharpe)
        # ─────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("CROSS-MODEL COMPARISON  (ranked by 15-path mean Sharpe)")
        print("=" * 70)
        ranked = summary_df.sort_values("Sharpe_mean", ascending=False)

        header = f"{'Rank':<5} {'Model':<22} {'Sharpe_mean':>12} {'Sharpe_std':>11} "
        header += f"{'[5th, 95th]':>18} {'P(neg)':>8} {'WF_ref':>8} {'IC_mean':>9}"
        print(header)
        print("-" * 96)
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            wf = f"{row['WF_Sharpe_ref']:.3f}" if pd.notna(row.get("WF_Sharpe_ref")) else "  n/a"
            ic = f"{row['IC_mean']:+.4f}" if pd.notna(row.get("IC_mean")) else "   n/a"
            p5 = row.get("Sharpe_p5", np.nan)
            p95 = row.get("Sharpe_p95", np.nan)
            interval = f"[{p5:+.3f}, {p95:+.3f}]" if pd.notna(p5) and pd.notna(p95) else "       n/a"
            line = (
                f"{rank:<5} {str(row['label']):<22} "
                f"{row['Sharpe_mean']:>+12.3f} "
                f"{row.get('Sharpe_std', np.nan):>+11.3f} "
                f"{interval:>18} "
                f"{row.get('Prob_neg_Sharpe', np.nan):>7.1%} "
                f"{wf:>8} "
                f"{ic:>9}"
            )
            print(line)

    total_elapsed = time.time() - t_start
    print(f"\nTotal time: {total_elapsed/60:.1f} min")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
