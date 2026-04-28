"""Run Enhanced MoE 6a (ret-only) with forward-filtered HMM.

Single-variant rerun on branch `moe-hmm-forward-filter` to test whether
the regime.py forward-filter fix changes:
  1. Gate distribution (collapse → meaningful specialization?)
  2. LS Sharpe vs prior baseline (0.481)

Output: output/results_5a_enhanced_moe_filtered.parquet + summary CSV
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATE_COL, STOCK_COL, SECTOR_COL,
    TARGET_COL, RET3M_COL, VOL_COL, FWD_VOL_COL,
    N_REGIMES,
)
from regime import build_market_monthly_features
from regmtl_enhanced import (
    STOCK_FEATURES, GATE_MACRO_COLS, DEFAULT_N_EXPERTS,
    add_interaction_features, walk_forward_evaluate, summarise,
)


def main():
    t0 = time.time()
    print("Loading panel...")
    df = pd.read_parquet("data/master_panel_v2.parquet")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df, ixn_cols = add_interaction_features(df)
    feature_cols = STOCK_FEATURES + ixn_cols
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    df = df.dropna(subset=[TARGET_COL, VOL_COL])
    for c in feature_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in GATE_MACRO_COLS:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    print(f"Rows: {len(df):,}  Months: {df[DATE_COL].nunique()}  "
          f"Tickers: {df[STOCK_COL].nunique()}")

    print("\nBuilding market features for HMM (forward-filter on this branch)...")
    market_features = build_market_monthly_features()
    print(f"Market features: {len(market_features)} months")

    print("\n" + "=" * 70)
    print("Enhanced MoE 5a (ret-only) — forward-filtered HMM")
    print("=" * 70)

    results = walk_forward_evaluate(
        df=df,
        market_features=market_features,
        active_tasks={"ret"},
        feature_cols=feature_cols,
        n_experts=DEFAULT_N_EXPERTS,
        device="cpu",
        verbose=True,
    )

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_5a_enhanced_moe_filtered.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")

    summary = summarise(results, label="5a_filtered")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Gate diagnostic: collapse check
    if {"gate_w_0", "gate_w_1", "gate_w_2"}.issubset(results.columns):
        gate_cols = ["gate_w_0", "gate_w_1", "gate_w_2"]
        per_month = results.groupby(DATE_COL)[gate_cols].mean()
        print("\nGate distribution (mean per test month):")
        print(f"  expert 0: mean={per_month['gate_w_0'].mean():.3f}  "
              f"std={per_month['gate_w_0'].std():.3f}  "
              f"range=[{per_month['gate_w_0'].min():.3f}, "
              f"{per_month['gate_w_0'].max():.3f}]")
        print(f"  expert 1: mean={per_month['gate_w_1'].mean():.3f}  "
              f"std={per_month['gate_w_1'].std():.3f}  "
              f"range=[{per_month['gate_w_1'].min():.3f}, "
              f"{per_month['gate_w_1'].max():.3f}]")
        print(f"  expert 2: mean={per_month['gate_w_2'].mean():.3f}  "
              f"std={per_month['gate_w_2'].std():.3f}  "
              f"range=[{per_month['gate_w_2'].min():.3f}, "
              f"{per_month['gate_w_2'].max():.3f}]")
        max_min_spread = (per_month.max(axis=1) - per_month.min(axis=1)).mean()
        print(f"\n  Avg per-month max-min spread: {max_min_spread:.3f}")
        print(f"  (collapse if ≈ 0; healthy specialization if > 0.10)")

    elapsed_min = (time.time() - t0) / 60
    print(f"\nTotal wall: {elapsed_min:.1f} min")


if __name__ == "__main__":
    main()
