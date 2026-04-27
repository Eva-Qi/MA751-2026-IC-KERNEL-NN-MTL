"""
Rung 1-2 on V3 data: OLS, IC-Ensemble, Fama-MacBeth, Barra-adjusted IC,
LASSO, Ridge, ElasticNet, Adaptive LASSO.

Uses the V3 panel (ALL_FEATURE_COLS_V3_WITH_MISS = 22 features + 5 flags = 27).

V3 vs V2 adds 8 Phase 2 features:
  Liquidity     : Turnover_zscore, AmihudIlliquidity_zscore
  Price patterns: High52W_Proximity_zscore, MaxDailyReturn_zscore, ReturnSkewness_zscore
  Growth        : ImpliedEPSGrowth_zscore, QRevGrowthYoY_zscore
  Coverage      : AnalystCoverageChg_zscore

All audit fixes (TimeSeriesSplit / winsorize / LASSO α / ElasticNet grid / Ridge
Spearman scoring) inherited from run_rung12_v2.py via import.

Run from project root:
    python scripts/rung12_v3.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V2, ALL_FEATURE_COLS_V3, ALL_FEATURE_COLS_V3_WITH_MISS,
    TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
ALL_FEATURE_COLS_V3 = ALL_FEATURE_COLS_V3_WITH_MISS  # audit fix

from metrics import compute_monthly_ic, compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

# Import all model functions + harness from V2 runner — single source of truth
from run_rung12_v2 import (
    run_walk_forward, expand_results,
    ols_model, ic_ensemble_model,
    fama_macbeth_model, corr_adj_ic_ensemble_model,
    lasso_model, ridge_model, elastic_net_model,
    adaptive_lasso_model,
    _ELASTIC_NET_DIAG, _FM_DIAG,
)

DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


def main():
    print("Loading panel (V3 features)...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    missing = [c for c in ALL_FEATURE_COLS_V3 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing V3 features in panel: {missing}")

    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months")
    print(f"  V3 features ({len(ALL_FEATURE_COLS_V3)}): {ALL_FEATURE_COLS_V3}")

    # Clear diagnostics buffers (V2 run may have populated them if run in same process)
    _ELASTIC_NET_DIAG.clear()
    _FM_DIAG.clear()

    models = [
        ("1a_OLS_v3",             ols_model),
        ("1b_IC_Ensemble_v3",     ic_ensemble_model),
        ("1c_FamaMacBeth_v3",     fama_macbeth_model),
        ("1d_Barra_v3",           corr_adj_ic_ensemble_model),  # True Barra
        ("2a_LASSO_v3",           lasso_model),                  # [unreliable_SNR]
        ("2b_Ridge_v3",           ridge_model),
        ("2d_ElasticNet_v3",      elastic_net_model),            # [unreliable_SNR]
        ("2e_AdaptiveLASSO_v3",   adaptive_lasso_model),         # [unreliable_SNR]
    ]

    all_results = {}
    for label, fn in models:
        print(f"\nRunning {label}...")
        monthly = run_walk_forward(df, fn, label, features=ALL_FEATURE_COLS_V3)
        all_results[label] = monthly

        ic_series = monthly["IC"].dropna()
        expanded = expand_results(monthly)
        sharpe = compute_long_short_sharpe(expanded)

        print(f"  {label}: IC={ic_series.mean():+.4f} (std={ic_series.std():.4f}), "
              f"IC_IR={compute_ic_ir(ic_series):.4f}, "
              f"Hit={compute_hit_rate(ic_series):.1%}, "
              f"Sharpe={sharpe:.3f}, "
              f"n_months={len(ic_series)}")

        expanded.to_parquet(OUTPUT / f"results_{label}.parquet", index=False)
        monthly.to_csv(OUTPUT / f"monthly_{label}.csv", index=False)

    # Summary table
    print("\n" + "=" * 80)
    print(f"RUNG 1-2 SUMMARY (V3 DATA — {len(ALL_FEATURE_COLS_V3)} features)")
    print("=" * 80)

    rows_v3 = []
    for label, monthly in all_results.items():
        ic = monthly["IC"].dropna()
        expanded = expand_results(monthly)
        rows_v3.append({
            "Model": label,
            "IC_mean": round(ic.mean(), 4),
            "IC_std": round(ic.std(), 4),
            "IC_IR": round(compute_ic_ir(ic), 4),
            "Hit_Rate": round(compute_hit_rate(ic), 3),
            "LS_Sharpe": round(compute_long_short_sharpe(expanded), 3),
            "n_months": len(ic),
        })

    summary_v3 = pd.DataFrame(rows_v3)
    print(summary_v3.to_string(index=False))
    summary_v3.to_csv(OUTPUT / "rung12_v3_summary.csv", index=False)

    # V2 vs V3 comparison
    v2_path = OUTPUT / "rung12_v2_summary.csv"
    if v2_path.exists():
        summary_v2 = pd.read_csv(v2_path)
        def base(name):
            return name.replace("_v2", "").replace("_v3", "")
        v2_map = {base(r["Model"]): r for _, r in summary_v2.iterrows()}
        v3_map = {base(r["Model"]): r for _, r in summary_v3.iterrows()}

        print("\n" + "=" * 80)
        print("V2 vs V3 COMPARISON")
        print("=" * 80)
        print(f"{'Model':<22} {'V2 IC':>8} {'V3 IC':>8} {'V2 Sharpe':>10} {'V3 Sharpe':>10} {'ΔSharpe':>9}")
        print("-" * 70)
        for key in sorted(v2_map.keys()):
            if key not in v3_map:
                continue
            r2 = v2_map[key]; r3 = v3_map[key]
            print(f"{key:<22} {r2['IC_mean']:>+8.4f} {r3['IC_mean']:>+8.4f} "
                  f"{r2['LS_Sharpe']:>10.3f} {r3['LS_Sharpe']:>10.3f} "
                  f"{r3['LS_Sharpe']-r2['LS_Sharpe']:>+9.3f}")

    # Save ElasticNet + FM diagnostics (V3)
    if _ELASTIC_NET_DIAG:
        diag_df = pd.DataFrame(_ELASTIC_NET_DIAG)
        diag_df.to_csv(OUTPUT / "elastic_net_diag_v3.csv", index=False)
        print(f"\nElasticNet V3 diagnostics ({len(diag_df)} folds):")
        print(f"  l1_ratio mean={diag_df['l1_ratio'].mean():.2f}")
        print(f"  alpha mean={diag_df['alpha'].mean():.4f}")
        print(f"  n_nonzero mean={diag_df['n_nonzero'].mean():.1f}")

    if _FM_DIAG:
        fm_df = pd.DataFrame(_FM_DIAG)
        fm_df.to_csv(OUTPUT / "fama_macbeth_diag_v3.csv", index=False)
        print(f"\nFama-MacBeth V3 diagnostics ({len(fm_df)} folds):")
        print(f"  avg monthly regressions per fold: {fm_df['n_monthly_regressions'].mean():.1f}")
        print(f"  avg beta L2 norm: {fm_df['beta_mean_L2'].mean():.4f}")

    print(f"\nAll outputs saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
