"""
Rung 1-2 on V3 data: OLS + IC-weighted ensemble + LASSO + Ridge
Uses the 22-feature V3 panel (master_panel_v2.parquet — already contains V3 features).

V3 vs V2 adds 8 Phase 2 features:
  Liquidity     : Turnover_zscore, AmihudIlliquidity_zscore
  Price patterns: High52W_Proximity_zscore, MaxDailyReturn_zscore, ReturnSkewness_zscore
  Growth        : ImpliedEPSGrowth_zscore, QRevGrowthYoY_zscore
  Coverage      : AnalystCoverageChg_zscore

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
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V2, ALL_FEATURE_COLS_V3,
    TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_monthly_ic, compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

IC_WEIGHT_WINDOW = 12
LASSO_ALPHAS = np.logspace(-4, 1, 50)
RIDGE_ALPHAS = np.logspace(-4, 4, 50)


# ── Model functions ──────────────────────────────────────────────────────────

def ols_model(X_tr, y_tr, X_te, features):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def ic_ensemble_model(X_tr, y_tr, X_te, features):
    """IC-weighted univariate ensemble."""
    n_features = X_tr.shape[1]
    n_recent = min(len(X_tr), IC_WEIGHT_WINDOW * 400)
    X_recent = X_tr[-n_recent:]
    y_recent = y_tr[-n_recent:]

    ics = []
    preds_per_feature = []
    for j in range(n_features):
        model = LinearRegression()
        model.fit(X_tr[:, j:j+1], y_tr)
        pred_j = model.predict(X_te[:, j:j+1])
        preds_per_feature.append(pred_j)

        ic_j = abs(np.corrcoef(X_recent[:, j], y_recent)[0, 1])
        ics.append(ic_j if not np.isnan(ic_j) else 0.0)

    ics = np.array(ics)
    total = ics.sum()
    if total < 1e-8:
        weights = np.ones(n_features) / n_features
    else:
        weights = ics / total

    y_pred = sum(w * p for w, p in zip(weights, preds_per_feature))
    return y_pred


def lasso_model(X_tr, y_tr, X_te, features):
    model = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=10000)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if np.std(pred) < 1e-10:
        fallback = LinearRegression()
        fallback.fit(X_tr, y_tr)
        pred = fallback.predict(X_te)
    return pred


def ridge_model(X_tr, y_tr, X_te, features):
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


# ── Walk-forward harness ─────────────────────────────────────────────────────

def run_walk_forward(df, model_fn, label, features):
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []

    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS
    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]

        if len(df_te) < 10:
            continue

        X_tr = np.nan_to_num(df_tr[features].values, nan=0.0)
        y_tr = df_tr[TARGET_COL].values
        X_te = np.nan_to_num(df_te[features].values, nan=0.0)
        y_te = df_te[TARGET_COL].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        y_pred = model_fn(X_tr_s, y_tr, X_te_s, features)

        ic = spearmanr(y_te, y_pred).statistic
        results.append({
            DATE_COL: test_month,
            "fold": i,
            "IC": ic,
            "n_test": len(df_te),
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
        })

        if i % 12 == 0:
            print(f"  {label} fold {i} | {str(test_month)[:7]} | IC={ic:+.4f}")

    return pd.DataFrame(results)


def expand_results(monthly_df):
    rows = []
    for _, r in monthly_df.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t, "y_pred": yp, "y_true": yt})
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading panel (V3 features)...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Validate all V3 features are present
    missing = [c for c in ALL_FEATURE_COLS_V3 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing V3 features in panel: {missing}")

    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months")
    print(f"  V3 features ({len(ALL_FEATURE_COLS_V3)}): {ALL_FEATURE_COLS_V3}")

    models = [
        ("1a_OLS_v3",         ols_model),
        ("1b_IC_Ensemble_v3", ic_ensemble_model),
        ("2a_LASSO_v3",       lasso_model),
        ("2b_Ridge_v3",       ridge_model),
    ]

    all_results = {}
    for label, fn in models:
        print(f"\nRunning {label}...")
        monthly = run_walk_forward(df, fn, label, ALL_FEATURE_COLS_V3)
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

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RUNG 1-2 SUMMARY (V3 DATA — 22 features)")
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

    # ── V2 vs V3 comparison ──────────────────────────────────────────────────
    v2_path = OUTPUT / "rung12_v2_summary.csv"
    if v2_path.exists():
        summary_v2 = pd.read_csv(v2_path)

        # Align model names for comparison (strip version suffix)
        def base(name):
            return name.replace("_v2", "").replace("_v3", "")

        v2_map = {base(r["Model"]): r for _, r in summary_v2.iterrows()}
        v3_map = {base(r["Model"]): r for _, r in summary_v3.iterrows()}

        print("\n" + "=" * 80)
        print("V2 vs V3 COMPARISON (14 → 22 features)")
        print("=" * 80)
        print(f"{'Model':<22} {'V2 IC':>8} {'V3 IC':>8} {'V2 Sharpe':>10} {'V3 Sharpe':>10} {'ΔSharpe':>9}")
        print("-" * 70)

        for key in sorted(v2_map.keys()):
            if key not in v3_map:
                continue
            r2 = v2_map[key]
            r3 = v3_map[key]
            v2_ic   = r2["IC_mean"]
            v3_ic   = r3["IC_mean"]
            v2_sh   = r2["LS_Sharpe"]
            v3_sh   = r3["LS_Sharpe"]
            delta   = v3_sh - v2_sh
            print(f"{key:<22} {v2_ic:>+8.4f} {v3_ic:>+8.4f} {v2_sh:>10.3f} {v3_sh:>10.3f} {delta:>+9.3f}")

        # Specific benchmarks
        print()
        ic_ens_v2 = v2_map.get("1b_IC_Ensemble", {})
        ic_ens_v3 = v3_map.get("1b_IC_Ensemble", {})
        lasso_v2  = v2_map.get("2a_LASSO", {})
        lasso_v3  = v3_map.get("2a_LASSO", {})

        if len(ic_ens_v2) > 0 and len(ic_ens_v3) > 0:
            v3_ens_sh = float(ic_ens_v3["LS_Sharpe"])
            v2_ens_sh = float(ic_ens_v2["LS_Sharpe"])
            delta_ens = v3_ens_sh - v2_ens_sh
            flag = "BEAT" if delta_ens > 0 else "MISSED"
            print(f"IC-Ensemble V3 vs V2 benchmark (0.863): {flag}  "
                  f"V3={v3_ens_sh:.3f}  V2={v2_ens_sh:.3f}  "
                  f"Δ={delta_ens:+.3f}")

        if len(lasso_v2) > 0 and len(lasso_v3) > 0:
            v3_lasso_sh = float(lasso_v3["LS_Sharpe"])
            v2_lasso_sh = float(lasso_v2["LS_Sharpe"])
            delta_lasso = v3_lasso_sh - v2_lasso_sh
            flag = "BEAT" if delta_lasso > 0 else "MISSED"
            print(f"LASSO V3 vs V2 benchmark (0.805):        {flag}  "
                  f"V3={v3_lasso_sh:.3f}  V2={v2_lasso_sh:.3f}  "
                  f"Δ={delta_lasso:+.3f}")
    else:
        print(f"\nNote: {v2_path} not found — skipping V2 vs V3 comparison.")
        print("Run run_rung12_v2.py first to generate V2 summary.")

    print(f"\nAll outputs saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
