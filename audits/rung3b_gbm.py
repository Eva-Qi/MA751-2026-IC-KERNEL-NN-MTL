"""
Rung 3b: Gradient Boosted Trees (XGBoost).

Alternative to Rung 3 GAM. Captures interactions that GAM (strictly additive)
cannot. Supports quantile loss (≈MAE) for fat-tail robustness.

Two variants run:
  - 3b_GBM_MSE_v3     : objective='reg:squarederror'
  - 3b_GBM_Quantile_v3: objective='reg:quantileerror', quantile_alpha=0.5 (median regression)

Uses V3 panel (22+5 features). Same walk-forward harness as other rungs.
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import (
    compute_long_short_sharpe, compute_ic_ir, compute_hit_rate
)

FEATURES = ALL_FEATURE_COLS_V3_WITH_MISS
DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

# Base XGB hyperparameters (shared by both variants)
XGB_KWARGS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=1,
    verbosity=0,
)


def gbm_mse_model(X_tr, y_tr, X_te):
    model = XGBRegressor(objective="reg:squarederror", **XGB_KWARGS)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def gbm_quantile_model(X_tr, y_tr, X_te):
    """Quantile regression at τ=0.5 (median) — minimizes |residual|, fat-tail robust."""
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        **XGB_KWARGS,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def run_walk_forward(df, model_fn, label):
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

        X_tr = np.nan_to_num(df_tr[FEATURES].values, nan=0.0)
        y_tr = df_tr[TARGET_COL].values  # raw y (audit 2026-04-24: no winsorize)
        X_te = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[TARGET_COL].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        y_pred = model_fn(X_tr_s, y_tr, X_te_s)
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


def main():
    print("Loading panel (V3)...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, {len(FEATURES)} features")

    models = [
        ("3b_GBM_MSE_v3",       gbm_mse_model),
        ("3b_GBM_Quantile_v3",  gbm_quantile_model),
    ]

    all_results = {}
    for label, fn in models:
        print(f"\nRunning {label}...")
        monthly = run_walk_forward(df, fn, label)
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

    # Summary
    print("\n" + "=" * 80)
    print("RUNG 3B SUMMARY (V3 DATA — XGBoost)")
    print("=" * 80)
    rows = []
    for label, monthly in all_results.items():
        ic = monthly["IC"].dropna()
        expanded = expand_results(monthly)
        rows.append({
            "Model": label,
            "IC_mean": round(ic.mean(), 4),
            "IC_std": round(ic.std(), 4),
            "IC_IR": round(compute_ic_ir(ic), 4),
            "Hit_Rate": round(compute_hit_rate(ic), 3),
            "LS_Sharpe": round(compute_long_short_sharpe(expanded), 3),
            "n_months": len(ic),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT / "rung3b_gbm_summary.csv", index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
