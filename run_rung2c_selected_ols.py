"""
Rung 2c: OLS on feature-selected subsets.

Three variants:
  2c_OLS_LASSO3 : top-3 by LASSO selection frequency (>=50% of folds)
  2c_OLS_LASSO5 : top-5 by LASSO selection frequency
  2c_OLS_FMB4   : features flagged significant by teammate's Fama-MacBeth

Uses the same walk-forward harness as run_rung12_v2.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from config import (
    TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")

FEATURE_SETS = {
    "2c_OLS_LASSO3": [
        "IVOL_zscore",
        "GrossProfitability_zscore",
        "Beta_zscore",
    ],
    "2c_OLS_LASSO5": [
        "IVOL_zscore",
        "GrossProfitability_zscore",
        "Beta_zscore",
        "ShortInterestRatio_zscore",
        "Momentum12_1_zscore",
    ],
    "2c_OLS_FMB4": [
        "NetDebtEBITDA_zscore",
        "GrossProfitability_zscore",
        "AnalystRevision_zscore",
        "RevisionBreadth_zscore",
    ],
}


def walk_forward(df, features):
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

        model = LinearRegression().fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        ic = spearmanr(y_te, y_pred).statistic
        results.append({
            DATE_COL: test_month,
            "IC": ic,
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
        })
    return pd.DataFrame(results)


def expand(monthly):
    rows = []
    for _, r in monthly.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t, "y_pred": yp, "y_true": yt})
    return pd.DataFrame(rows)


def ls_hit_rate(expanded):
    from scripts.compute_ls_hit_rate import ls_monthly
    m = ls_monthly(expanded)
    return (m["ls_ret"] > 0).mean(), len(m)


def main():
    print("Loading V2 panel...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    summary = []
    for label, feats in FEATURE_SETS.items():
        print(f"\n{label}: {feats}")
        monthly = walk_forward(df, feats)
        ic = monthly["IC"].dropna()
        expanded = expand(monthly)
        sharpe = compute_long_short_sharpe(expanded)
        ls_hit, n_ls = ls_hit_rate(expanded)

        row = {
            "Model": label,
            "n_features": len(feats),
            "features": ", ".join(f.replace("_zscore", "") for f in feats),
            "IC_mean": round(ic.mean(), 4),
            "IC_std": round(ic.std(), 4),
            "IC_IR": round(compute_ic_ir(ic), 4),
            "Hit_IC": round(compute_hit_rate(ic), 3),
            "LS_Sharpe": round(sharpe, 3),
            "LS_Hit": round(ls_hit, 3),
            "n_months": len(ic),
        }
        summary.append(row)

        expanded.to_parquet(OUTPUT / f"results_{label}.parquet", index=False)
        print(f"  IC={row['IC_mean']:+.4f}  IC_IR={row['IC_IR']:+.4f}  "
              f"Hit_IC={row['Hit_IC']:.1%}  Sharpe={row['LS_Sharpe']:.3f}  "
              f"LS_Hit={row['LS_Hit']:.1%}")

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(OUTPUT / "rung2c_selected_ols_summary.csv", index=False)
    print("\n" + "=" * 80)
    print("RUNG 2c SUMMARY (OLS on selected features)")
    print("=" * 80)
    print(df_sum.drop(columns=["features"]).to_string(index=False))
    print(f"\nSaved -> {OUTPUT / 'rung2c_selected_ols_summary.csv'}")


if __name__ == "__main__":
    main()
