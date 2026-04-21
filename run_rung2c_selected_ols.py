"""
Rung 2c: OLS on feature-selected subsets.

Four variants:
  2c_OLS_LASSO3    : top-3 by LASSO selection frequency (>=50% of folds)
                     [NOTE: LOOK-AHEAD — feature set chosen using full-sample data]
  2c_OLS_LASSO5    : top-5 by LASSO selection frequency
                     [NOTE: LOOK-AHEAD — feature set chosen using full-sample data]
  2c_OLS_FMB4      : features flagged significant by teammate's Fama-MacBeth
  2c_OLS_LASSO_WF  : walk-forward LASSO selection — NO look-ahead, feature set
                     re-selected inside each fold's training window

Uses the same walk-forward harness as run_rung12_v2.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler

from config import (
    TARGET_COL, DATE_COL, STOCK_COL,
    ALL_FEATURE_COLS_V2,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")

# ── NOTE: LASSO3 and LASSO5 feature sets below have a look-ahead bias. ──────
# The features were selected by computing LASSO selection frequency across the
# FULL walk-forward sample (including test folds). A clean alternative is the
# 2c_OLS_LASSO_WF variant, which re-selects features inside each fold's
# training window only.
FEATURE_SETS = {
    "2c_OLS_LASSO3": [  # LOOK-AHEAD: full-sample selection frequency
        "IVOL_zscore",
        "GrossProfitability_zscore",
        "Beta_zscore",
    ],
    "2c_OLS_LASSO5": [  # LOOK-AHEAD: full-sample selection frequency
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


def walk_forward_lasso_selected_ols(df, min_k=3, max_k=5):
    """
    Fold-by-fold relaxed LASSO: select features inside each fold's training window.
    No look-ahead — feature set varies per fold.

    For each test month:
      1. Fit LassoCV on training data (all ALL_FEATURE_COLS_V2 candidates).
      2. Identify features with non-zero coefficients.
      3. If fewer than min_k features survive, fall back to top-min_k by |coef|.
      4. Fit OLS on the selected features only.
      5. Predict on test month; record IC and predictions.
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []
    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS

    candidates = [c for c in ALL_FEATURE_COLS_V2 if c in df.columns]

    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]
        if len(df_te) < 10:
            continue

        X_tr_raw = np.nan_to_num(df_tr[candidates].values, nan=0.0)
        y_tr = df_tr[TARGET_COL].values
        X_te_raw = np.nan_to_num(df_te[candidates].values, nan=0.0)
        y_te = df_te[TARGET_COL].values

        # Scale on training data only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw)
        X_te_s = scaler.transform(X_te_raw)

        # Step 1: LassoCV to select features (cv=5, max_iter=5000)
        lasso = LassoCV(cv=5, max_iter=5000, n_jobs=1, random_state=42)
        lasso.fit(X_tr_s, y_tr)
        coefs = lasso.coef_

        # Step 2: non-zero features
        nonzero_mask = coefs != 0.0
        selected_idx = np.where(nonzero_mask)[0].tolist()

        # Step 3: fallback to top-min_k by |coef| if too few survive
        if len(selected_idx) < min_k:
            selected_idx = np.argsort(np.abs(coefs))[::-1][:min_k].tolist()

        # Optionally cap at max_k
        if len(selected_idx) > max_k:
            # keep the max_k with largest |coef| among the selected
            sel_coefs = np.abs(coefs[selected_idx])
            top_pos = np.argsort(sel_coefs)[::-1][:max_k]
            selected_idx = [selected_idx[p] for p in top_pos]

        selected_features = [candidates[j] for j in selected_idx]

        # Step 4: OLS on selected features only
        X_tr_sel = X_tr_s[:, selected_idx]
        X_te_sel = X_te_s[:, selected_idx]
        model = LinearRegression().fit(X_tr_sel, y_tr)
        y_pred = model.predict(X_te_sel)

        # Step 5: IC
        ic = spearmanr(y_te, y_pred).statistic
        results.append({
            DATE_COL: test_month,
            "IC": ic,
            "n_selected": len(selected_idx),
            "selected_features": selected_features,
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

    # ── Hardcoded-feature variants (LASSO3/LASSO5/FMB4 have look-ahead) ──
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

    # ── Walk-forward LASSO selection — no look-ahead ──────────────────────
    label_wf = "2c_OLS_LASSO_WF"
    print(f"\n{label_wf}: fold-by-fold LassoCV feature selection (no look-ahead)")
    monthly_wf = walk_forward_lasso_selected_ols(df, min_k=3, max_k=5)
    ic_wf = monthly_wf["IC"].dropna()
    expanded_wf = expand(monthly_wf)
    sharpe_wf = compute_long_short_sharpe(expanded_wf)
    ls_hit_wf, n_ls_wf = ls_hit_rate(expanded_wf)

    # Average number of features selected per fold
    avg_k = monthly_wf["n_selected"].mean() if "n_selected" in monthly_wf.columns else float("nan")

    row_wf = {
        "Model": label_wf,
        "n_features": round(avg_k, 1),   # average over folds
        "features": "dynamic (per-fold LassoCV)",
        "IC_mean": round(ic_wf.mean(), 4),
        "IC_std": round(ic_wf.std(), 4),
        "IC_IR": round(compute_ic_ir(ic_wf), 4),
        "Hit_IC": round(compute_hit_rate(ic_wf), 3),
        "LS_Sharpe": round(sharpe_wf, 3),
        "LS_Hit": round(ls_hit_wf, 3),
        "n_months": len(ic_wf),
    }
    summary.append(row_wf)

    expanded_wf.to_parquet(OUTPUT / f"results_{label_wf}.parquet", index=False)
    print(f"  IC={row_wf['IC_mean']:+.4f}  IC_IR={row_wf['IC_IR']:+.4f}  "
          f"Hit_IC={row_wf['Hit_IC']:.1%}  Sharpe={row_wf['LS_Sharpe']:.3f}  "
          f"LS_Hit={row_wf['LS_Hit']:.1%}  avg_k={avg_k:.1f}")

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(OUTPUT / "rung2c_selected_ols_summary.csv", index=False)
    print("\n" + "=" * 80)
    print("RUNG 2c SUMMARY (OLS on selected features)")
    print("=" * 80)
    print(df_sum.drop(columns=["features"]).to_string(index=False))
    print(f"\nSaved -> {OUTPUT / 'rung2c_selected_ols_summary.csv'}")


if __name__ == "__main__":
    main()
