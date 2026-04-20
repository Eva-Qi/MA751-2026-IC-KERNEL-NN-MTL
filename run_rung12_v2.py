"""
Rung 1-2 on V2 data: OLS + IC-weighted ensemble + LASSO + Ridge
Uses the 25-feature WRDS-based master_panel_v2.parquet
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from config import (
    ALL_FEATURE_COLS_V2, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_monthly_ic, compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

IC_WEIGHT_WINDOW = 12
LASSO_ALPHAS = np.logspace(-4, 1, 50)
RIDGE_ALPHAS = np.logspace(-4, 4, 50)


def run_walk_forward(df, model_fn, label, features=None):
    """Generic walk-forward for sklearn-style models."""
    if features is None:
        features = ALL_FEATURE_COLS_V2

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

        # Scale
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


def ols_model(X_tr, y_tr, X_te, features):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def ic_ensemble_model(X_tr, y_tr, X_te, features):
    """IC-weighted univariate ensemble."""
    n_features = X_tr.shape[1]
    # Use last IC_WEIGHT_WINDOW months of training data for IC computation
    n_recent = min(len(X_tr), IC_WEIGHT_WINDOW * 400)  # ~400 stocks * 12 months
    X_recent = X_tr[-n_recent:]
    y_recent = y_tr[-n_recent:]

    ics = []
    preds_per_feature = []
    for j in range(n_features):
        # Univariate OLS
        model = LinearRegression()
        model.fit(X_tr[:, j:j+1], y_tr)
        pred_j = model.predict(X_te[:, j:j+1])
        preds_per_feature.append(pred_j)

        # IC for weighting
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
    # If LASSO shrinks everything to 0 (constant prediction), fall back to OLS
    if np.std(pred) < 1e-10:
        fallback = LinearRegression()
        fallback.fit(X_tr, y_tr)
        pred = fallback.predict(X_te)
    return pred


def ridge_model(X_tr, y_tr, X_te, features):
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def expand_results(monthly_df):
    """Expand monthly summary into per-stock results for Sharpe computation."""
    rows = []
    for _, r in monthly_df.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t, "y_pred": yp, "y_true": yt})
    return pd.DataFrame(rows)


def main():
    print("Loading V2 panel...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, {len(ALL_FEATURE_COLS_V2)} features")

    models = [
        ("1a_OLS_v2", ols_model),
        ("1b_IC_Ensemble_v2", ic_ensemble_model),
        ("2a_LASSO_v2", lasso_model),
        ("2b_Ridge_v2", ridge_model),
    ]

    all_results = {}
    for label, fn in models:
        print(f"\nRunning {label}...")
        monthly = run_walk_forward(df, fn, label)
        all_results[label] = monthly

        # Compute metrics
        ic_series = monthly["IC"].dropna()
        expanded = expand_results(monthly)
        sharpe = compute_long_short_sharpe(expanded)

        print(f"  {label}: IC={ic_series.mean():+.4f} (std={ic_series.std():.4f}), "
              f"IC_IR={compute_ic_ir(ic_series):.4f}, "
              f"Hit={compute_hit_rate(ic_series):.1%}, "
              f"Sharpe={sharpe:.3f}, "
              f"n_months={len(ic_series)}")

        # Save
        expanded.to_parquet(OUTPUT / f"results_{label}.parquet", index=False)
        monthly.to_csv(OUTPUT / f"monthly_{label}.csv", index=False)

    # Summary table
    print("\n" + "=" * 80)
    print("RUNG 1-2 SUMMARY (V2 DATA)")
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
    summary.to_csv(OUTPUT / "rung12_v2_summary.csv", index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
