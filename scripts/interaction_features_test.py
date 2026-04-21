"""
Interaction Features Experiment — MA751 Project
Tests whether products of top features (IVOL, GrossProfitability, Beta) add
predictive power beyond the 14-feature baseline.

Three walk-forward LASSO variants:
  - baseline:       original 14 features
  - baseline+ixn:   14 + 5 interaction features (19 total)
  - ixn_only:       5 interaction features only

Output: output/interaction_features_test.csv
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from config import (
    ALL_FEATURE_COLS_V2, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# ── Interaction definitions ───────────────────────────────────────────────────
INTERACTIONS = {
    "IVOL_x_GP":      ("IVOL_zscore",             "GrossProfitability_zscore"),
    "IVOL_x_Beta":    ("IVOL_zscore",             "Beta_zscore"),
    "GP_x_ShortInt":  ("GrossProfitability_zscore","ShortInterestRatio_zscore"),
    "Mom_x_IVOL":     ("Momentum12_1_zscore",      "IVOL_zscore"),
    "Beta_x_GP":      ("Beta_zscore",              "GrossProfitability_zscore"),
}

INTERACTION_COLS = list(INTERACTIONS.keys())
LASSO_ALPHAS = np.logspace(-4, 1, 50)


# ── Feature engineering ───────────────────────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction terms (product of already-z-scored columns)."""
    df = df.copy()
    for name, (col_a, col_b) in INTERACTIONS.items():
        df[name] = df[col_a] * df[col_b]
    return df


# ── Walk-forward LASSO ────────────────────────────────────────────────────────
def run_lasso_walk_forward(df: pd.DataFrame, features: list[str], label: str):
    """
    Walk-forward LASSO with coefficient tracking.

    Returns:
        monthly_df  — one row per test month (IC, LS return, fold index)
        coef_df     — one row per fold × feature (non-zero coefficients)
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())

    monthly_rows = []
    coef_rows = []

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

        model = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=10000)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        # Fallback if LASSO zeros everything
        if np.std(y_pred) < 1e-10:
            fb = LinearRegression()
            fb.fit(X_tr_s, y_tr)
            y_pred = fb.predict(X_te_s)
            coefs = fb.coef_
        else:
            coefs = model.coef_

        ic = spearmanr(y_te, y_pred).statistic

        # Long-short return for this month
        g = pd.DataFrame({"y_pred": y_pred, "y_true": y_te})
        g = g.sort_values("y_pred", ascending=False)
        n_q = max(1, int(len(g) * 0.2))
        ls_ret = g.head(n_q)["y_true"].mean() - g.tail(n_q)["y_true"].mean()

        monthly_rows.append({
            DATE_COL: test_month,
            "fold": i,
            "IC": ic,
            "ls_ret": ls_ret,
            "n_test": len(df_te),
            "alpha": model.alpha_ if hasattr(model, "alpha_") else np.nan,
            "n_nonzero": int(np.sum(np.abs(coefs) > 1e-8)),
        })

        # Record non-zero coefficients
        for feat, coef in zip(features, coefs):
            if abs(coef) > 1e-8:
                coef_rows.append({
                    DATE_COL: test_month,
                    "fold": i,
                    "feature": feat,
                    "coef": coef,
                    "variant": label,
                })

        if i % 12 == 0:
            print(f"  [{label}] fold {i} | {str(test_month)[:7]} | IC={ic:+.4f} | "
                  f"n_nonzero={int(np.sum(np.abs(coefs) > 1e-8))}")

    monthly_df = pd.DataFrame(monthly_rows)
    coef_df = pd.DataFrame(coef_rows) if coef_rows else pd.DataFrame(
        columns=[DATE_COL, "fold", "feature", "coef", "variant"]
    )
    return monthly_df, coef_df


# ── Summary helpers ───────────────────────────────────────────────────────────
def summarize_monthly(monthly_df: pd.DataFrame, label: str, n_features: int) -> dict:
    ic = monthly_df["IC"].dropna()
    ls_rets = monthly_df["ls_ret"].dropna()

    sharpe = np.nan
    if len(ls_rets) >= 2 and ls_rets.std(ddof=1) > 1e-8:
        sharpe = float(np.sqrt(12) * ls_rets.mean() / ls_rets.std(ddof=1))

    return {
        "variant": label,
        "n_features": n_features,
        "n_months": len(ic),
        "IC_mean": round(float(ic.mean()), 4),
        "IC_std": round(float(ic.std(ddof=1)), 4),
        "IC_IR": round(compute_ic_ir(ic), 4),
        "hit_rate": round(compute_hit_rate(ic), 3),
        "LS_Sharpe": round(sharpe, 3),
    }


def selection_rate_summary(coef_df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Fraction of folds where each feature had a non-zero coefficient."""
    if len(coef_df) == 0:
        return pd.DataFrame()

    per_feature = (
        coef_df.groupby("feature")
        .agg(
            n_selected=("fold", "nunique"),
            mean_coef=("coef", "mean"),
            mean_abs_coef=("coef", lambda x: x.abs().mean()),
        )
        .reset_index()
    )
    per_feature["selection_rate"] = per_feature["n_selected"] / n_folds
    return per_feature.sort_values("selection_rate", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("INTERACTION FEATURES EXPERIMENT")
    print("=" * 70)

    print("\nLoading data...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows | {df[DATE_COL].nunique()} months | "
          f"{df[STOCK_COL].nunique()} tickers")

    print("\nComputing interaction features...")
    df = add_interaction_features(df)

    # Verify all required columns exist
    all_needed = ALL_FEATURE_COLS_V2 + INTERACTION_COLS
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Describe interaction distributions
    print("\nInteraction feature statistics (cross-section, all months):")
    print(df[INTERACTION_COLS].describe().round(3).to_string())

    # ── Three variants ────────────────────────────────────────────────────────
    variants = [
        ("baseline",     ALL_FEATURE_COLS_V2),
        ("baseline+ixn", ALL_FEATURE_COLS_V2 + INTERACTION_COLS),
        ("ixn_only",     INTERACTION_COLS),
    ]

    all_monthly = {}
    all_coefs = {}
    summaries = []

    for label, features in variants:
        print(f"\n{'─' * 60}")
        print(f"Running walk-forward LASSO: {label} ({len(features)} features)")
        print(f"{'─' * 60}")

        monthly_df, coef_df = run_lasso_walk_forward(df, features, label)
        all_monthly[label] = monthly_df
        all_coefs[label] = coef_df

        n_folds = monthly_df["fold"].nunique()
        row = summarize_monthly(monthly_df, label, len(features))
        summaries.append(row)

        print(f"\n  → IC={row['IC_mean']:+.4f} ± {row['IC_std']:.4f} | "
              f"IC_IR={row['IC_IR']:.4f} | Hit={row['hit_rate']:.1%} | "
              f"LS_Sharpe={row['LS_Sharpe']:.3f}")

    # ── Selection rates for baseline+ixn ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("LASSO FEATURE SELECTION RATES — baseline+ixn variant")
    print("=" * 70)

    ixn_coef_df = all_coefs["baseline+ixn"]
    n_folds_ixn = all_monthly["baseline+ixn"]["fold"].nunique()
    sel_df = selection_rate_summary(ixn_coef_df, n_folds_ixn)

    # Annotate whether feature is an interaction
    sel_df["is_interaction"] = sel_df["feature"].isin(INTERACTION_COLS)
    print(sel_df[["feature", "selection_rate", "mean_coef", "mean_abs_coef", "is_interaction"]]
          .to_string(index=False))

    # ── Interaction-specific summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERACTION FEATURES — selection rate (>50% = real signal)")
    print("=" * 70)
    ixn_sel = sel_df[sel_df["is_interaction"]].copy()
    if len(ixn_sel) == 0:
        print("  None of the interaction features were ever selected by LASSO.")
    else:
        for _, r in ixn_sel.iterrows():
            flag = "*** SIGNAL ***" if r["selection_rate"] > 0.5 else ""
            print(f"  {r['feature']:20s}  sel={r['selection_rate']:.1%}  "
                  f"mean_coef={r['mean_coef']:+.4f}  {flag}")

    # ── Overall summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))

    # ── IC improvement check ──────────────────────────────────────────────────
    base_ic = summaries[0]["IC_mean"]
    aug_ic  = summaries[1]["IC_mean"]
    delta   = aug_ic - base_ic
    print(f"\nIC delta (baseline+ixn vs baseline): {delta:+.4f}")
    if delta > 0.001:
        print("  → Interactions provide marginal IC lift.")
    elif delta < -0.001:
        print("  → Interactions hurt IC (noise / collinearity).")
    else:
        print("  → Interactions have negligible IC impact.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Combine all monthly results
    combined_monthly = []
    for label, mdf in all_monthly.items():
        mdf = mdf.copy()
        mdf["variant"] = label
        combined_monthly.append(mdf)
    combined_monthly_df = pd.concat(combined_monthly, ignore_index=True)
    combined_monthly_df.to_csv(OUTPUT / "interaction_features_test.csv", index=False)

    # Save coefficient / selection data
    combined_coefs = pd.concat(all_coefs.values(), ignore_index=True)
    combined_coefs.to_csv(OUTPUT / "interaction_coefs.csv", index=False)

    # Save summary
    summary_df.to_csv(OUTPUT / "interaction_summary.csv", index=False)

    print(f"\nSaved:")
    print(f"  {OUTPUT / 'interaction_features_test.csv'}  (monthly IC + LS returns per variant)")
    print(f"  {OUTPUT / 'interaction_coefs.csv'}           (per-fold LASSO coefficients)")
    print(f"  {OUTPUT / 'interaction_summary.csv'}         (summary table)")


if __name__ == "__main__":
    main()
