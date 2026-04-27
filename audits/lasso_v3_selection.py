"""
LASSO V3 Feature Selection Analysis
====================================
Walk-forward LASSO on the 22 V3 features to identify which survive fold-by-fold
selection. Also computes univariate IC and cross-correlation matrix.

Outputs:
  output/lasso_v3_coefs_by_fold.csv       -- per-fold coefficients
  output/lasso_v3_selection_summary.csv   -- per-feature pct_nonzero / mean_coef / mean_abs_coef
  output/v3_feature_correlation.csv       -- 22x22 Pearson correlation matrix

Run:
  python scripts/lasso_v3_selection.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"   # prevents segfault on macOS with LassoCV

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# Allow imports from project root when called as scripts/lasso_v3_selection.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from config import (
    ALL_FEATURE_COLS_V3,
    TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA    = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT  = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# ── LASSO hyperparameters ──────────────────────────────────────────────────
LASSO_ALPHAS = np.logspace(-4, 1, 50)
LASSO_CV     = 5
LASSO_ITER   = 10_000

HIGH_CORR_THRESHOLD = 0.7   # flag pairs with |corr| above this


# ── Walk-forward LASSO with coef tracking ─────────────────────────────────

def run_lasso_walk_forward(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Walk-forward LASSO with 60-month min train, 1-month purge.
    Returns DataFrame with columns: date, fold, alpha_chosen, + one column per feature.
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    n_months = len(months)

    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS
    records = []

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

        # Fit scaler on train only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        model = LassoCV(alphas=LASSO_ALPHAS, cv=LASSO_CV, max_iter=LASSO_ITER)
        model.fit(X_tr_s, y_tr)

        y_pred  = model.predict(X_te_s)
        ic_val  = spearmanr(y_te, y_pred).statistic if np.std(y_pred) > 1e-10 else np.nan

        row = {
            DATE_COL:      test_month,
            "fold":        i,
            "alpha_chosen": model.alpha_,
            "IC":          ic_val,
            "n_test":      len(df_te),
        }
        for feat, coef in zip(features, model.coef_):
            row[feat] = coef

        records.append(row)

        if i % 12 == 0:
            n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-10))
            print(f"  fold {i:3d} | {str(test_month)[:7]} | "
                  f"alpha={model.alpha_:.5f} | nonzero={n_nonzero:2d}/{len(features)} | IC={ic_val:+.4f}")

    return pd.DataFrame(records)


# ── Selection summary ──────────────────────────────────────────────────────

def compute_selection_summary(coef_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-feature: pct_nonzero, mean_coef, mean_abs_coef, n_folds."""
    rows = []
    n_folds = len(coef_df)
    for feat in features:
        vals = coef_df[feat].values
        nonzero_mask = np.abs(vals) > 1e-10
        rows.append({
            "feature":       feat,
            "pct_nonzero":   round(nonzero_mask.mean(), 4),
            "mean_coef":     round(vals.mean(), 6),
            "mean_abs_coef": round(np.abs(vals).mean(), 6),
            "n_folds_selected": int(nonzero_mask.sum()),
            "n_folds_total":    n_folds,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("pct_nonzero", ascending=False)
        .reset_index(drop=True)
    )


# ── Univariate IC (full sample) ────────────────────────────────────────────

def compute_univariate_ic(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Monthly univariate Spearman IC averaged over all months."""
    ic_records = {f: [] for f in features}
    for month, grp in df.groupby(DATE_COL):
        y = grp[TARGET_COL].values
        if len(grp) < 10:
            continue
        for feat in features:
            x = grp[feat].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 10:
                ic_records[feat].append(np.nan)
                continue
            ic_val = spearmanr(x[mask], y[mask]).statistic
            ic_records[feat].append(ic_val)

    return pd.Series(
        {f: np.nanmean(vals) for f, vals in ic_records.items()},
        name="univariate_IC_mean"
    )


# ── Correlation matrix + redundancy flags ─────────────────────────────────

def compute_feature_correlation(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix of all 22 features (row-wise, ignoring NaN)."""
    X = df[features].copy()
    return X.corr(method="pearson")


def flag_high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float) -> list[dict]:
    """Return list of (feat_a, feat_b, corr) for |corr| > threshold (upper triangle)."""
    flags = []
    feats = corr_matrix.columns.tolist()
    for i, fa in enumerate(feats):
        for j, fb in enumerate(feats):
            if j <= i:
                continue
            val = corr_matrix.loc[fa, fb]
            if abs(val) > threshold:
                flags.append({"feat_a": fa, "feat_b": fb, "correlation": round(val, 4)})
    return sorted(flags, key=lambda x: abs(x["correlation"]), reverse=True)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LASSO V3 FEATURE SELECTION ANALYSIS")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading {DATA} ...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    features = ALL_FEATURE_COLS_V3
    print(f"  {len(df):,} rows | {df[DATE_COL].nunique()} months | {len(features)} V3 features")
    print(f"  Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")

    # Verify all features present
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing V3 features in parquet: {missing_feats}")
    print(f"  All {len(features)} V3 features confirmed present in parquet.")

    # ── Walk-forward LASSO ─────────────────────────────────────────────────
    print(f"\nRunning walk-forward LASSO "
          f"(min_train={DEFAULT_MIN_TRAIN_MONTHS}m, purge={DEFAULT_PURGE_MONTHS}m) ...")
    coef_df = run_lasso_walk_forward(df, features)

    coef_out = OUTPUT / "lasso_v3_coefs_by_fold.csv"
    coef_df.to_csv(coef_out, index=False)
    print(f"\n  Saved: {coef_out}  ({len(coef_df)} folds)")

    # ── Selection summary ──────────────────────────────────────────────────
    summary = compute_selection_summary(coef_df, features)

    # Tag phase-2 features
    phase2_feats = set(
        ["Turnover_zscore", "AmihudIlliquidity_zscore",
         "High52W_Proximity_zscore", "MaxDailyReturn_zscore", "ReturnSkewness_zscore",
         "ImpliedEPSGrowth_zscore", "QRevGrowthYoY_zscore",
         "AnalystCoverageChg_zscore"]
    )
    summary["phase"] = summary["feature"].apply(
        lambda f: "Phase2" if f in phase2_feats else "V2"
    )

    # Univariate IC
    print("\nComputing univariate monthly IC (full sample) ...")
    univ_ic = compute_univariate_ic(df, features)
    summary["univariate_IC"] = summary["feature"].map(univ_ic).round(6)

    sel_out = OUTPUT / "lasso_v3_selection_summary.csv"
    summary.to_csv(sel_out, index=False)
    print(f"  Saved: {sel_out}")

    # ── Correlation matrix ─────────────────────────────────────────────────
    print("\nComputing feature cross-correlation matrix ...")
    corr_matrix = compute_feature_correlation(df, features)
    corr_out = OUTPUT / "v3_feature_correlation.csv"
    corr_matrix.to_csv(corr_out)
    print(f"  Saved: {corr_out}")

    high_corr_pairs = flag_high_correlation_pairs(corr_matrix, HIGH_CORR_THRESHOLD)

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # IC of LASSO walk-forward itself
    ic_series = coef_df["IC"].dropna()
    print(f"\nLASSO Walk-Forward IC: mean={ic_series.mean():+.4f}, "
          f"std={ic_series.std():.4f}, n_folds={len(ic_series)}")

    # Top 10 by selection frequency
    print("\nTop 10 features by LASSO selection frequency:")
    print(f"{'Rank':<5} {'Feature':<35} {'Pct_Nonzero':>12} {'Phase':>8} {'Univ_IC':>10}")
    print("-" * 75)
    for rank, row in summary.head(10).iterrows():
        print(f"  {rank+1:<3}  {row['feature']:<35} {row['pct_nonzero']:>11.1%}   "
              f"{row['phase']:>7}  {row['univariate_IC']:>+9.5f}")

    # Phase 2 features in top 10
    top10 = summary.head(10)
    phase2_in_top10 = top10[top10["phase"] == "Phase2"]
    print(f"\nPhase 2 features in top 10: {len(phase2_in_top10)}")
    if len(phase2_in_top10):
        for _, row in phase2_in_top10.iterrows():
            rank = summary[summary["feature"] == row["feature"]].index[0] + 1
            print(f"  #{rank}: {row['feature']}  (pct_nonzero={row['pct_nonzero']:.1%})")

    # Full ranking
    print("\nFull feature ranking (all 22):")
    print(f"{'Rank':<5} {'Feature':<35} {'Pct_Nonzero':>12} {'Mean_AbsCoef':>14} {'Phase':>8} {'Univ_IC':>10}")
    print("-" * 90)
    for rank, row in summary.iterrows():
        print(f"  {rank+1:<3}  {row['feature']:<35} {row['pct_nonzero']:>11.1%}  "
              f"{row['mean_abs_coef']:>13.6f}   {row['phase']:>7}  {row['univariate_IC']:>+9.5f}")

    # Redundancy flags
    print(f"\nHigh-correlation pairs (|corr| > {HIGH_CORR_THRESHOLD}):")
    if high_corr_pairs:
        print(f"  {'Feature A':<35} {'Feature B':<35} {'Corr':>8}")
        print("  " + "-" * 80)
        for pair in high_corr_pairs:
            tag_a = " [P2]" if pair["feat_a"] in phase2_feats else ""
            tag_b = " [P2]" if pair["feat_b"] in phase2_feats else ""
            print(f"  {pair['feat_a'] + tag_a:<35} {pair['feat_b'] + tag_b:<35} {pair['correlation']:>+8.4f}")
    else:
        print("  None found above threshold.")

    print(f"\nOutputs written to {OUTPUT}/")
    print("  lasso_v3_coefs_by_fold.csv")
    print("  lasso_v3_selection_summary.csv")
    print("  v3_feature_correlation.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
