"""
Sector-Adjusted Return Target Experiment
=========================================
Tests whether model signals capture true within-sector alpha
or just sector rotation.

Methodology:
  - Raw target:          fwd_ret_1m   (standard)
  - Sector-adj target:   fwd_ret_1m_secadj = fwd_ret_1m - sector_mean(fwd_ret_1m)

Walk-forward setup mirrors run_rung12_v2.py exactly:
  - Min train: 60 months, purge: 1 month
  - StandardScaler fit on train, transform on test
  - Models: OLS + LASSO (matching rung 1a and 2a)

Key question:
  If IVOL/Beta/GP are still selected by LASSO on sec-adj returns
  → signals capture within-sector alpha.
  If different features emerge (AnalystRevision, Accruals)
  → original signal was largely sector rotation.

Output: output/sector_adjusted_comparison.csv
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler

from config import (
    ALL_FEATURE_COLS_V2, TARGET_COL, SECADJ_TARGET_COL,
    DATE_COL, STOCK_COL, SECTOR_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

LASSO_ALPHAS = np.logspace(-4, 1, 50)
FEATURES = ALL_FEATURE_COLS_V2


# ── Helpers ──────────────────────────────────────────────────────────────

def ensure_secadj(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fwd_ret_1m_secadj if not already present."""
    if SECADJ_TARGET_COL not in df.columns:
        print("  [info] fwd_ret_1m_secadj not found — computing on the fly...")
        sector_mean = df.groupby([DATE_COL, SECTOR_COL])[TARGET_COL].transform("mean")
        df = df.copy()
        df[SECADJ_TARGET_COL] = df[TARGET_COL] - sector_mean
    return df


def walk_forward_ols(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward OLS.
    Returns (monthly_results, per_stock_results).
    monthly_results has DATE_COL and IC columns.
    per_stock_results has DATE_COL, STOCK_COL, y_pred, y_true columns.
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    monthly_rows = []
    stock_rows = []

    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS
    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end].dropna(subset=[target_col])
        df_te = df[df[DATE_COL] == test_month].dropna(subset=[target_col])

        if len(df_te) < 10:
            continue

        X_tr = np.nan_to_num(df_tr[FEATURES].values, nan=0.0)
        y_tr = df_tr[target_col].values
        X_te = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[target_col].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LinearRegression()
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        ic = spearmanr(y_te, y_pred).statistic
        monthly_rows.append({DATE_COL: test_month, "IC": ic, "n_test": len(df_te)})

        for t, yp, yt in zip(df_te[STOCK_COL].values, y_pred, y_te):
            stock_rows.append({DATE_COL: test_month, STOCK_COL: t, "y_pred": yp, "y_true": yt})

    return pd.DataFrame(monthly_rows), pd.DataFrame(stock_rows)


def walk_forward_lasso(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Walk-forward LASSO.
    Returns (monthly_results, per_stock_results, feature_coef_history).
    feature_coef_history: {feature: [coefs across folds]}
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    monthly_rows = []
    stock_rows = []
    coef_history = {f: [] for f in FEATURES}

    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS
    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end].dropna(subset=[target_col])
        df_te = df[df[DATE_COL] == test_month].dropna(subset=[target_col])

        if len(df_te) < 10:
            continue

        X_tr = np.nan_to_num(df_tr[FEATURES].values, nan=0.0)
        y_tr = df_tr[target_col].values
        X_te = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[target_col].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=10000)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)

        # Fallback if LASSO shrinks to 0
        if np.std(y_pred) < 1e-10:
            fallback = LinearRegression()
            fallback.fit(X_tr_s, y_tr)
            y_pred = fallback.predict(X_te_s)
            coefs = fallback.coef_
        else:
            coefs = model.coef_

        # Record coefs
        for f, c in zip(FEATURES, coefs):
            coef_history[f].append(c)

        ic = spearmanr(y_te, y_pred).statistic
        monthly_rows.append({DATE_COL: test_month, "IC": ic, "n_test": len(df_te)})

        for t, yp, yt in zip(df_te[STOCK_COL].values, y_pred, y_te):
            stock_rows.append({DATE_COL: test_month, STOCK_COL: t, "y_pred": yp, "y_true": yt})

    return pd.DataFrame(monthly_rows), pd.DataFrame(stock_rows), coef_history


def summarize(monthly: pd.DataFrame, stock: pd.DataFrame) -> dict:
    ic_s = monthly["IC"].dropna()
    sharpe = compute_long_short_sharpe(stock) if len(stock) > 0 else np.nan
    return {
        "IC_mean":  round(float(ic_s.mean()), 4) if len(ic_s) else np.nan,
        "IC_std":   round(float(ic_s.std(ddof=1)), 4) if len(ic_s) > 1 else np.nan,
        "IC_IR":    round(compute_ic_ir(ic_s), 4),
        "Hit_Rate": round(compute_hit_rate(ic_s), 3),
        "LS_Sharpe": round(float(sharpe), 3) if pd.notna(sharpe) else np.nan,
        "n_months": len(ic_s),
    }


def coef_summary(coef_history: dict) -> pd.DataFrame:
    rows = []
    for feat, vals in coef_history.items():
        arr = np.array(vals)
        nonzero_pct = float((arr != 0).mean()) if len(arr) else np.nan
        rows.append({
            "feature": feat,
            "mean_coef": round(float(arr.mean()), 5) if len(arr) else np.nan,
            "abs_mean_coef": round(float(np.abs(arr).mean()), 5) if len(arr) else np.nan,
            "nonzero_pct": round(nonzero_pct, 3),
        })
    return pd.DataFrame(rows).sort_values("abs_mean_coef", ascending=False)


def print_comparison_table(metrics_raw: dict, metrics_sa: dict):
    col_w = 16
    print("\n" + "=" * 72)
    print("IC / SHARPE COMPARISON: RAW vs SECTOR-ADJUSTED TARGET")
    print("=" * 72)
    header = f"{'Metric':<20} {'Raw fwd_ret_1m':>{col_w}} {'SecAdj':>{col_w}} {'Delta':>{col_w}}"
    print(header)
    print("-" * 72)
    keys = ["IC_mean", "IC_std", "IC_IR", "Hit_Rate", "LS_Sharpe", "n_months"]
    for k in keys:
        raw_v = metrics_raw.get(k, np.nan)
        sa_v  = metrics_sa.get(k, np.nan)
        try:
            delta = sa_v - raw_v
            delta_s = f"{delta:+.4f}"
        except Exception:
            delta_s = "n/a"
        print(f"  {k:<18} {str(raw_v):>{col_w}} {str(sa_v):>{col_w}} {delta_s:>{col_w}}")
    print("=" * 72)


def print_feature_comparison(coef_raw: dict, coef_sa: dict):
    raw_df = coef_summary(coef_raw).rename(columns={
        "mean_coef": "raw_mean_coef",
        "abs_mean_coef": "raw_abs_coef",
        "nonzero_pct": "raw_nonzero_pct",
    })
    sa_df = coef_summary(coef_sa).rename(columns={
        "mean_coef": "sa_mean_coef",
        "abs_mean_coef": "sa_abs_coef",
        "nonzero_pct": "sa_nonzero_pct",
    })
    merged = raw_df.merge(sa_df, on="feature")
    merged["abs_delta"] = (merged["sa_abs_coef"] - merged["raw_abs_coef"]).round(5)
    merged = merged.sort_values("raw_abs_coef", ascending=False)

    print("\n" + "=" * 100)
    print("LASSO FEATURE IMPORTANCE: RAW vs SECTOR-ADJUSTED")
    print("(mean absolute LASSO coefficient across walk-forward folds)")
    print("=" * 100)
    print(f"  {'Feature':<35} {'Raw |coef|':>12} {'SA |coef|':>12} {'Delta':>10} {'SA nonzero%':>12}")
    print("-" * 100)
    for _, row in merged.iterrows():
        direction = "↑" if row["abs_delta"] > 0 else ("↓" if row["abs_delta"] < 0 else " ")
        print(f"  {row['feature']:<35} {row['raw_abs_coef']:>12.5f} {row['sa_abs_coef']:>12.5f} "
              f"  {direction}{abs(row['abs_delta']):>8.5f} {row['sa_nonzero_pct']:>12.1%}")
    print("=" * 100)

    # Interpretation
    print("\nINTERPRETATION:")
    gainers = merged[merged["abs_delta"] > 0.0001]["feature"].tolist()
    losers  = merged[merged["abs_delta"] < -0.0001]["feature"].tolist()
    if gainers:
        print(f"  Gained importance (within-sector alpha):  {', '.join(gainers)}")
    if losers:
        print(f"  Lost importance  (was sector rotation):   {', '.join(losers)}")

    return merged


def main():
    print("=" * 72)
    print("SECTOR-ADJUSTED RETURN TARGET EXPERIMENT")
    print("=" * 72)

    # ── Load data ──
    print(f"\nLoading {DATA}...")
    if not DATA.exists():
        print(f"ERROR: {DATA} not found. Run load_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows | {df[DATE_COL].nunique()} months | "
          f"{df[STOCK_COL].nunique()} tickers")

    # ── Ensure sector-adjusted column ──
    df = ensure_secadj(df)

    # Quick sanity check
    sample = df[df[DATE_COL] == df[DATE_COL].iloc[0]]
    print(f"\n  Sanity check (first month, first 3 rows):")
    print(f"  {sample[[TARGET_COL, SECADJ_TARGET_COL]].head(3).to_string(index=False)}")
    print(f"  SecAdj mean (should ≈ 0 within sector): "
          f"{df.groupby([DATE_COL, SECTOR_COL])[SECADJ_TARGET_COL].mean().abs().mean():.6f}")

    # ── OLS: raw ──
    print(f"\n[1/4] OLS — raw target ({TARGET_COL})...")
    ols_m_raw, ols_s_raw = walk_forward_ols(df, TARGET_COL)
    metrics_ols_raw = summarize(ols_m_raw, ols_s_raw)
    print(f"  OLS raw: IC={metrics_ols_raw['IC_mean']:+.4f}, "
          f"Sharpe={metrics_ols_raw['LS_Sharpe']:.3f}")

    # ── OLS: sector-adjusted ──
    print(f"\n[2/4] OLS — sector-adjusted target ({SECADJ_TARGET_COL})...")
    ols_m_sa, ols_s_sa = walk_forward_ols(df, SECADJ_TARGET_COL)
    metrics_ols_sa = summarize(ols_m_sa, ols_s_sa)
    print(f"  OLS sec-adj: IC={metrics_ols_sa['IC_mean']:+.4f}, "
          f"Sharpe={metrics_ols_sa['LS_Sharpe']:.3f}")

    # ── LASSO: raw ──
    print(f"\n[3/4] LASSO — raw target ({TARGET_COL})...")
    lasso_m_raw, lasso_s_raw, coef_raw = walk_forward_lasso(df, TARGET_COL)
    metrics_lasso_raw = summarize(lasso_m_raw, lasso_s_raw)
    print(f"  LASSO raw: IC={metrics_lasso_raw['IC_mean']:+.4f}, "
          f"Sharpe={metrics_lasso_raw['LS_Sharpe']:.3f}")

    # ── LASSO: sector-adjusted ──
    print(f"\n[4/4] LASSO — sector-adjusted target ({SECADJ_TARGET_COL})...")
    lasso_m_sa, lasso_s_sa, coef_sa = walk_forward_lasso(df, SECADJ_TARGET_COL)
    metrics_lasso_sa = summarize(lasso_m_sa, lasso_s_sa)
    print(f"  LASSO sec-adj: IC={metrics_lasso_sa['IC_mean']:+.4f}, "
          f"Sharpe={metrics_lasso_sa['LS_Sharpe']:.3f}")

    # ── Print comparison tables ──
    print("\n\n### OLS COMPARISON ###")
    print_comparison_table(metrics_ols_raw, metrics_ols_sa)

    print("\n\n### LASSO COMPARISON ###")
    print_comparison_table(metrics_lasso_raw, metrics_lasso_sa)

    feature_compare = print_feature_comparison(coef_raw, coef_sa)

    # ── Save results ──
    out_rows = []
    for model_name, m_raw, m_sa in [
        ("OLS",   metrics_ols_raw,   metrics_ols_sa),
        ("LASSO", metrics_lasso_raw, metrics_lasso_sa),
    ]:
        for target_label, m in [("raw", m_raw), ("secadj", m_sa)]:
            row = {"model": model_name, "target": target_label}
            row.update(m)
            out_rows.append(row)

    results_df = pd.DataFrame(out_rows)
    results_path = OUTPUT / "sector_adjusted_comparison.csv"
    results_df.to_csv(results_path, index=False)

    # Also save feature comparison
    feat_path = OUTPUT / "sector_adjusted_feature_comparison.csv"
    feature_compare.to_csv(feat_path, index=False)

    print(f"\nSaved:")
    print(f"  {results_path}")
    print(f"  {feat_path}")

    # ── Final verdict ──
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)

    ic_drop = metrics_ols_sa["IC_mean"] - metrics_ols_raw["IC_mean"]
    print(f"  OLS IC change (raw→secadj): {ic_drop:+.4f}")
    lasso_ic_drop = metrics_lasso_sa["IC_mean"] - metrics_lasso_raw["IC_mean"]
    print(f"  LASSO IC change (raw→secadj): {lasso_ic_drop:+.4f}")

    # Which features are most important in secadj model?
    coef_sa_df = coef_summary(coef_sa).head(5)
    print(f"\n  Top-5 features in LASSO (sector-adjusted target):")
    for _, row in coef_sa_df.iterrows():
        print(f"    {row['feature']:<35} |coef|={row['abs_mean_coef']:.5f}  nonzero={row['nonzero_pct']:.0%}")

    # Risk features (Beta/IVOL) status
    risk_features = ["Beta_zscore", "IVOL_zscore"]
    all_top5_sa = coef_sa_df["feature"].tolist()
    risk_retained = [f for f in risk_features if f in all_top5_sa]
    print(f"\n  Risk features (Beta/IVOL) in top-5 (secadj): {risk_retained or 'NONE'}")
    if risk_retained:
        print("  → WITHIN-SECTOR alpha: risk signals work even after removing sector returns")
    else:
        print("  → SECTOR ROTATION: risk signals were capturing sector-level effects")

    print("=" * 72)


if __name__ == "__main__":
    main()
