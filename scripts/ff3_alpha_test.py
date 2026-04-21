"""
FF3 Alpha Test
==============
For each model's long-short return series, regress against Fama-French 3 factors:

    LS_ret_t = alpha + beta1*MktRF_t + beta2*SMB_t + beta3*HML_t + eps_t

Uses Newey-West HAC standard errors for alpha t-stat.

Interpretation:
  alpha ≈ 0, insignificant  -> model captures known risk premia only
  alpha > 0, significant    -> incremental predictive power beyond FF3

Output: output/ff3_alpha_test.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATE_COL  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output"
FF_PATH = PROJECT_ROOT / "data" / "wrds" / "ff_factors_monthly.parquet"
LS_PATH = OUTPUT_DIR / "ls_monthly_returns.csv"
OUT_PATH = OUTPUT_DIR / "ff3_alpha_test.csv"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_ls_returns() -> pd.DataFrame:
    """Load long-short monthly returns. Columns: model, date, ls_ret."""
    ls = pd.read_csv(LS_PATH)
    # Normalize date to month-start for merge alignment
    ls["date"] = pd.to_datetime(ls["date"])
    ls["month_key"] = ls["date"].dt.to_period("M")
    return ls


def load_ff3() -> pd.DataFrame:
    """Load FF3 factors. Columns: date, mktrf, smb, hml, rf."""
    ff = pd.read_parquet(FF_PATH)
    ff["date"] = pd.to_datetime(ff["date"])
    ff["month_key"] = ff["date"].dt.to_period("M")
    # Keep only FF3 columns needed
    return ff[["month_key", "mktrf", "smb", "hml", "rf"]].copy()


# ── Newey-West OLS regression ─────────────────────────────────────────────────

def nw_ols(y: np.ndarray, X: np.ndarray, max_lag: int | None = None) -> dict:
    """
    OLS regression with Newey-West HAC standard errors.

    Parameters
    ----------
    y      : dependent variable (T,)
    X      : regressors WITH constant as first column (T, k)
    max_lag: NW bandwidth; None => floor(T^(1/3)) (Andrews rule)

    Returns
    -------
    dict: alpha, alpha_tstat, alpha_pval, betas, r2, n
    """
    n = len(y)
    if n < 10:
        return {
            "alpha": np.nan, "alpha_tstat": np.nan, "alpha_pval": np.nan,
            "beta_mktrf": np.nan, "beta_smb": np.nan, "beta_hml": np.nan,
            "r2": np.nan, "n": n,
        }

    if max_lag is None:
        max_lag = max(1, int(np.floor(n ** (1 / 3))))

    model = sm.OLS(y, X)
    res = model.fit()

    # Newey-West covariance via statsmodels
    nw_cov = cov_hac(res, nlags=max_lag)
    nw_se = np.sqrt(np.diag(nw_cov))

    coef = res.params
    tstat = coef / nw_se
    pval = 2 * (1 - scipy_norm.cdf(np.abs(tstat)))  # two-sided

    return {
        "alpha":        float(coef[0]),
        "alpha_tstat":  float(tstat[0]),
        "alpha_pval":   float(pval[0]),
        "beta_mktrf":   float(coef[1]),
        "beta_smb":     float(coef[2]),
        "beta_hml":     float(coef[3]),
        "r2":           float(res.rsquared),
        "n":            n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ff3_alpha_test() -> pd.DataFrame:
    ls = load_ls_returns()
    ff = load_ff3()

    models = sorted(ls["model"].unique())
    records = []

    for model_name in models:
        sub = ls[ls["model"] == model_name].copy()
        merged = sub.merge(ff, on="month_key", how="inner")
        merged = merged.sort_values("date").reset_index(drop=True)

        if len(merged) < 10:
            print(f"  [SKIP] {model_name}: only {len(merged)} overlapping months")
            continue

        y = merged["ls_ret"].values
        X = sm.add_constant(merged[["mktrf", "smb", "hml"]].values)

        result = nw_ols(y, X)
        result["model"] = model_name
        result["date_start"] = merged["date"].min().strftime("%Y-%m")
        result["date_end"] = merged["date"].max().strftime("%Y-%m")
        records.append(result)

    df = pd.DataFrame(records)
    # Reorder columns
    cols = [
        "model", "n", "date_start", "date_end",
        "alpha", "alpha_tstat", "alpha_pval",
        "beta_mktrf", "beta_smb", "beta_hml",
        "r2",
    ]
    df = df[cols]
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("FF3 ALPHA TEST  —  Newey-West HAC t-stats")
    print("=" * 78)
    header = (
        f"{'Model':<22} {'N':>4}  "
        f"{'Alpha (ann%)':>12}  {'t-stat':>7}  {'p-val':>6}  "
        f"{'MktRF':>7}  {'SMB':>7}  {'HML':>7}  {'R²':>6}"
    )
    print(header)
    print("-" * 78)

    for _, row in df.iterrows():
        alpha_ann = row["alpha"] * 12 * 100  # annualise: monthly → annual %
        sig = (
            "***" if row["alpha_pval"] < 0.01 else
            "**"  if row["alpha_pval"] < 0.05 else
            "*"   if row["alpha_pval"] < 0.10 else
            ""
        )
        print(
            f"{row['model']:<22} {int(row['n']):>4}  "
            f"{alpha_ann:>11.2f}%  {row['alpha_tstat']:>7.2f}  {row['alpha_pval']:>6.3f}  "
            f"{row['beta_mktrf']:>7.3f}  {row['beta_smb']:>7.3f}  {row['beta_hml']:>7.3f}  "
            f"{row['r2']:>6.3f}  {sig}"
        )

    print("=" * 78)
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10  (Newey-West SE)")
    print("Alpha annualised = monthly alpha × 12 × 100%")
    print()

    # Interpretation
    sig_models = df[df["alpha_pval"] < 0.10]
    insig_models = df[df["alpha_pval"] >= 0.10]

    if len(sig_models):
        print("Models with significant alpha (p<0.10) → incremental power beyond FF3:")
        for _, r in sig_models.iterrows():
            sign = "+" if r["alpha"] > 0 else "-"
            print(f"  {r['model']:<22}  alpha={r['alpha']*1200:.1f}% ann  (t={r['alpha_tstat']:.2f})")
    else:
        print("No model shows significant alpha at p<0.10.")

    if len(insig_models):
        print("\nModels with insignificant alpha → returns explained by FF3 risk premia:")
        for _, r in insig_models.iterrows():
            print(f"  {r['model']:<22}  alpha={r['alpha']*1200:.1f}% ann  (t={r['alpha_tstat']:.2f})")
    print()


def main():
    print("Loading data...")
    df = run_ff3_alpha_test()

    print_summary(df)

    df.to_csv(OUT_PATH, index=False, float_format="%.6f")
    print(f"Results saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
