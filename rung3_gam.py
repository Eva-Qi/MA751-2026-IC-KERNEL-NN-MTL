"""
Rung 3: Generalized Additive Model (GAM) with Natural Cubic Splines
====================================================================

Model:  f(x) = β₀ + s₁(x₁) + s₂(x₂) + ... + s₁₅(x₁₅)

Each sⱼ is a natural cubic spline — linear beyond boundary knots,
preventing wild extrapolation.  The model is additive (no interactions)
but captures smooth nonlinearity per feature.

Library: pygam.LinearGAM

Walk-forward: expanding window, 60-month min train, 1-month purge,
predict one month at a time (matches main.py / Rung 5 exactly).

Key question this rung answers:
  Does smooth nonlinearity help, without risk of misbehaving?
  - GAM >> LASSO   →  factor-return relationships are nonlinear
  - MLP >> GAM     →  factor interactions matter
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Column names (must match load_data.py / main.py) ──────────────────────

FACTOR_ZSCORE_COLS = [
    "EarningsYield_zscore",
    "GrossProfitability_zscore",
    "AssetGrowth_zscore",
    "Accruals_zscore",
    "Momentum12_1_zscore",
    "NetDebtEBITDA_zscore",
]

MACRO_COLS = [
    "T10Y2Y", "VIXCLS", "UMCSENT", "CFNAI", "UNRATE",
    "BAMLH0A0HYM2", "CPI_YOY", "VIX_TERM_STRUCTURE", "LEADING_COMPOSITE",
]

ALL_FEATURE_COLS = FACTOR_ZSCORE_COLS + MACRO_COLS

TARGET_COL  = "fwd_ret_1m"
DATE_COL    = "date"
STOCK_COL   = "ticker"
SECTOR_COL  = "sector"


# ── Walk-forward evaluation ───────────────────────────────────────────────

def walk_forward_evaluate(
    df: pd.DataFrame,
    min_train_months: int = 60,
    purge_months: int = 1,
    n_splines: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward with a GAM at each fold.

    For each test month t:
      - train on all months up to (t - purge_months - 1)
      - fit LinearGAM with natural splines on the 15 features
      - predict cross-section at month t
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []

    for i, test_month in enumerate(
        months[min_train_months + purge_months:],
        start=min_train_months + purge_months,
    ):
        train_end = months[i - purge_months - 1]
        df_tr = df[df[DATE_COL] <= train_end].copy()
        df_te = df[df[DATE_COL] == test_month].copy()

        if df_te.empty:
            continue

        # ── Prepare features ──
        X_tr = np.nan_to_num(df_tr[ALL_FEATURE_COLS].values.astype(np.float64), nan=0.0)
        y_tr = df_tr[TARGET_COL].values.astype(np.float64)
        X_te = np.nan_to_num(df_te[ALL_FEATURE_COLS].values.astype(np.float64), nan=0.0)
        y_te = df_te[TARGET_COL].values.astype(np.float64)

        # StandardScaler fitted on train only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # ── Build GAM ──
        # One spline term per feature
        n_features = X_tr_s.shape[1]
        terms = s(0, n_splines=n_splines)
        for j in range(1, n_features):
            terms += s(j, n_splines=n_splines)

        gam = LinearGAM(terms)

        # Grid-search over smoothing penalty λ (log-spaced)
        lam_grid = np.logspace(-3, 3, 11)
        gam.gridsearch(
            X_tr_s, y_tr,
            lam=lam_grid,
            progress=False,
        )

        # ── Predict ──
        y_pred = gam.predict(X_te_s)

        out = pd.DataFrame({
            DATE_COL:   test_month,
            STOCK_COL:  df_te[STOCK_COL].values,
            "sector":   df_te[SECTOR_COL].values,
            "y_true":   y_te,
            "y_pred":   y_pred,
            "fold":     i,
        })
        results.append(out)

        if verbose:
            ic = spearmanr(y_te, y_pred).statistic
            print(
                f"fold {i:3d} | {str(test_month)[:7]} | "
                f"n_train={len(df_tr):6,d} | n_test={len(df_te):4d} | IC={ic:+.4f}"
            )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_monthly_ic(results: pd.DataFrame) -> pd.Series:
    monthly = results.groupby(DATE_COL).apply(
        lambda g: spearmanr(g["y_true"], g["y_pred"]).statistic
        if len(g) > 5 else np.nan
    )
    return monthly.dropna()


def compute_long_short_returns(
    results: pd.DataFrame,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
) -> np.ndarray:
    monthly_rets = []
    for _, g in results.groupby(DATE_COL):
        g = g.dropna(subset=["y_pred", "y_true"]).copy()
        if len(g) < 10:
            continue
        n_long  = max(1, int(len(g) * top_q))
        n_short = max(1, int(len(g) * bottom_q))
        g = g.sort_values("y_pred", ascending=False)
        long_ret  = g.head(n_long)["y_true"].mean()
        short_ret = g.tail(n_short)["y_true"].mean()
        monthly_rets.append(long_ret - short_ret)
    return np.asarray(monthly_rets, dtype=float)


def compute_long_short_sharpe(results: pd.DataFrame) -> float:
    monthly_rets = compute_long_short_returns(results)
    if len(monthly_rets) < 2:
        return np.nan
    mean_ret = monthly_rets.mean()
    std_ret  = monthly_rets.std(ddof=1)
    if std_ret < 1e-8:
        return np.nan
    return float(np.sqrt(12.0) * mean_ret / std_ret)


def summarise(results: pd.DataFrame) -> dict:
    ic = compute_monthly_ic(results)

    if len(ic) >= 2 and ic.std() > 1e-8:
        t_val = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)) + 1e-8)
    else:
        t_val = np.nan

    ls_rets = compute_long_short_returns(results)
    ls_sharpe = compute_long_short_sharpe(results)
    ls_mean   = float(ls_rets.mean()) if len(ls_rets) else np.nan
    pred_std_ratio = float(
        results["y_pred"].std() / (results["y_true"].std() + 1e-8)
    )

    return {
        "model":         "GAM (natural splines)",
        "IC_mean":       round(float(ic.mean()), 4) if len(ic) else np.nan,
        "IC_std":        round(float(ic.std(ddof=1)), 4) if len(ic) > 1 else np.nan,
        "IC_t":          round(float(t_val), 3) if pd.notna(t_val) else np.nan,
        "IC_pos_frac":   round(float((ic > 0).mean()), 3) if len(ic) else np.nan,
        "LS_mean_ret":   round(ls_mean, 5),
        "LS_Sharpe":     round(ls_sharpe, 3),
        "pred_std_ratio": round(pred_std_ratio, 4),
        "n_months":      int(len(ic)),
    }


# ── Partial dependence plots ─────────────────────────────────────────────

def plot_partial_dependence(
    df: pd.DataFrame,
    n_splines: int = 10,
    top_k: int = 3,
    save_path: str = "output/gam_partial_dependence.png",
):
    """
    Fit one final GAM on the full dataset, then plot the learned spline
    curves for the top_k most important features (by absolute partial effect range).
    """
    print("\nFitting full-sample GAM for partial dependence plots...")

    X = np.nan_to_num(df[ALL_FEATURE_COLS].values.astype(np.float64), nan=0.0)
    y = df[TARGET_COL].values.astype(np.float64)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    n_features = X_s.shape[1]
    terms = s(0, n_splines=n_splines)
    for j in range(1, n_features):
        terms += s(j, n_splines=n_splines)

    gam = LinearGAM(terms)
    lam_grid = np.logspace(-3, 3, 11)
    gam.gridsearch(X_s, y, lam=lam_grid, progress=False)

    # Compute importance = range of partial dependence for each feature
    importances = []
    for j in range(n_features):
        XX = gam.generate_X_grid(term=int(j), n=200)
        pdep, _ = gam.partial_dependence(term=int(j), X=XX, width=0.95)
        importances.append(pdep.max() - pdep.min())

    importances = np.array(importances)
    top_indices = np.argsort(importances)[::-1][:top_k]

    fig, axes = plt.subplots(1, top_k, figsize=(6 * top_k, 5))
    if top_k == 1:
        axes = [axes]

    for ax, j in zip(axes, top_indices):
        j = int(j)  # pygam requires native Python int
        XX = gam.generate_X_grid(term=j, n=200)
        pdep, conf = gam.partial_dependence(term=j, X=XX, width=0.95)

        ax.plot(XX[:, j], pdep, color="#1f77b4", lw=2)
        ax.fill_between(
            XX[:, j],
            conf[:, 0],
            conf[:, 1],
            alpha=0.2,
            color="#1f77b4",
        )
        ax.set_title(
            f"{ALL_FEATURE_COLS[j]}\n(range={importances[j]:.5f})",
            fontsize=11,
        )
        ax.set_xlabel("Standardized feature value")
        ax.set_ylabel("Partial effect on fwd_ret_1m")
        ax.axhline(0, color="gray", ls="--", lw=0.8)

    fig.suptitle(
        "GAM Partial Dependence — Top 3 Features by Effect Range",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved partial dependence plot -> {save_path}")

    # Print all feature importances sorted
    print("\nFeature importance (partial dependence range):")
    order = np.argsort(importances)[::-1]
    for rank, j in enumerate(order, 1):
        print(f"  {rank:2d}. {ALL_FEATURE_COLS[j]:30s}  range = {importances[j]:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rung 3: GAM with natural cubic splines"
    )
    parser.add_argument(
        "--data", type=str, default="data/master_panel.parquet",
        help="Path to master panel parquet",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--n_splines", type=int, default=10,
        help="Number of spline basis functions per feature (default 10)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress fold-level logging",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading panel...")
    df = pd.read_parquet(args.data)

    needed = [DATE_COL, STOCK_COL, SECTOR_COL, TARGET_COL] + ALL_FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"Rows:           {len(df):,}")
    print(f"Date range:     {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"Unique tickers: {df[STOCK_COL].nunique():,}")
    print(f"Unique months:  {df[DATE_COL].nunique():,}")
    print(f"Features:       {len(ALL_FEATURE_COLS)}")
    print(f"Splines/feature:{args.n_splines}")

    # ── Walk-forward ──
    print(f"\n{'=' * 60}")
    print("WALK-FORWARD EVALUATION — GAM (natural splines)")
    print(f"{'=' * 60}")

    results = walk_forward_evaluate(
        df=df,
        n_splines=args.n_splines,
        verbose=not args.quiet,
    )

    if results.empty:
        print("ERROR: no results produced.")
        return

    # ── Save results ──
    results_path = output_dir / "results_rung3_gam.parquet"
    results.to_parquet(results_path, index=False)
    print(f"\nSaved results -> {results_path}")

    # ── Summary ──
    summary = summarise(results)

    print(f"\n{'=' * 60}")
    print("GAM SUMMARY")
    print(f"{'=' * 60}")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    # Per-month IC table
    ic_series = compute_monthly_ic(results)
    ic_df = ic_series.reset_index()
    ic_df.columns = [DATE_COL, "IC"]
    print(f"\nMonthly IC distribution:")
    print(ic_df["IC"].describe().round(4).to_string())

    # Save summary CSV
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "gam_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    # ── Partial dependence plots ──
    plot_partial_dependence(
        df=df,
        n_splines=args.n_splines,
        top_k=3,
        save_path=str(output_dir / "gam_partial_dependence.png"),
    )

    print(f"\n{'=' * 60}")
    print("DONE — Rung 3 GAM complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
