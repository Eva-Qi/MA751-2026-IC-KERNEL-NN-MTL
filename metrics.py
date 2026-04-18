"""
Shared evaluation metrics for MA751 cross-sectional return prediction.

All rungs use the same metric definitions to ensure comparable results.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import DATE_COL


def compute_monthly_ic(
    results: pd.DataFrame,
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    min_stocks: int = 5,
) -> pd.Series:
    """Monthly cross-sectional Spearman rank IC."""
    monthly = results.groupby(DATE_COL).apply(
        lambda g: spearmanr(g[true_col], g[pred_col]).statistic
        if len(g) > min_stocks else np.nan
    )
    return monthly.dropna()


def compute_long_short_sharpe(
    results: pd.DataFrame,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
) -> float:
    """
    Annualized Sharpe of equal-weight long-short portfolio.
    Long top quintile, short bottom quintile by predicted return.
    """
    monthly_rets = []

    for _, g in results.groupby(DATE_COL):
        g = g.dropna(subset=["y_pred", "y_true"]).copy()
        if len(g) < 10:
            continue

        n_long = max(1, int(len(g) * top_q))
        n_short = max(1, int(len(g) * bottom_q))

        g = g.sort_values("y_pred", ascending=False)
        long_ret = g.head(n_long)["y_true"].mean()
        short_ret = g.tail(n_short)["y_true"].mean()
        monthly_rets.append(long_ret - short_ret)

    if len(monthly_rets) < 2:
        return np.nan

    arr = np.asarray(monthly_rets, dtype=float)
    std = arr.std(ddof=1)
    if std < 1e-8:
        return np.nan
    return float(np.sqrt(12.0) * arr.mean() / std)


def compute_ic_ir(ic_series: pd.Series) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    if len(ic_series) < 2 or ic_series.std() < 1e-8:
        return np.nan
    return float(ic_series.mean() / ic_series.std())


def compute_hit_rate(ic_series: pd.Series) -> float:
    """Fraction of months with positive IC."""
    if len(ic_series) == 0:
        return np.nan
    return float((ic_series > 0).mean())


def compute_auxiliary_ic(
    results: pd.DataFrame,
    true_col: str = "fwd_ret_3m_true",
    pred_col: str = "ret3m_pred",
) -> dict:
    """IC of an auxiliary head (diagnostic only)."""
    if pred_col not in results.columns:
        return {"ic_mean": np.nan, "ic_std": np.nan}

    sub = results.dropna(subset=[true_col, pred_col])
    if len(sub) == 0:
        return {"ic_mean": np.nan, "ic_std": np.nan}

    monthly = sub.groupby(DATE_COL).apply(
        lambda g: spearmanr(g[true_col], g[pred_col]).statistic
        if len(g) > 5 else np.nan
    ).dropna()

    return {
        "ic_mean": round(float(monthly.mean()), 4) if len(monthly) else np.nan,
        "ic_std": round(float(monthly.std()), 4) if len(monthly) else np.nan,
    }


def summarise(results: pd.DataFrame, label: str = "") -> dict:
    """Compute all summary metrics for a model's walk-forward results."""
    ic = compute_monthly_ic(results)

    if len(ic) >= 2 and ic.std() > 1e-8:
        t_val = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)) + 1e-8)
    else:
        t_val = np.nan

    # Auxiliary heads (if present)
    aux3m = compute_auxiliary_ic(results, "fwd_ret_3m_true", "ret3m_pred")

    vol_corr = np.nan
    if "vol_pred" in results.columns and "realized_vol_true" in results.columns:
        sub = results.dropna(subset=["realized_vol_true", "vol_pred"])
        if len(sub) > 0:
            vol_corr = round(float(sub["realized_vol_true"].corr(sub["vol_pred"])), 4)

    pred_std_ratio = float(
        results["y_pred"].std() / (results["y_true"].std() + 1e-8)
    )

    row = {
        "label": label,
        "n_months": len(ic),
        "IC_mean": round(float(ic.mean()), 4) if len(ic) else np.nan,
        "IC_std": round(float(ic.std(ddof=1)), 4) if len(ic) > 1 else np.nan,
        "IC_t": round(float(t_val), 3) if pd.notna(t_val) else np.nan,
        "IC_IR": round(compute_ic_ir(ic), 4),
        "hit_rate": round(compute_hit_rate(ic), 3),
        "LS_Sharpe": round(compute_long_short_sharpe(results), 3),
        "ret3m_ic_mean": aux3m["ic_mean"],
        "ret3m_ic_std": aux3m["ic_std"],
        "vol_corr": vol_corr,
        "pred_std_ratio": round(pred_std_ratio, 3),
    }

    return row
