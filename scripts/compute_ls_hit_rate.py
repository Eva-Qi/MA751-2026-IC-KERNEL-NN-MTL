"""
Compute per-month long-short portfolio returns and LS-hit-rate
(fraction of months with LS return > 0) for each model in output/.

Output: output/ls_hit_rate_summary.csv  (one row per model)
        output/ls_monthly_returns.csv   (one row per (model, month))
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("output")

MODELS = {
    "1a_OLS":         "results_1a_OLS_v2.parquet",
    "1b_IC_Ensemble": "results_1b_IC_Ensemble_v2.parquet",
    "2a_LASSO":       "results_2a_LASSO_v2.parquet",
    "2b_Ridge":       "results_2b_Ridge_v2.parquet",
    "3_GAM":          "results_rung3_gam.parquet",
    "5a_MLP":         "archive/results_5a_uw.parquet",
    "5b_MTL_ret3m":   "results_5b_uw.parquet",
    "5c_MTL_vol":     "results_5c_uw.parquet",
    "5d_MTL_ret3m_vol": "results_5d_uw.parquet",
}

DECILES = 10


def ls_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Per-month: top-decile mean y_true minus bottom-decile mean y_true."""
    rows = []
    for date, g in df.groupby("date"):
        g = g.dropna(subset=["y_pred", "y_true"])
        if len(g) < DECILES:
            continue
        q = pd.qcut(g["y_pred"].rank(method="first"), DECILES, labels=False)
        top = g.loc[q == DECILES - 1, "y_true"].mean()
        bot = g.loc[q == 0, "y_true"].mean()
        rows.append({"date": date, "ls_ret": top - bot, "n_stocks": len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    summary_rows = []
    monthly_rows = []

    for label, fname in MODELS.items():
        path = OUT / fname
        if not path.exists():
            print(f"  [skip] {label}: {path} missing")
            continue

        df = pd.read_parquet(path)[["date", "ticker", "y_pred", "y_true"]]
        m = ls_monthly(df)
        m.insert(0, "model", label)
        monthly_rows.append(m)

        n = len(m)
        pos = int((m["ls_ret"] > 0).sum())
        mean = m["ls_ret"].mean()
        std = m["ls_ret"].std()
        sharpe = (mean / std) * np.sqrt(12) if std > 0 else np.nan

        summary_rows.append({
            "model": label,
            "n_months": n,
            "ls_ret_mean": round(mean, 5),
            "ls_ret_std": round(std, 4),
            "ls_sharpe_annualized": round(sharpe, 3),
            "ls_hit_rate": round(pos / n, 3) if n else np.nan,
            "n_pos_months": pos,
            "n_neg_months": n - pos,
        })
        print(f"  {label}: LS-hit {pos}/{n} = {pos/n:.1%}, Sharpe {sharpe:.3f}")

    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_rows, ignore_index=True)

    summary.to_csv(OUT / "ls_hit_rate_summary.csv", index=False)
    monthly.to_csv(OUT / "ls_monthly_returns.csv", index=False)
    print(f"\nWrote {OUT / 'ls_hit_rate_summary.csv'}")
    print(f"Wrote {OUT / 'ls_monthly_returns.csv'}")


if __name__ == "__main__":
    main()
