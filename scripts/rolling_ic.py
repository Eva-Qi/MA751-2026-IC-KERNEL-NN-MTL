"""
Rolling 12-month IC per feature.

For each V2 feature:
  1. Monthly cross-sectional Spearman(feature, fwd_ret_1m).
  2. Rolling 12-month mean of that series.
  3. Plot as line chart; one subplot per feature.

Outputs:
  output/rolling_ic_monthly.csv   (date x feature, raw monthly IC)
  output/rolling_ic_12m.csv       (date x feature, rolling 12-mo mean)
  output/rolling_ic_grid.png      (4x4 grid of feature plots)
  output/rolling_ic_overlay.png   (all 14 on one axes, for regime visibility)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import ALL_FEATURE_COLS_V2, TARGET_COL, DATE_COL, STOCK_COL

DATA = Path("data/master_panel_v2.parquet")
OUT = Path("output")
WINDOW = 12


def monthly_ic(df: pd.DataFrame, feat: str) -> pd.Series:
    rows = []
    for date, g in df.groupby(DATE_COL):
        g = g[[feat, TARGET_COL]].dropna()
        if len(g) < 20:
            continue
        rho = spearmanr(g[feat], g[TARGET_COL]).statistic
        rows.append((date, rho))
    return pd.Series(dict(rows)).sort_index()


def main():
    print(f"Loading {DATA}")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    feats = [f for f in ALL_FEATURE_COLS_V2 if f in df.columns]
    print(f"Features: {len(feats)}")

    ic_tbl = {}
    for f in feats:
        s = monthly_ic(df, f)
        ic_tbl[f.replace("_zscore", "")] = s
        print(f"  {f:40s} mean IC = {s.mean():+.4f}  ({len(s)} months)")

    ic_df = pd.DataFrame(ic_tbl).sort_index()
    roll = ic_df.rolling(WINDOW, min_periods=WINDOW).mean()
    # Drop features with no valid rolling values (e.g. too-short history)
    empty = [c for c in roll.columns if roll[c].dropna().empty]
    if empty:
        print(f"Dropping features with no valid rolling IC: {empty}")
        roll = roll.drop(columns=empty)

    ic_df.to_csv(OUT / "rolling_ic_monthly.csv")
    roll.to_csv(OUT / "rolling_ic_12m.csv")

    # Grid
    n = len(feats)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3 * nrow), sharex=True)
    axes = axes.ravel()
    for i, col in enumerate(roll.columns):
        ax = axes[i]
        ax.plot(roll.index, roll[col], lw=1.3)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_title(col, fontsize=10)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelbottom=True, labelrotation=45, labelsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Rolling {WINDOW}-month IC per feature", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "rolling_ic_grid.png", dpi=130, bbox_inches="tight")

    # Overlay
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    for col in roll.columns:
        ax2.plot(roll.index, roll[col], lw=1.1, alpha=0.75, label=col)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_title(f"Rolling {WINDOW}-month IC — all features")
    ax2.set_ylabel("IC (12-mo rolling mean)")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(fontsize=7, ncol=2, loc="best")
    fig2.tight_layout()
    fig2.savefig(OUT / "rolling_ic_overlay.png", dpi=130, bbox_inches="tight")

    print(f"\nWrote:")
    print(f"  {OUT/'rolling_ic_monthly.csv'}")
    print(f"  {OUT/'rolling_ic_12m.csv'}")
    print(f"  {OUT/'rolling_ic_grid.png'}")
    print(f"  {OUT/'rolling_ic_overlay.png'}")


if __name__ == "__main__":
    main()
