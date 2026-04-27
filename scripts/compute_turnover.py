"""
Compute portfolio turnover and cost-adjusted Sharpe ratio for each model.

For each model results parquet:
  - Long leg  = top 20% by y_pred each month
  - Short leg = bottom 20% by y_pred each month
  - Turnover  = |symmetric_difference| / |union|  (per leg, averaged across both legs)
  - Monthly cost = turnover × 10 bps × 2  (round-trip, both legs)
  - Net return = gross L-S return − monthly cost
  - Net Sharpe = sqrt(12) × mean(net) / std(net)

Output: output/turnover_summary.csv
        columns: model, avg_turnover, gross_sharpe, net_sharpe, annual_cost_bps, n_months
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── repo root on sys.path so we can import config ────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATE_COL, STOCK_COL  # noqa: E402

OUT = ROOT / "output"

# ── Cost assumption ──────────────────────────────────────────────────────────
ROUND_TRIP_BPS = 10          # bps per leg (round-trip)
COST_BPS_PER_UNIT = ROUND_TRIP_BPS * 2   # long + short legs combined
COST_DECIMAL = COST_BPS_PER_UNIT / 10_000   # as a return fraction per turnover unit

# ── 20% quantile cut-offs ───────────────────────────────────────────────────
LONG_QUANTILE  = 0.80   # top 20%
SHORT_QUANTILE = 0.20   # bottom 20%

# ── Model files ─────────────────────────────────────────────────────────────
MODELS = {
    "1a_OLS":              "results_1a_OLS_v2.parquet",
    "1b_IC_Ensemble":      "results_1b_IC_Ensemble_v2.parquet",
    "2a_LASSO":            "results_2a_LASSO_v2.parquet",
    "2b_Ridge":            "results_2b_Ridge_v2.parquet",
    "2c_OLS_LASSO3":       "archive/results_2c_OLS_LASSO3.parquet",
    "2c_OLS_LASSO5":       "archive/results_2c_OLS_LASSO5.parquet",
    "3_GAM":               "results_rung3_gam.parquet",
    "5a_MLP":              "archive/results_5a_uw.parquet",
    "5b_MTL_ret3m":        "results_5b_uw.parquet",
    "5c_MTL_vol":          "results_5c_uw.parquet",
    "5d_MTL_ret3m_vol":    "results_5d_uw.parquet",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def long_short_sets(df: pd.DataFrame) -> tuple[set, set]:
    """Return (long_tickers, short_tickers) using top/bottom 20% by y_pred."""
    df = df.dropna(subset=["y_pred"])
    n = len(df)
    if n == 0:
        return set(), set()
    q_lo = df["y_pred"].quantile(SHORT_QUANTILE)
    q_hi = df["y_pred"].quantile(LONG_QUANTILE)
    longs  = set(df.loc[df["y_pred"] >= q_hi, STOCK_COL])
    shorts = set(df.loc[df["y_pred"] <= q_lo, STOCK_COL])
    return longs, shorts


def leg_turnover(prev: set, curr: set) -> float:
    """Turnover for one leg: |sym_diff| / |union|.  Returns nan if both empty."""
    union = prev | curr
    if not union:
        return np.nan
    return len(prev.symmetric_difference(curr)) / len(union)


def compute_turnover(df: pd.DataFrame) -> tuple[list[float], list[float], list[float]]:
    """
    Walk through months chronologically; compute per-month turnover.

    Returns:
        dates       – list of date values (starting from 2nd month)
        turnovers   – list of per-month average turnover (long + short legs)
        ls_rets     – list of gross L-S returns aligned with turnover months
    """
    dates_sorted = sorted(df[DATE_COL].unique())
    prev_long:  set = set()
    prev_short: set = set()

    turnover_list: list[float] = []
    date_list:     list        = []

    for i, date in enumerate(dates_sorted):
        g = df[df[DATE_COL] == date]
        curr_long, curr_short = long_short_sets(g)

        if i > 0:  # no previous month on first observation
            to_long  = leg_turnover(prev_long,  curr_long)
            to_short = leg_turnover(prev_short, curr_short)
            valid = [v for v in (to_long, to_short) if not np.isnan(v)]
            avg_to = np.mean(valid) if valid else np.nan
            turnover_list.append(avg_to)
            date_list.append(date)

        prev_long  = curr_long
        prev_short = curr_short

    return date_list, turnover_list


def gross_ls_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-month gross L-S returns (top 20% minus bottom 20% y_true).
    Returns DataFrame with columns [DATE_COL, 'ls_ret'].
    """
    rows = []
    for date, g in df.groupby(DATE_COL):
        g = g.dropna(subset=["y_pred", "y_true"])
        if len(g) < 10:
            continue
        q_lo = g["y_pred"].quantile(SHORT_QUANTILE)
        q_hi = g["y_pred"].quantile(LONG_QUANTILE)
        top = g.loc[g["y_pred"] >= q_hi, "y_true"].mean()
        bot = g.loc[g["y_pred"] <= q_lo, "y_true"].mean()
        rows.append({DATE_COL: date, "ls_ret": top - bot})
    return pd.DataFrame(rows).set_index(DATE_COL)


def sharpe_annualized(returns: np.ndarray) -> float:
    """Annualized Sharpe (sqrt-12 scaling, monthly returns)."""
    s = np.std(returns, ddof=1)
    if s == 0 or np.isnan(s):
        return np.nan
    return np.sqrt(12) * np.mean(returns) / s


# ── Main ─────────────────────────────────────────────────────────────────────

def process_model(label: str, fname: str) -> dict | None:
    path = OUT / fname
    if not path.exists():
        print(f"  [skip] {label}: {path.name} not found")
        return None

    df = pd.read_parquet(path)

    # Validate required columns
    required = {DATE_COL, STOCK_COL, "y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [skip] {label}: missing columns {missing}")
        return None

    # Gross L-S returns (all months)
    ls_df = gross_ls_returns(df)   # indexed by date

    # Turnover (aligned to 2nd month onward)
    date_list, to_list = compute_turnover(df)

    if len(date_list) == 0:
        print(f"  [skip] {label}: insufficient months for turnover")
        return None

    # Align: only keep months that appear in both ls_df and date_list
    to_series = pd.Series(to_list, index=pd.Index(date_list, name=DATE_COL))
    aligned = pd.concat([ls_df["ls_ret"], to_series.rename("turnover")], axis=1).dropna()

    n_months    = len(aligned)
    avg_turnover = aligned["turnover"].mean()

    # Gross Sharpe (use all months in ls_df, including first)
    all_ls = ls_df["ls_ret"].dropna().values
    gross_sharpe = sharpe_annualized(all_ls)

    # Monthly cost = avg_turnover × cost_decimal
    monthly_cost = aligned["turnover"] * COST_DECIMAL
    net_returns  = aligned["ls_ret"] - monthly_cost
    net_sharpe   = sharpe_annualized(net_returns.values)

    annual_cost_bps = avg_turnover * COST_BPS_PER_UNIT * 12   # annualized

    print(
        f"  {label:<22}  gross Sharpe {gross_sharpe:+.3f}  "
        f"turnover {avg_turnover:.1%}  net Sharpe {net_sharpe:+.3f}  "
        f"annual cost {annual_cost_bps:.1f} bps  ({n_months} months)"
    )

    return {
        "model":           label,
        "avg_turnover":    round(avg_turnover, 4),
        "gross_sharpe":    round(gross_sharpe, 3),
        "net_sharpe":      round(net_sharpe, 3),
        "annual_cost_bps": round(annual_cost_bps, 2),
        "n_months":        n_months,
    }


def main() -> None:
    print("=" * 70)
    print("Portfolio Turnover & Cost-Adjusted Sharpe")
    print(f"  Assumption: {ROUND_TRIP_BPS} bps round-trip per leg (L+S = {COST_BPS_PER_UNIT} bps)")
    print("=" * 70)

    rows = []
    for label, fname in MODELS.items():
        result = process_model(label, fname)
        if result is not None:
            rows.append(result)

    if not rows:
        print("\nNo models processed — nothing to save.")
        return

    summary = pd.DataFrame(rows)
    out_path = OUT / "turnover_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # ── Pretty summary table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Model':<24} {'Gross Sharpe':>13} {'Turnover%':>11} {'Net Sharpe':>11}")
    print("-" * 70)
    for r in rows:
        print(
            f"  {r['model']:<22}  {r['gross_sharpe']:>+10.3f}  "
            f"{r['avg_turnover']:>9.1%}  {r['net_sharpe']:>+10.3f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
