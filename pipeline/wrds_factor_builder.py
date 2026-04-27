"""
wrds_factor_builder.py
======================
Build 6 academic factors from WRDS Compustat + CRSP data.

Factors:
    1. EarningsYield      = ib / mktcap
    2. GrossProfitability = gp / at
    3. AssetGrowth        = at(t) / at(t-1) - 1
    4. Accruals           = (ib - oancf) / at
    5. Momentum12_1       = cumulative return months t-12 to t-1
    6. NetDebtEBITDA      = (dltt + dlc - che) / oiadp

Point-in-time: Compustat annual data is treated as available 4 months
after fiscal year-end (datadate + 4 months) to avoid look-ahead bias.

CCM link: only LU/LC linktype and P/C linkprim, date-valid links.
"""

import os
import warnings
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "wrds")
OUT_PATH = os.path.join(DATA_DIR, "wrds_factors_monthly.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise division; returns NaN where denom is zero or NaN."""
    result = num / denom
    result[denom == 0] = np.nan
    return result


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------

def load_data(data_dir: str):
    """Load and minimally preprocess the three WRDS source files."""
    print("[1/6] Loading raw data ...")

    comp = pd.read_parquet(os.path.join(data_dir, "compustat_annual.parquet"))
    crsp = pd.read_parquet(os.path.join(data_dir, "crsp_monthly.parquet"))
    link = pd.read_parquet(os.path.join(data_dir, "ccm_link.parquet"))

    # -- Compustat: cast types
    comp["datadate"] = pd.to_datetime(comp["datadate"])
    comp["gvkey"] = comp["gvkey"].astype(str)

    # -- CRSP: ensure float permno, parse date
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["permno"] = crsp["permno"].astype("Int64")

    # -- CCM link: keep only valid link types/prims
    link = link[
        link["linktype"].isin(["LU", "LC"]) &
        link["linkprim"].isin(["P", "C"])
    ].copy()
    link["gvkey"] = link["gvkey"].astype(str)
    link["permno"] = link["permno"].astype("Int64")
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"])

    print(f"    compustat: {len(comp):,} rows | crsp: {len(crsp):,} rows | link: {len(link):,} rows")
    return comp, crsp, link


# ---------------------------------------------------------------------------
# Prepare Compustat fundamentals
# ---------------------------------------------------------------------------

def prepare_compustat(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Add availability_date = datadate + 4 months (reporting lag).
    Sort and keep one record per (gvkey, datadate).
    """
    print("[2/6] Preparing Compustat annual fundamentals ...")

    comp = comp.drop_duplicates(subset=["gvkey", "datadate"]).copy()
    comp["availability_date"] = comp["datadate"] + pd.DateOffset(months=4)
    comp = comp.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

    # -- AssetGrowth: shift at by 1 year within each gvkey
    comp["at_lag1"] = comp.groupby("gvkey")["at"].shift(1)
    comp["AssetGrowth"] = safe_div(comp["at"], comp["at_lag1"]) - 1

    # -- Pre-compute the ratio columns (will be merged onto CRSP universe)
    comp["GrossProfitability"] = safe_div(comp["gp"], comp["at"])
    comp["Accruals"]           = safe_div(comp["ib"] - comp["oancf"], comp["at"])
    comp["NetDebtEBITDA"]      = safe_div(
        comp["dltt"].fillna(0) + comp["dlc"].fillna(0) - comp["che"].fillna(0),
        comp["oiadp"]
    )
    # EarningsYield needs mktcap from CRSP, computed after merge

    keep_cols = [
        "gvkey", "datadate", "availability_date",
        "ib",                  # needed for EarningsYield after merge
        "GrossProfitability", "AssetGrowth", "Accruals", "NetDebtEBITDA"
    ]
    return comp[keep_cols]


# ---------------------------------------------------------------------------
# Build CCM-linked universe at monthly frequency
# ---------------------------------------------------------------------------

def build_crsp_comp_panel(crsp: pd.DataFrame,
                           comp: pd.DataFrame,
                           link: pd.DataFrame) -> pd.DataFrame:
    """
    For every (permno, date) in CRSP, find the matching gvkey from CCM link
    and join the most recent Compustat annual filing available at that date
    (availability_date <= crsp.date).

    Returns a panel indexed by (permno, date) with CRSP market data +
    Compustat fundamental signals.
    """
    print("[3/6] Linking CRSP <-> Compustat via CCM ...")

    # Keep only CRSP columns we need
    crsp_base = crsp[["permno", "date", "ticker", "ret", "mktcap"]].copy()

    # Merge link onto CRSP (many-to-many on permno, then filter by date)
    crsp_link = crsp_base.merge(
        link[["gvkey", "permno", "linkdt", "linkenddt"]],
        on="permno",
        how="inner"
    )

    # Enforce date validity of link
    crsp_link = crsp_link[
        (crsp_link["linkdt"] <= crsp_link["date"]) &
        (crsp_link["linkenddt"].isna() | (crsp_link["linkenddt"] >= crsp_link["date"]))
    ].copy()

    # If multiple gvkeys map to one permno on the same date (rare), keep first
    crsp_link = crsp_link.sort_values(["permno", "date", "gvkey"])
    crsp_link = crsp_link.drop_duplicates(subset=["permno", "date"])

    print(f"    linked panel: {len(crsp_link):,} (permno, date) obs")

    # -----------------------------------------------------------------------
    # Point-in-time join: for each (permno, date), find the most recent
    # Compustat filing where availability_date <= date (same gvkey)
    # -----------------------------------------------------------------------
    print("[4/6] Merging most-recent-available Compustat fundamentals ...")

    # Sort comp for merge_asof
    comp_sorted = comp.sort_values(["gvkey", "availability_date"])

    # merge_asof requires both keys sorted; do per-gvkey via a two-step approach:
    # 1. merge all comp rows onto crsp_link by gvkey
    # 2. filter to availability_date <= date, keep last
    panel = crsp_link.merge(comp_sorted, on="gvkey", how="left")
    panel = panel[panel["availability_date"] <= panel["date"]]

    # Keep only the most recent filing per (permno, date)
    panel = (
        panel
        .sort_values(["permno", "date", "availability_date"])
        .drop_duplicates(subset=["permno", "date"], keep="last")
    )

    print(f"    panel after fundamental merge: {len(panel):,} rows")
    return panel


# ---------------------------------------------------------------------------
# Compute EarningsYield
# ---------------------------------------------------------------------------

def compute_earnings_yield(panel: pd.DataFrame) -> pd.DataFrame:
    """
    EarningsYield = ib / mktcap.

    Unit alignment:
      - Compustat ib  : millions USD
      - CRSP mktcap   : dollars (abs(prc) * shrout * 1000)
    Multiply ib by 1e6 to convert to dollars before dividing.
    """
    panel["EarningsYield"] = safe_div(panel["ib"] * 1e6, panel["mktcap"])
    return panel


# ---------------------------------------------------------------------------
# Compute Momentum (pure CRSP, no Compustat)
# ---------------------------------------------------------------------------

def compute_momentum(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum12_1 = product(1+ret) over months [t-12, t-1] minus 1.

    For each (permno, date), compute using the 12 months ending one month
    prior to the signal date.  Returns a DataFrame with permno, date,
    Momentum12_1 columns.
    """
    print("[5/6] Computing Momentum12_1 ...")

    ret = crsp[["permno", "date", "ret"]].copy()
    ret = ret.sort_values(["permno", "date"]).reset_index(drop=True)

    # Use a rolling 12-period product on log(1+r) for efficiency
    # We shift by 1 so the most recent month is excluded (standard momentum skip)
    ret["log1r"] = np.log1p(ret["ret"].astype(float))

    # Rolling sum of log returns over a 12-month window on the *shifted* series
    # shift(1) within permno pushes most-recent month out; then rolling(12) = t-12 to t-1
    # Vectorized approach: sort entire frame, compute shift+rolling per permno
    ret = ret.sort_values(["permno", "date"]).reset_index(drop=True)

    # shift log1r by 1 within permno (NaN at group boundaries is fine)
    ret["log1r_shift"] = ret.groupby("permno")["log1r"].shift(1)
    # rolling 12-period sum of shifted log returns within permno
    ret["roll12"] = (
        ret.groupby("permno")["log1r_shift"]
           .transform(lambda s: s.rolling(12, min_periods=10).sum())
    )
    ret["Momentum12_1"] = np.expm1(ret["roll12"])

    mom = ret[["permno", "date", "Momentum12_1"]].copy()
    mom = mom.reset_index(drop=True)
    print(f"    momentum rows: {len(mom):,}")
    return mom


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_wrds_factors(save: bool = True) -> pd.DataFrame:
    """
    Build all 6 academic factors at monthly frequency from WRDS data.

    Returns
    -------
    pd.DataFrame
        columns: permno, date, ticker, EarningsYield, GrossProfitability,
                 AssetGrowth, Accruals, Momentum12_1, NetDebtEBITDA
    """
    print("=" * 60)
    print("WRDS Factor Builder — starting")
    print("=" * 60)

    comp_raw, crsp, link = load_data(DATA_DIR)
    comp = prepare_compustat(comp_raw)
    panel = build_crsp_comp_panel(crsp, comp, link)
    panel = compute_earnings_yield(panel)

    # Merge momentum
    mom = compute_momentum(crsp)
    panel = panel.merge(mom, on=["permno", "date"], how="left")

    # Final column selection
    output_cols = [
        "permno", "date", "ticker",
        "EarningsYield", "GrossProfitability", "AssetGrowth",
        "Accruals", "Momentum12_1", "NetDebtEBITDA"
    ]
    # ticker comes from CRSP via the linked panel
    out = panel[output_cols].copy()

    # Sort
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)

    print("\n[6/6] Summary statistics")
    print(f"    Total rows : {len(out):,}")
    print(f"    Unique permno : {out['permno'].nunique():,}")
    print(f"    Date range : {out['date'].min().date()} - {out['date'].max().date()}")

    factor_cols = [
        "EarningsYield", "GrossProfitability", "AssetGrowth",
        "Accruals", "Momentum12_1", "NetDebtEBITDA"
    ]
    print("\n    NaN rates per factor:")
    for col in factor_cols:
        nan_pct = out[col].isna().mean() * 100
        print(f"      {col:<22}: {nan_pct:.1f}% NaN")

    # AAPL sample
    aapl = out[out["ticker"] == "AAPL"].tail(6)
    if len(aapl):
        print("\n    AAPL (last 6 months):")
        print(aapl[["date"] + factor_cols].to_string(index=False))
    else:
        print("\n    AAPL not found in output (check permno/ticker join)")

    if save:
        out.to_parquet(OUT_PATH, index=False)
        print(f"\n    Saved -> {OUT_PATH}")

    print("=" * 60)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_wrds_factors(save=True)
