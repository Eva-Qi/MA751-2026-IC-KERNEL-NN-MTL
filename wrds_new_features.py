"""
Build 8 new features from WRDS Tier 1 + Tier 2 data.

Features:
  1. SUE (Standardized Unexpected Earnings) — from IBES surprises
  2. AnalystRevision — month-over-month consensus EPS change
  3. AnalystDispersion — std / |mean| of EPS estimates
  4. RevisionBreadth — (upgrades - downgrades) / total analysts
  5. Beta — 252-day rolling CAPM beta (pre-computed)
  6. IVOL — 60-day idiosyncratic volatility (pre-computed)
  7. ShortInterestRatio — short interest / shares outstanding
  8. InstOwnershipChg — QoQ change in institutional ownership %

Output: data/wrds/wrds_new_features_monthly.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

WRDS = Path("data/wrds")


def _load_ibes_crsp_link() -> pd.DataFrame:
    """Load IBES ticker → CRSP PERMNO mapping with date validity."""
    link = pd.read_parquet(WRDS / "ibes_crsp_link.parquet")
    link["sdate"] = pd.to_datetime(link["sdate"])
    link["edate"] = pd.to_datetime(link["edate"])
    return link


def _map_ibes_to_permno(ibes_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Map IBES ticker to CRSP permno, respecting date validity windows."""
    link = _load_ibes_crsp_link()

    # Rename for join
    ibes_col = "ticker" if "ticker" in ibes_df.columns else "ibes_ticker"
    merged = ibes_df.merge(
        link, left_on=ibes_col, right_on="ibes_ticker", how="inner"
    )

    # Filter: date must be within link validity window
    merged[date_col] = pd.to_datetime(merged[date_col])
    mask = (merged[date_col] >= merged["sdate"]) & (merged[date_col] <= merged["edate"])
    return merged[mask].drop(columns=["ibes_ticker", "sdate", "edate"], errors="ignore")


def build_sue() -> pd.DataFrame:
    """
    SUE: Standardized Unexpected Earnings.
    Use surpmean from ibes_surprises (already standardized).
    For each permno-month, take the most recent surprise BEFORE month-end.
    """
    print("  Building SUE...")
    surp = pd.read_parquet(WRDS / "ibes_surprises.parquet")
    surp["anndats"] = pd.to_datetime(surp["anndats"])

    # Map to permno
    surp = _map_ibes_to_permno(surp, "anndats")

    # Snap to month-end
    surp["date"] = surp["anndats"] + pd.offsets.MonthEnd(0)

    # For each permno-month, take the latest surprise
    surp = surp.sort_values(["permno", "anndats"])
    monthly = surp.groupby(["permno", "date"]).last().reset_index()

    return monthly[["permno", "date", "surpmean"]].rename(columns={"surpmean": "SUE"})


def build_analyst_features() -> pd.DataFrame:
    """
    From IBES consensus:
    - AnalystRevision: meanest(t) - meanest(t-3m), using fpi='1'
    - AnalystDispersion: stdev / |meanest|, where numest >= 3
    - RevisionBreadth: (numup - numdown) / numest
    """
    print("  Building analyst features (revision, dispersion, breadth)...")
    cons = pd.read_parquet(WRDS / "ibes_consensus.parquet")
    cons["statpers"] = pd.to_datetime(cons["statpers"])

    # Filter: current fiscal year only
    cons = cons[cons["fpi"] == "1"].copy()

    # Map to permno
    cons = _map_ibes_to_permno(cons, "statpers")

    # Snap to month-end
    cons["date"] = cons["statpers"] + pd.offsets.MonthEnd(0)

    # For each permno-month, take the latest consensus
    cons = cons.sort_values(["permno", "statpers"])
    monthly = cons.groupby(["permno", "date"]).last().reset_index()

    # AnalystDispersion = stdev / |meanest| (only where numest >= 3)
    monthly["AnalystDispersion"] = np.where(
        (monthly["numest"] >= 3) & (monthly["meanest"].abs() > 1e-6),
        monthly["stdev"] / monthly["meanest"].abs(),
        np.nan,
    )

    # RevisionBreadth = (numup - numdown) / numest
    monthly["RevisionBreadth"] = np.where(
        monthly["numest"] > 0,
        (monthly["numup"] - monthly["numdown"]) / monthly["numest"],
        np.nan,
    )

    # AnalystRevision = change in meanest over 3 months
    monthly = monthly.sort_values(["permno", "date"])
    monthly["meanest_lag3"] = monthly.groupby("permno")["meanest"].shift(3)
    monthly["AnalystRevision"] = monthly["meanest"] - monthly["meanest_lag3"]

    cols = ["permno", "date", "AnalystRevision", "AnalystDispersion", "RevisionBreadth"]
    return monthly[cols]


def build_beta_ivol() -> pd.DataFrame:
    """Load pre-computed beta and IVOL."""
    print("  Loading beta and IVOL...")
    beta = pd.read_parquet(WRDS / "stock_beta_monthly.parquet")
    ivol = pd.read_parquet(WRDS / "stock_ivol_monthly.parquet")

    beta["date"] = pd.to_datetime(beta["date"])
    ivol["date"] = pd.to_datetime(ivol["date"])

    merged = beta.merge(ivol, on=["permno", "date"], how="outer")
    merged = merged.rename(columns={"beta_252d": "Beta", "ivol_60d": "IVOL"})
    return merged[["permno", "date", "Beta", "IVOL"]]


def build_short_interest() -> pd.DataFrame:
    """
    ShortInterestRatio = shortint / (shrout * 1000)
    Join short_interest (gvkey) → ccm_link → permno → crsp (shrout).
    """
    print("  Building short interest ratio...")
    short = pd.read_parquet(WRDS / "short_interest.parquet")
    short["datadate"] = pd.to_datetime(short["datadate"])

    ccm = pd.read_parquet(WRDS / "ccm_link.parquet")
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])

    # Join short interest to permno via ccm
    short = short.merge(ccm[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="inner")
    mask = (short["datadate"] >= short["linkdt"]) & (
        short["linkenddt"].isna() | (short["datadate"] <= short["linkenddt"])
    )
    short = short[mask].copy()

    # Snap to month-end
    short["date"] = short["datadate"] + pd.offsets.MonthEnd(0)

    # Take latest short interest per permno-month
    short = short.sort_values(["permno", "datadate"])
    monthly_short = short.groupby(["permno", "date"])["shortint"].last().reset_index()

    # Get CRSP shares outstanding
    crsp = pd.read_parquet(WRDS / "crsp_monthly.parquet")[["permno", "date", "shrout"]]
    crsp["date"] = pd.to_datetime(crsp["date"])

    merged = monthly_short.merge(crsp, on=["permno", "date"], how="inner")
    merged["ShortInterestRatio"] = np.where(
        merged["shrout"] > 0,
        merged["shortint"] / (merged["shrout"] * 1000),
        np.nan,
    )

    return merged[["permno", "date", "ShortInterestRatio"]]


def build_inst_ownership() -> pd.DataFrame:
    """
    InstOwnershipChg = QoQ change in (total_inst_shares / shares_outstanding).
    inst_ownership uses CUSIP; join via compustat.cusip → gvkey → ccm → permno.
    """
    print("  Building institutional ownership change...")
    inst = pd.read_parquet(WRDS / "inst_ownership_13f.parquet")
    inst["rdate"] = pd.to_datetime(inst["rdate"])

    # Join to permno via: cusip → compustat → gvkey → ccm
    comp = pd.read_parquet(WRDS / "compustat_annual.parquet")[["gvkey", "cusip"]].drop_duplicates()
    # CUSIP: inst uses 8-char, compustat uses 9-char. Match on first 8.
    comp["cusip8"] = comp["cusip"].str[:8]
    inst["cusip8"] = inst["cusip"].str[:8]

    inst = inst.merge(comp[["gvkey", "cusip8"]].drop_duplicates(), on="cusip8", how="inner")

    ccm = pd.read_parquet(WRDS / "ccm_link.parquet")
    ccm = ccm[ccm["linktype"].isin(["LU", "LC"]) & ccm["linkprim"].isin(["P", "C"])]
    inst = inst.merge(ccm[["gvkey", "permno"]].drop_duplicates(), on="gvkey", how="inner")

    # Snap to month-end
    inst["date"] = inst["rdate"] + pd.offsets.MonthEnd(0)

    # Get CRSP shares outstanding
    crsp = pd.read_parquet(WRDS / "crsp_monthly.parquet")[["permno", "date", "shrout"]]
    crsp["date"] = pd.to_datetime(crsp["date"])

    # Aggregate per permno-quarter (in case multiple cusip matches)
    inst_agg = inst.groupby(["permno", "date"]).agg(
        total_inst_shares=("total_inst_shares", "sum"),
        num_inst=("num_inst", "max"),
    ).reset_index()

    merged = inst_agg.merge(crsp, on=["permno", "date"], how="inner")
    merged["inst_pct"] = np.where(
        merged["shrout"] > 0,
        merged["total_inst_shares"] / (merged["shrout"] * 1000),
        np.nan,
    )

    # Forward-fill quarterly to monthly, then compute QoQ change
    # First, build monthly date index per permno
    all_dates = crsp[["permno", "date"]].drop_duplicates()
    merged = all_dates.merge(
        merged[["permno", "date", "inst_pct"]], on=["permno", "date"], how="left"
    )
    merged = merged.sort_values(["permno", "date"])
    merged["inst_pct"] = merged.groupby("permno")["inst_pct"].ffill()

    # QoQ change (3-month lag)
    merged["inst_pct_lag3"] = merged.groupby("permno")["inst_pct"].shift(3)
    merged["InstOwnershipChg"] = merged["inst_pct"] - merged["inst_pct_lag3"]

    return merged[["permno", "date", "InstOwnershipChg"]].dropna(subset=["InstOwnershipChg"])


def build_regime() -> pd.DataFrame:
    """Load pre-computed market regime variables."""
    print("  Loading regime variables...")
    regime = pd.read_parquet(WRDS / "market_regime_monthly.parquet")
    regime["date"] = pd.to_datetime(regime["date"])
    return regime


def build_wrds_new_features(save: bool = True) -> pd.DataFrame:
    """Build all 8 new features + regime, merge on permno+date."""
    print("Building WRDS new features...")

    sue = build_sue()
    analyst = build_analyst_features()
    beta_ivol = build_beta_ivol()
    short = build_short_interest()
    inst = build_inst_ownership()
    regime = build_regime()

    # Start with beta/ivol as the widest base (most permno-dates)
    panel = beta_ivol.copy()

    # Merge analyst features
    panel = panel.merge(sue, on=["permno", "date"], how="left")
    panel = panel.merge(analyst, on=["permno", "date"], how="left")

    # Merge alternative data
    panel = panel.merge(short, on=["permno", "date"], how="left")
    panel = panel.merge(inst, on=["permno", "date"], how="left")

    # Merge regime (date-only join — same for all stocks)
    panel = panel.merge(regime, on="date", how="left")

    # Sort
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    print(f"\nPanel shape: {panel.shape}")
    print(f"Date range: {panel.date.min()} to {panel.date.max()}")
    print(f"Unique permno: {panel.permno.nunique()}")
    print(f"\nNaN rates:")
    feature_cols = [
        "SUE", "AnalystRevision", "AnalystDispersion", "RevisionBreadth",
        "Beta", "IVOL", "ShortInterestRatio", "InstOwnershipChg",
        "mkt_vol_regime", "mkt_trend_regime",
    ]
    for c in feature_cols:
        if c in panel.columns:
            print(f"  {c}: {panel[c].isna().mean():.1%}")

    if save:
        out_path = WRDS / "wrds_new_features_monthly.parquet"
        panel.to_parquet(out_path, index=False)
        print(f"\nSaved -> {out_path}")

    return panel


if __name__ == "__main__":
    build_wrds_new_features(save=True)
