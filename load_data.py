"""
Master panel construction for MA751 cross-sectional return prediction.

Supports two modes:
  - V1 (legacy): XBRL factors + yfinance prices (15 features)
  - V2 (WRDS):   Compustat/CRSP factors + IBES/alt data (25 features)

Usage:
  python load_data.py           # build V2 (default)
  python load_data.py --v1      # build V1 (legacy)
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    FACTOR_ZSCORE_COLS, FACTOR_RAW_COLS, MACRO_COLS,
    ALL_FEATURE_COLS_V1, ALL_FEATURE_COLS_V2,
    TARGET_COL, DATE_COL, STOCK_COL, SECTOR_COL,
    TYPE_A_SECTORS, VOL_COL, RET3M_COL,
    ANALYST_COLS, RISK_COLS, ALT_DATA_COLS, REGIME_COLS,
)

# ── Paths ────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data/raw/")
WRDS_DIR   = Path("data/wrds/")
OUTPUT_DIR = Path("data/")

# Legacy (V1) paths
FACTOR_PANEL   = DATA_DIR / "factor_panel.parquet"
SPLIT_HISTORY  = DATA_DIR / "split_history.parquet"
PRICES_CACHE   = DATA_DIR / "prices_cache.parquet"
MACRO_FEATURES = DATA_DIR / "macro_features.parquet"
SECTOR_MAP     = DATA_DIR / "sector_map.json"

# WRDS (V2) paths
WRDS_FACTORS      = WRDS_DIR / "wrds_factors_monthly.parquet"
WRDS_NEW_FEATURES = WRDS_DIR / "wrds_new_features_monthly.parquet"
CRSP_MONTHLY      = WRDS_DIR / "crsp_monthly.parquet"

OUTPUT_V1 = OUTPUT_DIR / "master_panel.parquet"
OUTPUT_V2 = OUTPUT_DIR / "master_panel_v2.parquet"

# New feature raw columns (before z-scoring)
NEW_RAW_COLS = [
    "SUE", "AnalystRevision", "AnalystDispersion", "RevisionBreadth",
    "Beta", "IVOL", "ShortInterestRatio", "InstOwnershipChg",
]

NEW_ZSCORE_MAP = {
    "SUE": "SUE_zscore",
    "AnalystRevision": "AnalystRevision_zscore",
    "AnalystDispersion": "AnalystDispersion_zscore",
    "RevisionBreadth": "RevisionBreadth_zscore",
    "Beta": "Beta_zscore",
    "IVOL": "IVOL_zscore",
    "ShortInterestRatio": "ShortInterestRatio_zscore",
    "InstOwnershipChg": "InstOwnershipChg_zscore",
}


# ── Cross-sectional z-score ─────────────────────────────────────────────

def _cs_zscore(g):
    mu = g.mean()
    sigma = g.std()
    return (g - mu) / sigma if sigma > 1e-8 else g * 0.0


# ── V1 legacy builders (yfinance + XBRL) ────────────────────────────────

def build_monthly_returns_v1(prices_path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)
    monthly = prices.resample("M").last()
    fwd = monthly.pct_change(1).shift(-1)
    return (
        fwd.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="fwd_ret_1m")
        .dropna(subset=["fwd_ret_1m"])
    )


def build_fwd_ret_3m_v1(prices_path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)
    monthly = prices.resample("M").last()
    fwd3 = monthly.pct_change(3).shift(-3)
    return (
        fwd3.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="fwd_ret_3m")
    )


def build_realized_vol_v1(prices_path: Path, month_ends, window=21):
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)
    rv = prices.pct_change().rolling(window).std()
    rv_monthly = rv.reindex(month_ends, method="ffill")
    return (
        rv_monthly.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="realized_vol")
        .dropna(subset=["realized_vol"])
    )


def load_factors_v1(factor_path, month_ends):
    fp = pd.read_parquet(factor_path)
    fp["signal_date"] = pd.DatetimeIndex(fp["signal_date"])

    if "EarningsYield" in fp.columns:
        fp = fp[fp["EarningsYield"].abs() <= 1.0]
    if "AssetGrowth" in fp.columns:
        fp = fp[fp["AssetGrowth"].abs() <= 10.0]

    # Split contamination fix (C2)
    try:
        split_hist = pd.read_parquet(SPLIT_HISTORY)
        split_hist["date"] = pd.to_datetime(split_hist["date"], utc=True).dt.tz_localize(None)
        major = split_hist[split_hist["ratio"] >= 2.0][["ticker", "date"]].copy()
        earliest_split = major.groupby("ticker")["date"].min().reset_index()
        earliest_split.columns = ["ticker", "first_split_date"]
        if "EarningsYield" in fp.columns:
            fp = fp.merge(earliest_split, on="ticker", how="left")
            contaminated = fp["first_split_date"].notna() & (fp["signal_date"] < fp["first_split_date"])
            fp.loc[contaminated, "EarningsYield"] = np.nan
            fp.drop(columns=["first_split_date"], inplace=True)
    except FileNotFoundError:
        pass

    # Cross-sectional z-score per signal_date (C1 fix)
    for raw_col in FACTOR_RAW_COLS:
        zscore_col = f"{raw_col}_zscore"
        if raw_col in fp.columns and zscore_col in fp.columns:
            fp[zscore_col] = fp.groupby("signal_date")[raw_col].transform(_cs_zscore)

    sig_df = pd.DataFrame({"signal_date": fp["signal_date"].unique()}).sort_values("signal_date")
    me_df = pd.DataFrame({"date": month_ends}).sort_values("date")
    snapped = pd.merge_asof(sig_df, me_df, left_on="signal_date", right_on="date",
                            tolerance=pd.Timedelta("5D"), direction="nearest")
    fp["date"] = fp["signal_date"].map(dict(zip(snapped["signal_date"], snapped["date"])))
    return fp.dropna(subset=["date"])[["ticker", "date"] + FACTOR_ZSCORE_COLS].reset_index(drop=True)


# ── V2 WRDS builders ────────────────────────────────────────────────────

def build_returns_v2() -> pd.DataFrame:
    """Forward returns from CRSP monthly ret — no yfinance dependency."""
    crsp = pd.read_parquet(CRSP_MONTHLY)[["permno", "date", "ret", "ticker"]].copy()
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp = crsp.sort_values(["permno", "date"])

    # 1-month forward return = next month's ret
    crsp["fwd_ret_1m"] = crsp.groupby("permno")["ret"].shift(-1)

    # 3-month forward return = (1+r_t+1)(1+r_t+2)(1+r_t+3) - 1
    crsp["ret1"] = crsp.groupby("permno")["ret"].shift(-1)
    crsp["ret2"] = crsp.groupby("permno")["ret"].shift(-2)
    crsp["ret3"] = crsp.groupby("permno")["ret"].shift(-3)
    crsp["fwd_ret_3m"] = (1 + crsp["ret1"]) * (1 + crsp["ret2"]) * (1 + crsp["ret3"]) - 1
    crsp.drop(columns=["ret1", "ret2", "ret3"], inplace=True)

    return crsp[["permno", "date", "ticker", "fwd_ret_1m", "fwd_ret_3m"]]


def load_factors_v2_raw() -> pd.DataFrame:
    """Load WRDS-built factors with plausibility filters. NO z-scoring here —
    z-scoring happens AFTER filtering to S&P 500 universe."""
    factors = pd.read_parquet(WRDS_FACTORS)
    factors["date"] = pd.to_datetime(factors["date"])

    # Plausibility caps (same as V1)
    if "EarningsYield" in factors.columns:
        factors = factors[factors["EarningsYield"].abs() <= 1.0]
    if "AssetGrowth" in factors.columns:
        factors = factors[factors["AssetGrowth"].abs() <= 10.0]

    return factors


def load_new_features_v2_raw() -> pd.DataFrame:
    """Load WRDS new features. NO z-scoring here —
    z-scoring happens AFTER filtering to S&P 500 universe."""
    nf = pd.read_parquet(WRDS_NEW_FEATURES)
    nf["date"] = pd.to_datetime(nf["date"])
    return nf


def compute_realized_vol_v2() -> pd.DataFrame:
    """Compute TRUE realized volatility from CRSP daily returns,
    NOT IVOL (which is CAPM-residual vol, a different concept).

    Aligns to CRSP monthly dates (last trading day of month),
    not calendar month-end — fixes join mismatch (Jan 30 vs Jan 31).
    """
    daily = pd.read_parquet(WRDS_DIR / "crsp_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.dropna(subset=["ret"]).sort_values(["permno", "date"])

    # 21-day trailing realized vol per stock
    daily["ret_float"] = daily["ret"].astype(float)
    daily["rv_21d"] = daily.groupby("permno")["ret_float"].transform(
        lambda x: x.rolling(21, min_periods=15).std()
    )

    # Get the last trading day per permno-month and its rv_21d
    daily["ym"] = daily["date"].dt.to_period("M")
    monthly_rv = (
        daily.sort_values("date")
        .groupby(["permno", "ym"])
        .agg(date=("date", "last"), realized_vol=("rv_21d", "last"))
        .reset_index()
        .drop(columns=["ym"])
    )
    return monthly_rv[["permno", "date", "realized_vol"]].dropna(subset=["realized_vol"])


# ── Shared helpers ───────────────────────────────────────────────────────

def load_macro(macro_path: Path) -> pd.DataFrame:
    macro = pd.read_parquet(macro_path)
    macro.index = pd.DatetimeIndex(macro.index)
    return macro[MACRO_COLS].resample("M").last().shift(1).reset_index()


def winsorise(df, col="fwd_ret_1m", pct=0.005):
    lo = df.groupby("date")[col].transform(lambda x: x.quantile(pct))
    hi = df.groupby("date")[col].transform(lambda x: x.quantile(1 - pct))
    df = df.copy()
    df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def safe_corr(x, y):
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))


# ── Diagnostics ──────────────────────────────────────────────────────────

def print_data_report(panel, feature_cols):
    target_cols = ["fwd_ret_1m", "fwd_ret_3m", "realized_vol"]

    print("\n" + "=" * 72)
    print("MASTER PANEL DATA REPORT")
    print("=" * 72)

    print(f"\n[1] PANEL OVERVIEW")
    print(f"Rows           : {len(panel):,}")
    print(f"Date range     : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Unique tickers : {panel['ticker'].nunique():,}")
    print(f"Unique months  : {panel['date'].nunique():,}")
    print(f"Sectors        : {panel['sector'].nunique():,}")
    print(f"Features       : {len(feature_cols)}")

    monthly_n = panel.groupby("date")["ticker"].nunique()
    print(f"\n[2] MONTHLY CROSS-SECTION")
    print(f"Mean stocks/month : {monthly_n.mean():.1f}")
    print(f"Min / Max         : {monthly_n.min():.0f} / {monthly_n.max():.0f}")

    print(f"\n[3] TARGET SUMMARY")
    print(panel[target_cols].describe().round(4).to_string())

    print(f"\n[4] FEATURE NaN RATES")
    for c in feature_cols:
        if c in panel.columns:
            pct = panel[c].isna().mean()
            if pct > 0.001:
                print(f"  {c:<35} {pct:.1%}")

    print(f"\n[5] FEATURE -> TARGET ICs")
    rows = []
    for c in feature_cols:
        if c in panel.columns:
            rows.append({
                "feature": c,
                "IC_1m": safe_corr(panel[c], panel["fwd_ret_1m"]),
                "IC_3m": safe_corr(panel[c], panel["fwd_ret_3m"]),
            })
    print(pd.DataFrame(rows).set_index("feature").round(4).to_string())
    print("\n" + "=" * 72)


# ── V1 builder (legacy) ─────────────────────────────────────────────────

def build_master_panel_v1(save=True, verbose=True):
    """Original pipeline: XBRL + yfinance. Output: master_panel.parquet"""
    if verbose:
        print("=== Building V1 panel (XBRL + yfinance) ===")

    prices = pd.read_parquet(PRICES_CACHE)
    prices.index = pd.DatetimeIndex(prices.index)
    month_ends = prices.resample("M").last().index

    fwd_1m = build_monthly_returns_v1(PRICES_CACHE)
    fwd_3m = build_fwd_ret_3m_v1(PRICES_CACHE)
    rv = build_realized_vol_v1(PRICES_CACHE, month_ends)
    factors = load_factors_v1(FACTOR_PANEL, month_ends)
    macro = load_macro(MACRO_FEATURES)
    with open(SECTOR_MAP) as f:
        sector_map = json.load(f)

    panel = (
        factors
        .pipe(lambda d: pd.merge(d, fwd_1m, on=["ticker", "date"], how="inner"))
        .pipe(lambda d: pd.merge(d, rv, on=["ticker", "date"], how="left"))
        .pipe(lambda d: pd.merge(d, fwd_3m, on=["ticker", "date"], how="left"))
    )
    panel["sector"] = panel["ticker"].map(sector_map)
    panel = panel.dropna(subset=["sector"])
    panel = pd.merge(panel, macro, on="date", how="left")

    # Impute
    for col in FACTOR_ZSCORE_COLS:
        if col in TYPE_A_SECTORS:
            type_a_mask = panel["sector"].isin(TYPE_A_SECTORS[col])
            panel.loc[~type_a_mask, col] = panel.loc[~type_a_mask, col].fillna(0.0)
        else:
            panel[col] = panel[col].fillna(0.0)

    macro_present = [c for c in MACRO_COLS if c in panel.columns]
    panel[macro_present] = panel.sort_values("date")[macro_present].ffill().fillna(0.0)

    rv_median = panel.groupby("date")["realized_vol"].transform("median")
    panel["realized_vol"] = panel["realized_vol"].fillna(rv_median)
    panel = winsorise(panel, col="fwd_ret_1m")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    if verbose:
        print_data_report(panel, ALL_FEATURE_COLS_V1)
    if save:
        panel.to_parquet(OUTPUT_V1, index=False)
        if verbose:
            print(f"Saved -> {OUTPUT_V1}")
    return panel


# ── V2 builder (WRDS) ───────────────────────────────────────────────────

def build_master_panel_v2(save=True, verbose=True):
    """
    WRDS pipeline: Compustat/CRSP + IBES + alt data.
    Output: master_panel_v2.parquet

    Key design decisions (audit-validated):
    1. Z-scoring AFTER filtering to S&P 500 (not full CRSP universe)
    2. Macro/regime NOT included as direct features (zero CS variance)
       → kept as metadata columns for regime-conditioning, not model input
    3. Realized vol computed from CRSP daily returns (not IVOL proxy)
    4. SUE NaN → missingness indicator instead of fillna(0)
    5. Accruals Type A added for Financials
    """
    if verbose:
        print("=== Building V2 panel (WRDS) — audit-corrected ===")

    # ── Load raw data (NO z-scoring yet) ──
    if verbose:
        print("Loading CRSP returns...")
    returns = build_returns_v2()

    if verbose:
        print("Loading WRDS factors (raw)...")
    factors = load_factors_v2_raw()

    if verbose:
        print("Loading new features (raw)...")
    new_feat = load_new_features_v2_raw()

    if verbose:
        print("Computing realized vol from CRSP daily...")
    rv = compute_realized_vol_v2()

    if verbose:
        print("Loading macro + sectors...")
    macro = load_macro(MACRO_FEATURES)
    with open(SECTOR_MAP) as f:
        sector_map = json.load(f)

    # ── Merge: factors + returns (on permno+date) ──
    panel = factors.merge(returns, on=["permno", "date", "ticker"], how="inner")

    # ── Merge new features (on permno+date) ──
    panel = panel.merge(new_feat, on=["permno", "date"], how="left")

    # ── Merge realized vol (on permno+date) ──
    panel = panel.merge(rv, on=["permno", "date"], how="left")

    # ══════════════════════════════════════════════════════════════
    # FILTER TO S&P 500 FIRST — before any z-scoring
    # ══════════════════════════════════════════════════════════════
    panel["sector"] = panel["ticker"].map(sector_map)
    panel = panel.dropna(subset=["sector"])
    panel = panel[(panel["date"] >= "2015-01-01") & (panel["date"] <= "2024-12-31")]
    panel = panel.dropna(subset=["fwd_ret_1m"])

    if verbose:
        print(f"After S&P 500 filter: {len(panel):,} rows, "
              f"{panel['ticker'].nunique()} tickers, "
              f"{panel['date'].nunique()} months")

    # ══════════════════════════════════════════════════════════════
    # Z-SCORE WITHIN S&P 500 UNIVERSE (not full CRSP)
    # ══════════════════════════════════════════════════════════════
    if verbose:
        print("Cross-sectional z-scoring within S&P 500...")

    # Original 6 factors
    for raw_col in FACTOR_RAW_COLS:
        zscore_col = f"{raw_col}_zscore"
        if raw_col in panel.columns:
            panel[zscore_col] = panel.groupby("date")[raw_col].transform(_cs_zscore)

    # New 8 features
    for raw_col, zscore_col in NEW_ZSCORE_MAP.items():
        if raw_col in panel.columns:
            panel[zscore_col] = panel.groupby("date")[raw_col].transform(_cs_zscore)

    # ══════════════════════════════════════════════════════════════
    # IMPUTATION — audit-corrected
    # ══════════════════════════════════════════════════════════════

    # Extended Type A sectors (audit finding: add Accruals for Financials)
    TYPE_A_EXTENDED = {
        "GrossProfitability_zscore": ["Financials", "Utilities", "Real Estate"],
        "NetDebtEBITDA_zscore": ["Financials", "Real Estate"],
        "Accruals_zscore": ["Financials"],  # audit: different accrual rules
    }

    all_zscore_cols = FACTOR_ZSCORE_COLS + list(NEW_ZSCORE_MAP.values())
    for col in all_zscore_cols:
        if col not in panel.columns:
            continue
        if col in TYPE_A_EXTENDED:
            type_a_mask = panel["sector"].isin(TYPE_A_EXTENDED[col])
            panel.loc[~type_a_mask, col] = panel.loc[~type_a_mask, col].fillna(0.0)
        else:
            panel[col] = panel[col].fillna(0.0)

    # SUE missingness indicator (audit: 72.9% NaN → 0 is misleading)
    # "no analyst coverage" ≠ "zero surprise"
    panel["has_sue"] = (~panel["SUE"].isna()).astype(float)

    # ── Macro: keep as metadata, NOT in feature set ──
    # Audit finding: zero cross-sectional variance → useless in CS regression
    # Kept for regime-conditioning and diagnostic purposes
    macro_present = [c for c in MACRO_COLS if c in panel.columns]
    if not macro_present:
        panel = pd.merge(panel, macro, on="date", how="left")
        macro_present = [c for c in MACRO_COLS if c in panel.columns]
    panel[macro_present] = panel.sort_values("date")[macro_present].ffill().fillna(0.0)

    # ── Regime: keep as metadata ──
    for c in REGIME_COLS:
        if c in panel.columns:
            panel[c] = panel[c].ffill().fillna(0)

    # ── Realized vol imputation ──
    rv_median = panel.groupby("date")["realized_vol"].transform("median")
    panel["realized_vol"] = panel["realized_vol"].fillna(rv_median)

    # ── Winsorise primary target ──
    panel = winsorise(panel, col="fwd_ret_1m")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── Convert nullable dtypes to standard numpy (sklearn/torch compatibility) ──
    for col in panel.columns:
        if str(panel[col].dtype) in ("Float64", "Float32"):
            panel[col] = panel[col].astype("float64")
        elif str(panel[col].dtype) in ("Int64", "Int32"):
            panel[col] = panel[col].astype("float64")

    # ── Report ──
    # Model features = stock-level z-scores only (no macro/regime)
    stock_features = FACTOR_ZSCORE_COLS + list(NEW_ZSCORE_MAP.values())
    if verbose:
        print_data_report(panel, stock_features)

        # Verify z-score correctness
        sample_month = sorted(panel["date"].unique())[60]
        sample = panel[panel["date"] == sample_month]
        print(f"\nZ-score verification (month={str(sample_month)[:7]}):")
        for c in stock_features[:5]:
            if c in sample.columns:
                print(f"  {c:<35} mean={sample[c].mean():+.4f} std={sample[c].std():.4f}")

    if save:
        panel.to_parquet(OUTPUT_V2, index=False)
        if verbose:
            print(f"\nSaved -> {OUTPUT_V2}")
    return panel


# ── Entry point ──────────────────────────────────────────────────────────

# Backward compatibility alias
build_master_panel = build_master_panel_v2

if __name__ == "__main__":
    if "--v1" in sys.argv:
        build_master_panel_v1(save=True, verbose=True)
    else:
        build_master_panel_v2(save=True, verbose=True)
