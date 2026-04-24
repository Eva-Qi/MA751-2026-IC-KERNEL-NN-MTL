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
    LIQUIDITY_COLS, PRICE_PATTERN_COLS, GROWTH_COLS, COVERAGE_COLS,
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

# Phase 2 new feature paths
NEW_FEATURES_COMPUTED   = WRDS_DIR / "new_features_computed.parquet"
NEW_FEATURES_IBES       = WRDS_DIR / "new_features_ibes.parquet"
NEW_FEATURES_QUARTERLY  = WRDS_DIR / "new_features_quarterly.parquet"

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

# Phase 2 raw → zscore mapping (z-scoring happens inside build_master_panel_v2)
PHASE2_ZSCORE_MAP = {
    "Turnover":           "Turnover_zscore",
    "AmihudIlliquidity":  "AmihudIlliquidity_zscore",
    "High52W_Proximity":  "High52W_Proximity_zscore",
    "MaxDailyReturn":     "MaxDailyReturn_zscore",
    "ReturnSkewness":     "ReturnSkewness_zscore",
    "ImpliedEPSGrowth":   "ImpliedEPSGrowth_zscore",
    "QRevGrowthYoY":      "QRevGrowthYoY_zscore",
    "AnalystCoverageChg": "AnalystCoverageChg_zscore",
}


# ── Cross-sectional z-score ─────────────────────────────────────────────

def _cs_zscore(g):
    mu = g.mean()
    sigma = g.std()
    return (g - mu) / sigma if sigma > 1e-8 else g * 0.0


def _cs_zscore_winsorized(g):
    """Winsorize at 1st/99th percentile before z-scoring.

    Prevents extreme outliers (e.g. EarningsYield for micro-caps) from
    collapsing the z-score distribution of the rest of the cross-section.
    """
    lower = g.quantile(0.01)
    upper = g.quantile(0.99)
    g_clipped = g.clip(lower, upper)
    mu = g_clipped.mean()
    sigma = g_clipped.std()
    return (g_clipped - mu) / (sigma + 1e-8) if sigma > 1e-8 else g_clipped * 0.0


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


def load_new_computed_features(panel_dates: pd.Series) -> pd.DataFrame:
    """Load Phase 2 features computed from existing WRDS data.

    Merges all three new feature files onto the set of (permno, date) pairs
    that appear in the panel.  Uses merge_asof so that small calendar offsets
    (last-trading-day vs calendar-month-end) don't cause false NaNs.

    Parameters
    ----------
    panel_dates : pd.Series
        Sorted unique panel dates — used as the right-side key so that
        new_features_computed (1M+ rows) is never expanded to panel size.

    Returns
    -------
    pd.DataFrame with columns: permno, date, <8 raw feature columns>
    """
    tolerance = pd.Timedelta("3D")

    # ── 1. new_features_computed (daily-granularity → 1M+ rows) ──────────
    # Strategy: snap each raw row to the nearest panel date first (many-to-one),
    # then group-aggregate so the merge target has at most one row per
    # (permno, snapped_date).  This prevents panel explosion.
    computed = pd.read_parquet(NEW_FEATURES_COMPUTED)
    computed["date"] = pd.to_datetime(computed["date"])
    computed["permno"] = computed["permno"].astype("int64")
    computed = computed.sort_values(["permno", "date"])

    # Snap raw dates → nearest panel date within tolerance
    panel_dates_df = pd.DataFrame({"panel_date": panel_dates}).sort_values("panel_date")
    computed_dates = pd.DataFrame({"date": computed["date"].unique()}).sort_values("date")
    snapped = pd.merge_asof(
        computed_dates, panel_dates_df,
        left_on="date", right_on="panel_date",
        tolerance=tolerance, direction="nearest",
    )
    date_to_panel = dict(zip(snapped["date"], snapped["panel_date"]))
    computed["panel_date"] = computed["date"].map(date_to_panel)
    computed = computed.dropna(subset=["panel_date"])

    # Aggregate: take last value per (permno, panel_date) — most recent obs wins
    computed_feat_cols = ["Turnover", "High52W_Proximity", "MaxDailyReturn",
                          "AmihudIlliquidity", "ReturnSkewness"]
    computed_agg = (
        computed.sort_values("date")
        .groupby(["permno", "panel_date"])[computed_feat_cols]
        .last()
        .reset_index()
        .rename(columns={"panel_date": "date"})
    )

    # ── 2. new_features_ibes (monthly-ish, same approach) ────────────────
    ibes = pd.read_parquet(NEW_FEATURES_IBES)
    ibes["date"] = pd.to_datetime(ibes["date"])
    ibes["permno"] = ibes["permno"].astype("int64")
    ibes = ibes.sort_values(["permno", "date"])

    # IBES dates are mid-month (e.g. Jan 17) while panel is month-end (Jan 30).
    # Use backward snap: assign each IBES row to the NEXT panel date (within 31 days).
    ibes_dates = pd.DataFrame({"date": ibes["date"].unique()}).sort_values("date")
    ibes_snapped = pd.merge_asof(
        ibes_dates, panel_dates_df,
        left_on="date", right_on="panel_date",
        tolerance=pd.Timedelta("31D"), direction="forward",
    )
    ibes_date_map = dict(zip(ibes_snapped["date"], ibes_snapped["panel_date"]))
    ibes["panel_date"] = ibes["date"].map(ibes_date_map)
    ibes = ibes.dropna(subset=["panel_date"])

    ibes_feat_cols = ["ImpliedEPSGrowth", "AnalystCoverageChg"]
    ibes_agg = (
        ibes.sort_values("date")
        .groupby(["permno", "panel_date"])[ibes_feat_cols]
        .last()
        .reset_index()
        .rename(columns={"panel_date": "date"})
    )

    # ── 3. new_features_quarterly (avail_date = datadate + 90 days) ───────
    # 'date' is already the point-in-time availability date.
    # Use merge_asof backward: for each panel date, use the most recent
    # available quarterly observation (no tolerance cap needed here).
    quarterly = pd.read_parquet(NEW_FEATURES_QUARTERLY)
    quarterly["date"] = pd.to_datetime(quarterly["date"])
    quarterly["permno"] = quarterly["permno"].astype("int64")
    quarterly = quarterly.sort_values(["permno", "date"])

    # merge_asof requires the panel as left side and features as right side,
    # both sorted by date, grouped by permno.
    # Build a skeleton of unique (permno, panel_date) pairs to merge against.
    permnos_in_quarterly = quarterly["permno"].unique()
    skeleton = pd.DataFrame(
        [(p, d) for p in permnos_in_quarterly for d in panel_dates.values],
        columns=["permno", "date"],
    ).sort_values("date")  # merge_asof requires left sorted by 'on' key

    quarterly = quarterly.sort_values("date")  # right also must be sorted

    quarterly_agg = pd.merge_asof(
        skeleton, quarterly.drop_duplicates(subset=["permno", "date"], keep="last"),
        on="date", by="permno",
        direction="backward",  # most recent available (point-in-time safe)
    )

    # ── 4. Combine all three onto a single (permno, date) frame ──────────
    result = computed_agg.merge(ibes_agg, on=["permno", "date"], how="outer")
    result = result.merge(quarterly_agg, on=["permno", "date"], how="outer")
    result = result.sort_values(["permno", "date"]).reset_index(drop=True)

    return result


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

    # ── Merge new features (fuzzy date match — NF uses calendar month-end,
    #    panel uses last trading day; differ by 1-2 days in ~36/119 months) ──
    panel["permno"] = panel["permno"].astype("int64")
    new_feat["permno"] = new_feat["permno"].astype("int64")

    # merge_asof requires sorting by the 'on' key (date)
    panel = panel.sort_values("date")
    new_feat = new_feat.sort_values("date")
    panel = pd.merge_asof(
        panel, new_feat,
        on="date", by="permno",
        tolerance=pd.Timedelta("3D"),
        direction="nearest",
    )

    # ── Merge realized vol (fuzzy date match — same issue) ──
    rv["permno"] = rv["permno"].astype("int64")
    rv = rv.sort_values("date")
    panel = panel.sort_values("date")
    panel = pd.merge_asof(
        panel, rv,
        on="date", by="permno",
        tolerance=pd.Timedelta("3D"),
        direction="nearest",
    )

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
    # MERGE PHASE 2 FEATURES (after S&P 500 filter, before z-scoring)
    # panel is ~47K rows here; load_new_computed_features() snaps the
    # 1M-row computed file to panel dates BEFORE merging, so no explosion.
    # ══════════════════════════════════════════════════════════════
    if verbose:
        print("Loading Phase 2 computed features (liquidity, price patterns, growth)...")
    panel_unique_dates = panel["date"].drop_duplicates().sort_values()
    phase2_feat = load_new_computed_features(panel_unique_dates)
    phase2_feat["permno"] = phase2_feat["permno"].astype("int64")

    pre_merge_rows = len(panel)
    panel = panel.merge(phase2_feat, on=["permno", "date"], how="left")
    assert len(panel) == pre_merge_rows, (
        f"Phase 2 merge exploded panel: {pre_merge_rows} -> {len(panel)}"
    )
    if verbose:
        print(f"Phase 2 merge OK — panel still {len(panel):,} rows")

    # ══════════════════════════════════════════════════════════════
    # Z-SCORE WITHIN S&P 500 UNIVERSE (not full CRSP)
    # ══════════════════════════════════════════════════════════════
    if verbose:
        print("Cross-sectional z-scoring within S&P 500...")

    # Original 6 factors — winsorized z-score (prevents outlier collapse)
    for raw_col in FACTOR_RAW_COLS:
        zscore_col = f"{raw_col}_zscore"
        if raw_col in panel.columns:
            panel[zscore_col] = panel.groupby("date")[raw_col].transform(_cs_zscore_winsorized)

    # New 8 features — winsorized z-score
    for raw_col, zscore_col in NEW_ZSCORE_MAP.items():
        if raw_col in panel.columns:
            panel[zscore_col] = panel.groupby("date")[raw_col].transform(_cs_zscore_winsorized)

    # Phase 2 features — winsorized z-score
    for raw_col, zscore_col in PHASE2_ZSCORE_MAP.items():
        if raw_col in panel.columns:
            panel[zscore_col] = panel.groupby("date")[raw_col].transform(_cs_zscore_winsorized)

    # ══════════════════════════════════════════════════════════════
    # IMPUTATION — audit-corrected
    # ══════════════════════════════════════════════════════════════

    # Extended Type A sectors (audit finding: add Accruals for Financials)
    TYPE_A_EXTENDED = {
        "GrossProfitability_zscore": ["Financials", "Utilities", "Real Estate"],
        "NetDebtEBITDA_zscore": ["Financials", "Real Estate"],
        "Accruals_zscore": ["Financials"],  # audit: different accrual rules
    }

    # Columns that MUST NOT be zero-filled — NaN carries economic meaning:
    #   SUE_zscore            : NaN = "IBES doesn't cover this stock" (Type D)
    #   ShortInterestRatio_zscore : NaN = "no short interest data in Compustat" (Type D)
    #   InstOwnershipChg_zscore   : NaN = "no new 13F filing this month" (expected quarterly)
    # Models handle NaN at tensor-creation time via np.nan_to_num(nan=0.0),
    # so keeping NaN here does NOT break training — it just preserves signal integrity.
    KEEP_NAN_COLS = {
        "SUE_zscore",
        "ShortInterestRatio_zscore",
        "InstOwnershipChg_zscore",
    }
    # NOTE: has_analyst_consensus and has_positive_earnings are flag columns
    # (not feature z-scores) and are constructed later in this function.
    # They do NOT belong in KEEP_NAN_COLS — they have no NaN by construction
    # (notna() and > 0 both produce boolean → 0.0/1.0 with no NaN).

    # Beta/IVOL: sector-cross-sectional median (not zero — new IPOs are typically high-beta)
    # Small volume: 0.45% / 0.28% of panel; empirically 39 / 38 tickers, ~5-6 months each (post-IPO window)
    for risk_col in ["Beta_zscore", "IVOL_zscore"]:
        if risk_col in panel.columns:
            panel[risk_col] = panel.groupby(["date", "sector"])[risk_col].transform(
                lambda x: x.fillna(x.median())
            )
            # Fallback: if sector median is also NaN (all-NaN sector-month), fall back to cross-sectional median
            panel[risk_col] = panel.groupby("date")[risk_col].transform(
                lambda x: x.fillna(x.median())
            )

    # Columns handled by sector-median above — exclude from generic zero-fill
    SECTOR_MEDIAN_FILLED_COLS = {"Beta_zscore", "IVOL_zscore"}

    all_zscore_cols = FACTOR_ZSCORE_COLS + list(NEW_ZSCORE_MAP.values())
    for col in all_zscore_cols:
        if col not in panel.columns:
            continue
        if col in KEEP_NAN_COLS:
            # Do NOT zero-fill — NaN = missing coverage, not zero signal
            continue
        if col in SECTOR_MEDIAN_FILLED_COLS:
            # Already filled by sector-median block above — skip zero-fill
            continue
        if col in TYPE_A_EXTENDED:
            type_a_mask = panel["sector"].isin(TYPE_A_EXTENDED[col])
            panel.loc[~type_a_mask, col] = panel.loc[~type_a_mask, col].fillna(0.0)
        else:
            panel[col] = panel[col].fillna(0.0)

    # Phase 2 z-score imputation — all are economic signals, zero-fill NaN
    # (no Type A exceptions needed: these are price/liquidity/growth signals,
    #  not accounting ratios with sector-specific definitional issues)
    phase2_zscore_cols = list(PHASE2_ZSCORE_MAP.values())
    for col in phase2_zscore_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)

    # ── Phase 2 coverage stats ──────────────────────────────────────────
    if verbose:
        print("\n[Phase 2 Feature Coverage]")
        for raw_col, zscore_col in PHASE2_ZSCORE_MAP.items():
            if raw_col in panel.columns:
                n_nonnull = panel[raw_col].notna().sum()
                pct = n_nonnull / len(panel)
                print(f"  {raw_col:<25} {n_nonnull:>7,} / {len(panel):,}  ({pct:.1%})")

    # ── Missingness indicators ──────────────────────────────────────────
    # SUE: "no analyst coverage" ≠ "zero surprise"
    panel["has_sue"] = (~panel["SUE"].isna()).astype(float)

    # ShortInterestRatio: "no Compustat short interest record" ≠ "zero short interest"
    panel["has_short_interest"] = (~panel["ShortInterestRatio"].isna()).astype(float)

    # InstOwnershipChg: only populated in 13F filing months (quarterly)
    # NaN in non-filing months is structurally expected, not a pipeline error
    panel["has_inst_ownership"] = (~panel["InstOwnershipChg"].isna()).astype(float)

    # AnalystRevision/Dispersion/Breadth: unified IBES coverage indicator
    # NaN in any of the three = missing IBES consensus row this month
    # Empirically <1.1% NaN in S&P 500 panel but inconsistency with SUE flagged by audit
    panel["has_analyst_consensus"] = (
        panel["AnalystRevision"].notna()
        & panel["AnalystDispersion"].notna()
        & panel["RevisionBreadth"].notna()
    ).astype(float)

    # Compustat net income sign: firms in profit vs loss regime
    # NOTE: raw `ib` column is not carried into the panel (wrds_factor_builder.py
    # uses it to compute EarningsYield then drops it in output_cols).
    # EarningsYield = ib / mktcap; mktcap is always positive, so
    # sign(EarningsYield) == sign(ib) — use it as an exact proxy.
    # EY NaN → profit regime unknown → flag resolves to 0.0 (conservative)
    panel["has_positive_earnings"] = (panel["EarningsYield"] > 0).astype(float)

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

    # ── Forward realized vol (MTL auxiliary target: NEXT month's vol) ──
    # shift(-1) within each ticker so fwd_realized_vol[t] = realized_vol[t+1]
    # Last month per ticker will be NaN — handled by NaN masking in the loss
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["fwd_realized_vol"] = panel.groupby("ticker")["realized_vol"].shift(-1)

    # ── Winsorise primary target ──
    panel = winsorise(panel, col="fwd_ret_1m")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── Sector-adjusted return: removes sector-level effects ──
    sector_mean = panel.groupby(["date", "sector"])["fwd_ret_1m"].transform("mean")
    panel["fwd_ret_1m_secadj"] = panel["fwd_ret_1m"] - sector_mean

    # ── Convert nullable dtypes to standard numpy (sklearn/torch compatibility) ──
    for col in panel.columns:
        if str(panel[col].dtype) in ("Float64", "Float32"):
            panel[col] = panel[col].astype("float64")
        elif str(panel[col].dtype) in ("Int64", "Int32"):
            panel[col] = panel[col].astype("float64")

    # ── Report ──
    # Model features = stock-level z-scores only (no macro/regime)
    stock_features = (
        FACTOR_ZSCORE_COLS
        + list(NEW_ZSCORE_MAP.values())
        + list(PHASE2_ZSCORE_MAP.values())
    )
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
