import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("data/raw/")
OUTPUT_DIR = Path("data/")

FACTOR_PANEL   = DATA_DIR / "factor_panel.parquet"
SPLIT_HISTORY  = DATA_DIR / "split_history.parquet"
PRICES_CACHE   = DATA_DIR / "prices_cache.parquet"
MACRO_FEATURES = DATA_DIR / "macro_features.parquet"
SECTOR_MAP     = DATA_DIR / "sector_map.json"
OUTPUT_FILE    = OUTPUT_DIR / "master_panel.parquet"

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

FACTOR_RAW_COLS = [
    "EarningsYield",
    "GrossProfitability",
    "AssetGrowth",
    "Accruals",
    "Momentum12_1",
    "NetDebtEBITDA",
]


# Price-derived targets

def build_monthly_returns(prices_path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)
    monthly = prices.resample("M").last()
    fwd = monthly.pct_change(1).shift(-1)
    return (
        fwd.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="fwd_ret_1m")
        .dropna(subset=["fwd_ret_1m"])
    )


def build_fwd_ret_3m(prices_path: Path) -> pd.DataFrame:
    """
    3-month forward return, computed as price(t+3) / price(t) - 1.

    Rows in the final 3 months of the price history will be NaN — this is
    intentional and signals to MTLLoss that the 3m target is not yet
    observable for those rows (they should be masked in the loss).
    """
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)
    monthly = prices.resample("M").last()
    fwd3 = monthly.pct_change(3).shift(-3)   # NaN for last 3 months by design
    return (
        fwd3.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="fwd_ret_3m")
        # Do NOT dropna here
    )


def build_realized_vol(
    prices_path: Path,
    month_ends: pd.DatetimeIndex,
    window: int = 21,
) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)
    rv = prices.pct_change().rolling(window).std()
    rv_monthly = rv.reindex(month_ends, method="ffill")
    return (
        rv_monthly.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="realized_vol")
        .dropna(subset=["realized_vol"])
    )

# Factor and macro loaders

def load_factors(factor_path: Path, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    fp = pd.read_parquet(factor_path)
    fp["signal_date"] = pd.DatetimeIndex(fp["signal_date"])

    if "EarningsYield" in fp.columns:
        fp = fp[fp["EarningsYield"].abs() <= 1.0]
    if "AssetGrowth" in fp.columns:
        fp = fp[fp["AssetGrowth"].abs() <= 10.0]

    # Flag split-contaminated EY: only NaN rows BEFORE the split date
    # (yfinance retroactively adjusts prices pre-split, so pre-split
    #  market cap = adjusted_price × raw_shares is wrong)
    try:
        split_hist = pd.read_parquet(SPLIT_HISTORY)
        split_hist["date"] = pd.to_datetime(split_hist["date"], utc=True).dt.tz_localize(None)
        major = split_hist[split_hist["ratio"] >= 2.0][["ticker", "date"]].copy()
        # For each ticker, find the EARLIEST major split in sample period
        earliest_split = major.groupby("ticker")["date"].min().reset_index()
        earliest_split.columns = ["ticker", "first_split_date"]
        if "EarningsYield" in fp.columns:
            fp = fp.merge(earliest_split, on="ticker", how="left")
            # Only NaN rows where signal_date is BEFORE the split
            contaminated = (
                fp["first_split_date"].notna()
                & (fp["signal_date"] < fp["first_split_date"])
            )
            fp.loc[contaminated, "EarningsYield"] = np.nan
            fp.drop(columns=["first_split_date"], inplace=True)
    except FileNotFoundError:
        pass  # No split history available

    # Cross-sectional z-score: compute per signal_date, NOT pooled
    def _cs_zscore(g):
        mu = g.mean()
        sigma = g.std()
        return (g - mu) / sigma if sigma > 1e-8 else g * 0.0

    for raw_col in FACTOR_RAW_COLS:
        zscore_col = f"{raw_col}_zscore"
        if raw_col in fp.columns and zscore_col in fp.columns:
            fp[zscore_col] = fp.groupby("signal_date")[raw_col].transform(_cs_zscore)

    sig_df = (
        pd.DataFrame({"signal_date": fp["signal_date"].unique()})
        .sort_values("signal_date")
    )
    me_df = pd.DataFrame({"date": month_ends}).sort_values("date")

    snapped = pd.merge_asof(
        sig_df,
        me_df,
        left_on="signal_date",
        right_on="date",
        tolerance=pd.Timedelta("5D"),
        direction="nearest",
    )

    fp["date"] = fp["signal_date"].map(
        dict(zip(snapped["signal_date"], snapped["date"]))
    )
    return fp.dropna(subset=["date"])[
        ["ticker", "date"] + FACTOR_ZSCORE_COLS
    ].reset_index(drop=True)


def load_macro(macro_path: Path) -> pd.DataFrame:
    macro = pd.read_parquet(macro_path)
    macro.index = pd.DatetimeIndex(macro.index)
    return macro[MACRO_COLS].resample("M").last().shift(1).reset_index()


# Panel construction helpers

def winsorise(df: pd.DataFrame, col: str = "fwd_ret_1m", pct: float = 0.005) -> pd.DataFrame:
    lo = df.groupby("date")[col].transform(lambda x: x.quantile(pct))
    hi = df.groupby("date")[col].transform(lambda x: x.quantile(1 - pct))
    df = df.copy()
    df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1]))


# Diagnostics report

def print_data_report(panel: pd.DataFrame) -> None:
    target_cols = ["fwd_ret_1m", "fwd_ret_3m", "realized_vol"]

    print("\n" + "=" * 72)
    print("MASTER PANEL DATA REPORT  (v4)")
    print("=" * 72)

    n_rows    = len(panel)
    n_tickers = panel["ticker"].nunique()
    n_months  = panel["date"].nunique()
    n_sectors = panel["sector"].nunique()

    print("\n[1] PANEL OVERVIEW")
    print(f"Rows                  : {n_rows:,}")
    print(f"Date range            : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Unique tickers        : {n_tickers:,}")
    print(f"Unique months         : {n_months:,}")
    print(f"Unique sectors        : {n_sectors:,}")

    monthly_n = panel.groupby("date")["ticker"].nunique()
    print("\n[2] MONTHLY CROSS-SECTION SIZE")
    print(f"Mean stocks/month     : {monthly_n.mean():.1f}")
    print(f"Min / Max             : {monthly_n.min():.0f} / {monthly_n.max():.0f}")

    print("\n[3] TARGET SUMMARY")
    print(panel[target_cols].describe().round(4).to_string())

    fwd3m_null_frac = panel["fwd_ret_3m"].isna().mean()
    print(f"\nfwd_ret_3m NaN fraction : {fwd3m_null_frac:.3f}")
    print("  (Expected ~3/n_months for tail months where t+3 not yet observed)")

    print("\n[4] TARGET CORRELATIONS")
    print(f"corr(fwd_ret_1m, fwd_ret_3m)    = {safe_corr(panel['fwd_ret_1m'], panel['fwd_ret_3m']):.4f}")
    print(f"corr(fwd_ret_1m, realized_vol)  = {safe_corr(panel['fwd_ret_1m'], panel['realized_vol']):.4f}")
    print(f"corr(fwd_ret_3m, realized_vol)  = {safe_corr(panel['fwd_ret_3m'], panel['realized_vol']):.4f}")

    monthly_ic_3m = panel.groupby("date").apply(
        lambda g: safe_corr(g["fwd_ret_1m"], g["fwd_ret_3m"])
    )
    print(f"\nMonthly cross-sectional corr(fwd_ret_1m, fwd_ret_3m):")
    print(monthly_ic_3m.describe().round(4).to_string())

    print("\n[5] FEATURE -> TARGET ICs")
    corr_rows = []
    for col in ALL_FEATURE_COLS:
        corr_rows.append({
            "feature": col,
            "IC_fwd_ret_1m":  safe_corr(panel[col], panel["fwd_ret_1m"]),
            "IC_fwd_ret_3m":  safe_corr(panel[col], panel["fwd_ret_3m"]),
            "IC_realized_vol": safe_corr(panel[col], panel["realized_vol"]),
        })
    corr_df = pd.DataFrame(corr_rows).set_index("feature")
    print(corr_df.round(4).to_string())

    print("\n" + "=" * 72)
    print("END OF DATA REPORT")
    print("=" * 72 + "\n")


# Main builder

def build_master_panel(save: bool = True, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("Loading prices...")
    prices = pd.read_parquet(PRICES_CACHE)
    prices.index = pd.DatetimeIndex(prices.index)
    month_ends = prices.resample("M").last().index

    if verbose:
        print("Computing forward returns (1m)...")
    fwd_ret_1m = build_monthly_returns(PRICES_CACHE)

    if verbose:
        print("Computing forward returns (3m)...")
    fwd_ret_3m = build_fwd_ret_3m(PRICES_CACHE)

    if verbose:
        print("Computing realized volatility...")
    rv = build_realized_vol(PRICES_CACHE, month_ends)

    if verbose:
        print("Loading factors, macro, sectors...")
    factors = load_factors(FACTOR_PANEL, month_ends)
    macro   = load_macro(MACRO_FEATURES)
    with open(SECTOR_MAP) as f:
        sector_map = json.load(f)

    panel = (
        factors
        .pipe(lambda d: pd.merge(d, fwd_ret_1m, on=["ticker", "date"], how="inner"))
        .pipe(lambda d: pd.merge(d, rv, on=["ticker", "date"], how="left"))
        .pipe(lambda d: pd.merge(d, fwd_ret_3m, on=["ticker", "date"], how="left"))
    )

    panel["sector"] = panel["ticker"].map(sector_map)
    panel = panel.dropna(subset=["sector"])
    panel = pd.merge(panel, macro, on="date", how="left")

    # Impute features — respect Type A missing (economically undefined)
    # GP/A is undefined for Financials, Utilities, Real Estate
    # NetDebt/EBITDA is undefined for Financials, Real Estate
    TYPE_A_SECTORS = {
        "GrossProfitability_zscore": ["Financials", "Utilities", "Real Estate"],
        "NetDebtEBITDA_zscore": ["Financials", "Real Estate"],
    }
    for col in FACTOR_ZSCORE_COLS:
        if col in TYPE_A_SECTORS:
            # Only fill non-Type-A rows; Type A stays NaN
            type_a_mask = panel["sector"].isin(TYPE_A_SECTORS[col])
            panel.loc[~type_a_mask, col] = panel.loc[~type_a_mask, col].fillna(0.0)
            # Type A rows stay NaN — the model must handle them
        else:
            panel[col] = panel[col].fillna(0.0)

    macro_present = [c for c in MACRO_COLS if c in panel.columns]
    panel[macro_present] = panel.sort_values("date")[macro_present].ffill().fillna(0.0)

    rv_median = panel.groupby("date")["realized_vol"].transform("median")
    panel["realized_vol"] = panel["realized_vol"].fillna(rv_median)

    # Winsorise primary target only
    panel = winsorise(panel, col="fwd_ret_1m")

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    if verbose:
        print_data_report(panel)

    if save:
        panel.to_parquet(OUTPUT_FILE, index=False)
        if verbose:
            print(f"Saved -> {OUTPUT_FILE}")

    return panel


if __name__ == "__main__":
    build_master_panel(save=True, verbose=True)