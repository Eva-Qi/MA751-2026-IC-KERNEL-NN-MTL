"""
Compute derived features from CRSP daily returns:
1. Monthly Beta (CAPM, trailing 252 trading days)
2. Idiosyncratic Volatility (IVOL, trailing 60 trading days)
3. Market Regime Variables (vol_regime, trend_regime, mkt_ret_6m)

Input:  data/wrds/crsp_daily.parquet (11.5M rows)
Output: data/wrds/stock_beta_monthly.parquet
        data/wrds/stock_ivol_monthly.parquet
        data/wrds/market_regime_monthly.parquet
"""

import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = "data/wrds"
MIN_OBS_BETA = 120   # minimum trading days for beta (out of 252 window)
MIN_OBS_IVOL = 40    # minimum trading days for IVOL (out of 60 window)
BETA_WINDOW = 252
IVOL_WINDOW = 60

def load_data():
    """Load CRSP daily and parse dates."""
    print("Loading CRSP daily data...")
    t0 = time.time()
    df = pd.read_parquet(f"{DATA_DIR}/crsp_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['permno'].nunique():,} stocks in {time.time()-t0:.1f}s")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  NaN ret: {df['ret'].isna().sum():,}")
    return df


def compute_daily_market_return(df):
    """Compute equal-weighted cross-sectional mean daily return as market proxy."""
    print("\nComputing daily market return (EW cross-sectional mean)...")
    mkt = df.groupby("date")["ret"].mean().rename("mkt_ret")
    print(f"  {len(mkt):,} trading days")
    print(f"  Mean daily mkt ret: {mkt.mean()*100:.4f}%")
    print(f"  Std daily mkt ret:  {mkt.std()*100:.4f}%")
    return mkt


def compute_beta_and_ivol(df, mkt):
    """
    Compute rolling beta (252d) and IVOL (60d) per stock.
    Uses vectorized rolling covariance for beta.
    """
    print("\nComputing beta and IVOL...")
    t0 = time.time()

    # Merge market return
    df = df.merge(mkt, left_on="date", right_index=True, how="left")

    # Drop rows where ret or mkt_ret is NaN — can't compute anything
    mask = df["ret"].notna() & df["mkt_ret"].notna()
    df = df[mask].copy()

    # Add year-month for aggregation
    df["ym"] = df["date"].dt.to_period("M")

    # Sort within each permno
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    # --- Beta (252-day rolling) ---
    print("  Computing 252-day rolling beta via groupby rolling cov/var...")
    # For each permno, rolling cov(ret, mkt_ret) / var(mkt_ret)
    # pandas groupby().rolling() approach

    # Set permno+date as index for groupby rolling
    df = df.set_index("date")

    # Rolling covariance and market variance per stock
    # We'll do this per-group to avoid cross-group contamination
    betas = []
    ivols = []
    permnos = df["permno"].unique()
    n_stocks = len(permnos)
    print(f"  Processing {n_stocks:,} stocks...")

    # Process in chunks for memory efficiency and progress reporting
    chunk_size = 500
    for i in range(0, n_stocks, chunk_size):
        chunk_permnos = permnos[i:i+chunk_size]
        chunk = df[df["permno"].isin(chunk_permnos)].copy()

        for pid in chunk_permnos:
            stk = chunk[chunk["permno"] == pid].copy()
            if len(stk) < MIN_OBS_IVOL:
                continue

            ret = stk["ret"]
            mkt = stk["mkt_ret"]

            # --- Beta: 252-day rolling ---
            if len(stk) >= MIN_OBS_BETA:
                roll_cov = ret.rolling(BETA_WINDOW, min_periods=MIN_OBS_BETA).cov(mkt)
                roll_var = mkt.rolling(BETA_WINDOW, min_periods=MIN_OBS_BETA).var()
                beta = roll_cov / roll_var

                # Take last observation per month
                stk_beta = beta.copy()
                stk_beta.name = "beta_252d"
                stk_ym = stk["ym"]
                beta_df = pd.DataFrame({"beta_252d": stk_beta, "ym": stk_ym, "permno": pid})
                beta_df = beta_df.dropna(subset=["beta_252d"])
                if len(beta_df) > 0:
                    # Last obs per month
                    beta_monthly = beta_df.groupby("ym").last().reset_index()
                    betas.append(beta_monthly)

            # --- IVOL: 60-day rolling ---
            # IVOL = std of residuals from CAPM
            # residual = ret - beta * mkt_ret
            # Use 60-day rolling beta for IVOL computation
            roll_cov_60 = ret.rolling(IVOL_WINDOW, min_periods=MIN_OBS_IVOL).cov(mkt)
            roll_var_60 = mkt.rolling(IVOL_WINDOW, min_periods=MIN_OBS_IVOL).var()
            beta_60 = roll_cov_60 / roll_var_60

            resid = ret - beta_60 * mkt
            ivol = resid.rolling(IVOL_WINDOW, min_periods=MIN_OBS_IVOL).std()

            stk_ivol = pd.DataFrame({
                "ivol_60d": ivol,
                "ym": stk["ym"],
                "permno": pid
            })
            stk_ivol = stk_ivol.dropna(subset=["ivol_60d"])
            if len(stk_ivol) > 0:
                ivol_monthly = stk_ivol.groupby("ym").last().reset_index()
                ivols.append(ivol_monthly)

        elapsed = time.time() - t0
        done = min(i + chunk_size, n_stocks)
        print(f"  Processed {done:,}/{n_stocks:,} stocks ({done/n_stocks*100:.1f}%) in {elapsed:.0f}s")

    # Combine results
    print("  Combining results...")
    beta_all = pd.concat(betas, ignore_index=True) if betas else pd.DataFrame()
    ivol_all = pd.concat(ivols, ignore_index=True) if ivols else pd.DataFrame()

    # Convert ym period to month-end date
    if len(beta_all) > 0:
        beta_all["date"] = beta_all["ym"].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)
        beta_all = beta_all[["permno", "date", "beta_252d"]]
        print(f"  Beta: {len(beta_all):,} rows, {beta_all['permno'].nunique():,} stocks")
        print(f"  Beta stats: mean={beta_all['beta_252d'].mean():.3f}, "
              f"median={beta_all['beta_252d'].median():.3f}, "
              f"std={beta_all['beta_252d'].std():.3f}")

    if len(ivol_all) > 0:
        ivol_all["date"] = ivol_all["ym"].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)
        ivol_all = ivol_all[["permno", "date", "ivol_60d"]]
        print(f"  IVOL: {len(ivol_all):,} rows, {ivol_all['permno'].nunique():,} stocks")
        print(f"  IVOL stats: mean={ivol_all['ivol_60d'].mean():.4f}, "
              f"median={ivol_all['ivol_60d'].median():.4f}, "
              f"std={ivol_all['ivol_60d'].std():.4f}")

    print(f"  Total time: {time.time()-t0:.0f}s")
    return beta_all, ivol_all, df


def compute_regime(df_with_mkt):
    """
    Compute market regime variables from daily market returns.
    - vol_regime: 1 if trailing 3-month realized vol > expanding median
    - trend_regime: 1 if trailing 6-month cumulative market return > 0
    - mkt_ret_6m: trailing 6-month cumulative market return
    """
    print("\nComputing market regime variables...")

    # Get unique daily market returns (already in df)
    mkt_daily = df_with_mkt.groupby(df_with_mkt.index)["mkt_ret"].first().sort_index()

    # Trading days per month ~ 21, 3 months ~ 63 days, 6 months ~ 126 days
    VOL_WINDOW = 63    # ~3 months
    TREND_WINDOW = 126  # ~6 months

    # Trailing realized vol (annualized)
    mkt_vol = mkt_daily.rolling(VOL_WINDOW, min_periods=42).std() * np.sqrt(252)

    # Trailing cumulative return (6 months)
    mkt_cum_ret = mkt_daily.rolling(TREND_WINDOW, min_periods=84).sum()

    # Combine into daily df
    regime_daily = pd.DataFrame({
        "mkt_vol": mkt_vol,
        "mkt_cum_ret_6m": mkt_cum_ret
    })
    regime_daily["ym"] = regime_daily.index.to_period("M")

    # Take last trading day of each month
    regime_monthly = regime_daily.groupby("ym").last().reset_index()

    # Vol regime: 1 if above expanding median
    expanding_median_vol = regime_monthly["mkt_vol"].expanding().median()
    regime_monthly["mkt_vol_regime"] = (regime_monthly["mkt_vol"] > expanding_median_vol).astype(int)

    # Trend regime: 1 if 6-month cumulative return > 0
    regime_monthly["mkt_trend_regime"] = (regime_monthly["mkt_cum_ret_6m"] > 0).astype(int)

    # Rename for clarity
    regime_monthly["mkt_ret_6m"] = regime_monthly["mkt_cum_ret_6m"]

    # Convert to month-end date
    regime_monthly["date"] = regime_monthly["ym"].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)

    result = regime_monthly[["date", "mkt_vol_regime", "mkt_trend_regime", "mkt_ret_6m", "mkt_vol"]].copy()
    result = result.dropna()

    print(f"  {len(result)} monthly observations")
    print(f"  Vol regime distribution: {result['mkt_vol_regime'].value_counts().to_dict()}")
    print(f"  Trend regime distribution: {result['mkt_trend_regime'].value_counts().to_dict()}")
    print(f"  Mkt vol (annualized): mean={result['mkt_vol'].mean():.4f}, std={result['mkt_vol'].std():.4f}")
    print(f"  Mkt ret 6m: mean={result['mkt_ret_6m'].mean():.4f}, std={result['mkt_ret_6m'].std():.4f}")

    # Drop mkt_vol column (intermediate)
    result = result.drop(columns=["mkt_vol"])

    return result


def main():
    print("=" * 60)
    print("DERIVED FEATURES: Beta, IVOL, Market Regime")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Daily market return
    mkt = compute_daily_market_return(df)

    # 3. Beta and IVOL
    beta_df, ivol_df, df_with_mkt = compute_beta_and_ivol(df, mkt)

    # 4. Regime
    regime_df = compute_regime(df_with_mkt)

    # 5. Save
    print("\nSaving outputs...")
    beta_df.to_parquet(f"{DATA_DIR}/stock_beta_monthly.parquet", index=False)
    print(f"  Saved stock_beta_monthly.parquet ({len(beta_df):,} rows)")

    ivol_df.to_parquet(f"{DATA_DIR}/stock_ivol_monthly.parquet", index=False)
    print(f"  Saved stock_ivol_monthly.parquet ({len(ivol_df):,} rows)")

    regime_df.to_parquet(f"{DATA_DIR}/market_regime_monthly.parquet", index=False)
    print(f"  Saved market_regime_monthly.parquet ({len(regime_df):,} rows)")

    # 6. Sample output for verification
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)

    print("\nBeta (first 10 rows):")
    print(beta_df.head(10).to_string(index=False))

    print("\nIVOL (first 10 rows):")
    print(ivol_df.head(10).to_string(index=False))

    print("\nRegime (first 10 rows):")
    print(regime_df.head(10).to_string(index=False))

    # Cross-check with known stock if possible
    # AAPL permno = 14593
    aapl_beta = beta_df[beta_df["permno"] == 14593]
    if len(aapl_beta) > 0:
        print(f"\nAAPL (permno=14593) beta sample:")
        print(aapl_beta.tail(12).to_string(index=False))
    else:
        # Try first available permno
        sample_permno = beta_df["permno"].iloc[0]
        sample = beta_df[beta_df["permno"] == sample_permno].tail(6)
        print(f"\nSample stock (permno={sample_permno}) beta:")
        print(sample.to_string(index=False))

    aapl_ivol = ivol_df[ivol_df["permno"] == 14593]
    if len(aapl_ivol) > 0:
        print(f"\nAAPL (permno=14593) IVOL sample:")
        print(aapl_ivol.tail(12).to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
