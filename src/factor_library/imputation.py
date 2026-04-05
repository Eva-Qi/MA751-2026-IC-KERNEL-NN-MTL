"""
imputation.py — Hard Missing-Data Classification + Sector-Aware Imputation

Three types of missing data:
  Type A: Economically Undefined → NEVER impute, NEVER rank
  Type B: Temporal (IPO ramp) → OK to impute with sector median
  Type C: Random/sporadic → OK to impute with sector median

Type A is determined by an explicit (factor, sector) validity map,
NOT by an arbitrary threshold.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard classification: which (factor, sector) combinations are economically
# undefined and should NEVER be imputed or ranked.
# ---------------------------------------------------------------------------

# Type A sectors per factor — these get permanent NaN.
# Any sector NOT listed here is considered valid (Type B/C).
INVALID_SECTORS: dict[str, set[str]] = {
    "GrossProfitability": {
        "Financials",      # No COGS concept — revenue is interest/fee income
        "Utilities",       # Regulated tariff model, no meaningful COGS
        "Real Estate",     # Rental income, no COGS
    },
    "NetDebtEBITDA": {
        "Financials",      # Debt IS the product (deposits); EBITDA excludes core business
        "Real Estate",     # REIT leverage measured differently (Debt/FFO standard)
    },
    # All other factors: valid for all sectors
    "EarningsYield": set(),
    "AssetGrowth": set(),
    "Accruals": set(),
    "Momentum12_1": set(),
}

RAW_FACTOR_NAMES = [
    "EarningsYield",
    "GrossProfitability",
    "AssetGrowth",
    "Accruals",
    "Momentum12_1",
    "NetDebtEBITDA",
]

# Tickers that did not exist during the sample period (2015-2022).
# They appear in the panel because SEC lists current S&P 500 members,
# but imputing their data would fabricate non-existent company history.
# These should NEVER be imputed — keep all-NaN or drop entirely.
NONEXISTENT_TICKERS: set[str] = {
    "GEHC",   # GE Healthcare — 2023 spin-off from GE
    "KVUE",   # Kenvue — 2023 spin-off from J&J
    "CEG",    # Constellation Energy — 2022 spin-off from Exelon
    "SOLV",   # Solventum — 2024 spin-off from 3M
    "VLTO",   # Veralto — 2023 spin-off from Danaher
    "GEV",    # GE Vernova — 2024 spin-off from GE
}

# Only impute if the sector has enough non-NaN values to compute a meaningful median
_MIN_NON_NULL_FOR_IMPUTATION = 5


def impute_factor_panel(
    factor_panel: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    Apply hard classification + sector-aware imputation.

    1. Type A (factor, sector) combinations: force NaN — never impute
    2. Type B/C: cross-sectional median imputation per sector per date
    3. Track all imputed values in {factor}_imputed columns

    Parameters
    ----------
    factor_panel : pd.DataFrame
        Must contain 'ticker', 'signal_date', and raw factor columns.
    sector_map : pd.Series
        index=ticker, values=sector string.

    Returns
    -------
    pd.DataFrame
        Same structure with imputed values + {factor}_imputed columns.
    """
    df = factor_panel.copy()
    df["_sector"] = df["ticker"].map(sector_map)

    # Guard: never impute tickers that didn't exist during sample period
    nonexistent_mask = df["ticker"].isin(NONEXISTENT_TICKERS)
    n_nonexistent = nonexistent_mask.sum()
    if n_nonexistent > 0:
        logger.info(
            "impute: skipping %d rows for %d non-existent tickers: %s",
            n_nonexistent, len(NONEXISTENT_TICKERS), sorted(NONEXISTENT_TICKERS),
        )

    total_imputed = 0
    total_forced_nan = 0

    for factor in RAW_FACTOR_NAMES:
        if factor not in df.columns:
            continue

        imputed_col = f"{factor}_imputed"
        df[imputed_col] = False

        invalid_sectors = INVALID_SECTORS.get(factor, set())

        # --- Phase 1: Force NaN for Type A (economically undefined) ---
        if invalid_sectors:
            type_a_mask = df["_sector"].isin(invalid_sectors)
            n_forced = type_a_mask.sum()
            # Force raw value, z-score, and quintile to NaN
            df.loc[type_a_mask, factor] = np.nan
            zscore_col = f"{factor}_zscore"
            quintile_col = f"{factor}_quintile"
            if zscore_col in df.columns:
                df.loc[type_a_mask, zscore_col] = np.nan
            if quintile_col in df.columns:
                df.loc[type_a_mask, quintile_col] = np.nan
            total_forced_nan += n_forced
            logger.info(
                "impute: %s — forced %d Type-A NaN (sectors: %s)",
                factor, n_forced, sorted(invalid_sectors),
            )

        # --- Phase 2: Sector median imputation for Type B/C ---
        factor_imputed_count = 0

        for date in df["signal_date"].unique():
            date_mask = df["signal_date"] == date

            for sector in df.loc[date_mask, "_sector"].dropna().unique():
                # Skip Type A sectors
                if sector in invalid_sectors:
                    continue

                sector_mask = date_mask & (df["_sector"] == sector)
                # Exclude non-existent tickers from imputation candidates
                valid_sector_mask = sector_mask & ~nonexistent_mask
                sector_values = df.loc[valid_sector_mask, factor]

                n_missing = sector_values.isna().sum()
                if n_missing == 0:
                    continue

                n_valid = sector_values.notna().sum()
                if n_valid < _MIN_NON_NULL_FOR_IMPUTATION:
                    continue

                sector_median = sector_values.median()
                if pd.isna(sector_median):
                    continue

                null_mask = valid_sector_mask & df[factor].isna()
                n_to_impute = null_mask.sum()
                if n_to_impute > 0:
                    df.loc[null_mask, factor] = sector_median
                    df.loc[null_mask, imputed_col] = True
                    factor_imputed_count += n_to_impute

        total_imputed += factor_imputed_count
        logger.info(
            "impute: %s — imputed %d values (Type B/C)",
            factor, factor_imputed_count,
        )

    df = df.drop(columns=["_sector"])

    logger.info(
        "impute_factor_panel: %d Type-A forced NaN, %d Type-B/C imputed",
        total_forced_nan, total_imputed,
    )

    return df
