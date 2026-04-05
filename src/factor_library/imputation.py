"""
imputation.py — Hard Missing-Data Classification + Sector-Aware Imputation

Three types of missing data:
  Type A: Economically Undefined → NEVER impute (wrong sector for this factor)
  Type B: Pre-existence → NEVER impute (company not yet public/available)
  Type C: Post-existence gap → OK to impute with sector median

Key principle: we only impute a ticker×factor×date if the ticker
has REAL data for at least one factor on or before that date.
Otherwise we'd be fabricating history for a non-existent company.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type A: Economically undefined (factor, sector) combinations.
# These get permanent NaN regardless of data availability.
# ---------------------------------------------------------------------------

INVALID_SECTORS: dict[str, set[str]] = {
    "GrossProfitability": {
        "Financials",      # No COGS concept
        "Utilities",       # No meaningful COGS
        "Real Estate",     # Rental income, no COGS
    },
    "NetDebtEBITDA": {
        "Financials",      # Debt IS the product
        "Real Estate",     # REIT leverage measured differently
    },
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

_MIN_NON_NULL_FOR_IMPUTATION = 5


def _compute_first_real_date(
    factor_panel: pd.DataFrame,
) -> dict[str, pd.Timestamp | None]:
    """
    For each ticker, find the earliest date where ANY factor has real data.
    This approximates when the company became available in our dataset.
    Tickers with no real data at all → None (non-existent in sample period).
    """
    result = {}
    for ticker in factor_panel["ticker"].unique():
        t_data = factor_panel[factor_panel["ticker"] == ticker]
        has_any = t_data[RAW_FACTOR_NAMES].notna().any(axis=1)
        real_dates = t_data.loc[has_any, "signal_date"]
        result[ticker] = real_dates.min() if len(real_dates) > 0 else None
    return result


def impute_factor_panel(
    factor_panel: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    Apply hard classification + time-aware sector median imputation.

    Rules:
    1. Type A: (factor, sector) economically undefined → force NaN
    2. Pre-existence: date < ticker's first real data date → keep NaN (not yet public)
    3. Post-existence gap: date >= first real date, factor is NaN → sector median impute

    This ensures we never fabricate history for companies that didn't exist yet.
    """
    df = factor_panel.copy()
    df["_sector"] = df["ticker"].map(sector_map)

    # Compute when each ticker first appears with real data
    first_real = _compute_first_real_date(df)
    n_nonexistent = sum(1 for v in first_real.values() if v is None)
    n_late = sum(1 for v in first_real.values()
                 if v is not None and v > pd.Timestamp("2015-02-01"))

    logger.info(
        "impute: %d tickers never had data (non-existent), "
        "%d tickers appeared after start of sample",
        n_nonexistent, n_late,
    )

    total_imputed = 0
    total_type_a = 0
    total_pre_existence_skipped = 0

    for factor in RAW_FACTOR_NAMES:
        if factor not in df.columns:
            continue

        imputed_col = f"{factor}_imputed"
        df[imputed_col] = False
        invalid_sectors = INVALID_SECTORS.get(factor, set())

        # --- Phase 1: Force NaN for Type A ---
        if invalid_sectors:
            type_a_mask = df["_sector"].isin(invalid_sectors)
            df.loc[type_a_mask, factor] = np.nan
            for col_suffix in ["_zscore", "_quintile"]:
                col = f"{factor}{col_suffix}"
                if col in df.columns:
                    df.loc[type_a_mask, col] = np.nan
            total_type_a += type_a_mask.sum()

        # --- Phase 2: Sector median imputation (post-existence only) ---
        factor_imputed = 0
        factor_pre_skipped = 0

        for date in df["signal_date"].unique():
            date_mask = df["signal_date"] == date

            for sector_name in df.loc[date_mask, "_sector"].dropna().unique():
                if sector_name in invalid_sectors:
                    continue

                sector_date_mask = date_mask & (df["_sector"] == sector_name)
                sector_df = df.loc[sector_date_mask]

                # Only consider tickers that existed by this date
                eligible_tickers = []
                for _, row in sector_df.iterrows():
                    t = row["ticker"]
                    frd = first_real.get(t)
                    if frd is not None and date >= frd:
                        eligible_tickers.append(row.name)  # DataFrame index
                    elif pd.isna(row[factor]):
                        factor_pre_skipped += 1

                if not eligible_tickers:
                    continue

                eligible_mask = df.index.isin(eligible_tickers)
                eligible_values = df.loc[eligible_mask, factor]

                n_valid = eligible_values.notna().sum()
                if n_valid < _MIN_NON_NULL_FOR_IMPUTATION:
                    continue

                sector_median = eligible_values.median()
                if pd.isna(sector_median):
                    continue

                null_eligible = eligible_mask & df[factor].isna()
                n_to_impute = null_eligible.sum()
                if n_to_impute > 0:
                    df.loc[null_eligible, factor] = sector_median
                    df.loc[null_eligible, imputed_col] = True
                    factor_imputed += n_to_impute

        total_imputed += factor_imputed
        total_pre_existence_skipped += factor_pre_skipped

        logger.info(
            "impute: %s — imputed %d (post-existence gaps), "
            "skipped %d (pre-existence)",
            factor, factor_imputed, factor_pre_skipped,
        )

    df = df.drop(columns=["_sector"])

    logger.info(
        "impute_factor_panel: %d Type-A forced NaN, %d imputed, "
        "%d pre-existence skipped",
        total_type_a, total_imputed, total_pre_existence_skipped,
    )

    return df
