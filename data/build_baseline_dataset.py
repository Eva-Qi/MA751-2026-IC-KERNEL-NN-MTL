"""
build_baseline_dataset.py — Build a complete baseline dataset T = (X, y).

Pipeline:
1. Load the raw factor panel from data/raw/
2. Apply factor caps to the 4 baseline raw predictors
3. Median-impute legitimate post-existence gaps by signal_date
4. Drop any remaining predictor NaNs
5. Recompute cross-sectional z-scores on the retained rows
6. Compute 21-trading-day forward returns as the raw target
7. Drop rows with missing raw target
8. Cross-sectionally winsorize + z-score the target by signal_date
8. Write a complete model dataset to data/baseline/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.factor_library.academic_factors import cap_factor_values, cross_sectional_zscore  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BASELINE_DIR = DATA_DIR / "baseline"

DEFAULT_FACTOR_PANEL = RAW_DIR / "factor_panel.parquet"
DEFAULT_PRICES = RAW_DIR / "prices_cache.parquet"
DEFAULT_OUTPUT = BASELINE_DIR / "model_dataset.parquet"

BASELINE_FACTORS = [
    "EarningsYield",
    "AssetGrowth",
    "Accruals",
    "Momentum12_1",
]
TARGET_RAW_COLUMN = "target_forward_return_21d_raw"
TARGET_Z_COLUMN = "target_forward_return_21d_zscore"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-panel", type=Path, default=DEFAULT_FACTOR_PANEL)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _compute_first_real_date(panel: pd.DataFrame) -> dict[str, pd.Timestamp | None]:
    result: dict[str, pd.Timestamp | None] = {}
    for ticker in panel["ticker"].unique():
        ticker_df = panel.loc[panel["ticker"] == ticker]
        has_any = ticker_df[BASELINE_FACTORS].notna().any(axis=1)
        real_dates = ticker_df.loc[has_any, "signal_date"]
        result[ticker] = real_dates.min() if len(real_dates) > 0 else None
    return result


def _cap_predictors(panel: pd.DataFrame) -> pd.DataFrame:
    capped = panel.copy()
    for factor in BASELINE_FACTORS:
        capped[factor] = cap_factor_values(capped[factor], factor)
    return capped


def _median_impute_post_existence(panel: pd.DataFrame) -> pd.DataFrame:
    imputed = panel.copy()
    first_real = _compute_first_real_date(imputed)
    first_real_series = imputed["ticker"].map(first_real)

    for factor in BASELINE_FACTORS:
        for signal_date in sorted(imputed["signal_date"].unique()):
            date_mask = imputed["signal_date"] == signal_date
            existed_mask = first_real_series.notna() & (first_real_series <= signal_date)
            eligible_mask = date_mask & existed_mask

            median = imputed.loc[eligible_mask, factor].median()
            if pd.isna(median):
                continue

            fill_mask = eligible_mask & imputed[factor].isna()
            if fill_mask.any():
                imputed.loc[fill_mask, factor] = median

    return imputed


def _compute_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    scored = panel.copy()
    for factor in BASELINE_FACTORS:
        scored[f"{factor}_zscore"] = pd.NA

    for _, idx in scored.groupby("signal_date").groups.items():
        date_slice = scored.loc[idx]
        for factor in BASELINE_FACTORS:
            zscores = cross_sectional_zscore(date_slice.set_index("ticker")[factor])
            scored.loc[idx, f"{factor}_zscore"] = (
                zscores.reindex(date_slice["ticker"]).to_numpy()
            )

    return scored


def _compute_target(prices: pd.DataFrame, signal_dates: pd.Series) -> pd.DataFrame:
    forward_returns = prices.shift(-21) / prices - 1.0
    monthly_targets = forward_returns.reindex(pd.DatetimeIndex(sorted(signal_dates.unique())))
    monthly_targets.index.name = "signal_date"
    target_df = (
        monthly_targets.stack()
        .rename(TARGET_RAW_COLUMN)
        .reset_index()
    )
    target_df = target_df.rename(columns={"level_1": "ticker", "Ticker": "ticker"})
    return target_df


def _compute_target_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    scored = panel.copy()
    scored[TARGET_Z_COLUMN] = pd.NA

    for _, idx in scored.groupby("signal_date").groups.items():
        date_slice = scored.loc[idx]
        zscores = cross_sectional_zscore(
            date_slice.set_index("ticker")[TARGET_RAW_COLUMN]
        )
        scored.loc[idx, TARGET_Z_COLUMN] = (
            zscores.reindex(date_slice["ticker"]).to_numpy()
        )

    return scored


def main() -> int:
    args = _parse_args()

    raw_panel = pd.read_parquet(args.factor_panel)
    prices = pd.read_parquet(args.prices)

    working = raw_panel[["ticker", "signal_date", *BASELINE_FACTORS]].copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"])

    working = _cap_predictors(working)
    working = _median_impute_post_existence(working)
    working = working.dropna(subset=BASELINE_FACTORS).reset_index(drop=True)
    working = _compute_zscores(working)

    zscore_cols = [f"{factor}_zscore" for factor in BASELINE_FACTORS]
    target_df = _compute_target(prices, working["signal_date"])

    baseline = working[["ticker", "signal_date", *zscore_cols]].merge(
        target_df,
        on=["ticker", "signal_date"],
        how="left",
    )
    baseline = baseline.dropna(subset=[*zscore_cols, TARGET_RAW_COLUMN]).reset_index(drop=True)
    baseline = _compute_target_zscores(baseline)
    baseline = baseline.dropna(subset=[TARGET_Z_COLUMN]).reset_index(drop=True)
    baseline = baseline.drop(columns=[TARGET_RAW_COLUMN])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(args.output, index=False)

    print("Wrote:", args.output)
    print("Rows:", len(baseline))
    print("Tickers:", baseline["ticker"].nunique())
    print("Months:", baseline["signal_date"].nunique())
    print("Columns:", ", ".join(baseline.columns))
    print("Complete:", bool(not baseline.isna().any().any()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
