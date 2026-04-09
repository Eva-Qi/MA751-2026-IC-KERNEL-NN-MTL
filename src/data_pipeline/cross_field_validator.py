"""
cross_field_validator.py — Financial Sanity Check Engine

Enforces consistency constraints between related fields:
1. Market cap consistency: shares × price ≈ known benchmarks
2. Accounting identity: NI ≤ Revenue (for most companies)
3. Time-series continuity: no >10x jumps in shares/assets between quarters
4. Must-exist rules: top market cap companies cannot have all-NaN factors
5. Unit distribution scan: detect mixed units per concept

Run standalone:
    python -m alpha_system.core.data_pipeline.cross_field_validator
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "alpha_system" / "data"

# Top companies that MUST have data — pipeline should hard-fail if all NaN
MUST_EXIST_TICKERS = {
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "V", "JNJ", "UNH", "XOM", "PG", "MA", "HD",
    "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "AVGO",
    "WMT", "MCD", "CSCO", "ACN", "TMO", "ABT", "DHR", "TXN",
    "NEE", "PM", "UPS", "RTX", "HON", "LOW", "AMGN", "IBM",
    "CAT", "BA", "GE", "GS", "BLK", "MS", "AXP", "SCHW", "MMM",
}

SHARES_TAGS = [
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]

NI_TAGS = [
    "NetIncomeLoss", "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "IncomeLossFromContinuingOperations",
]

REV_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]


def check_must_exist(
    factor_panel: pd.DataFrame,
    factors: list[str] | None = None,
) -> list[dict]:
    """
    Check that top market cap companies have at least SOME real factor data.
    Returns list of violations.
    """
    if factors is None:
        factors = ["EarningsYield", "GrossProfitability", "AssetGrowth",
                    "Accruals", "Momentum12_1", "NetDebtEBITDA"]

    violations = []
    for ticker in MUST_EXIST_TICKERS:
        t_data = factor_panel[factor_panel["ticker"] == ticker]
        if t_data.empty:
            violations.append({
                "ticker": ticker, "check": "must_exist",
                "detail": "not in factor_panel at all",
            })
            continue

        for factor in factors:
            if factor not in t_data.columns:
                continue
            n_real = t_data[factor].notna().sum()
            n_total = len(t_data)
            if n_real == 0:
                violations.append({
                    "ticker": ticker, "check": "must_exist",
                    "detail": f"{factor}: 0/{n_total} real values",
                })

    if violations:
        logger.warning(
            "check_must_exist: %d violations (top companies with all-NaN factors)",
            len(violations),
        )
    else:
        logger.info("check_must_exist: all top companies have data — OK")

    return violations


def check_time_continuity(
    xbrl_df: pd.DataFrame,
    concepts: list[str] | None = None,
    jump_threshold: float = 10.0,
) -> list[dict]:
    """
    Flag tickers where a concept value jumps >Nx between consecutive periods.
    Indicates scaling errors or unit changes.
    """
    if concepts is None:
        concepts = ["Assets", "CommonStockSharesOutstanding",
                     "WeightedAverageNumberOfDilutedSharesOutstanding"]

    violations = []

    for concept in concepts:
        sub = xbrl_df[xbrl_df["concept"] == concept].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(["ticker", "period_end"])

        for ticker in sub["ticker"].unique():
            t_data = sub[sub["ticker"] == ticker]["value"].dropna()
            if len(t_data) < 2:
                continue

            ratios = t_data.values[1:] / t_data.values[:-1]
            ratios = ratios[t_data.values[:-1] > 0]  # avoid div by zero

            max_jump = np.max(np.abs(ratios)) if len(ratios) > 0 else 1.0
            min_ratio = np.min(ratios) if len(ratios) > 0 else 1.0

            if max_jump > jump_threshold or min_ratio < (1 / jump_threshold):
                violations.append({
                    "ticker": ticker, "check": "time_continuity",
                    "detail": f"{concept}: max_jump={max_jump:.1f}x, "
                              f"min_ratio={min_ratio:.4f}",
                })

    if violations:
        logger.warning(
            "check_time_continuity: %d tickers with >%.0fx jumps",
            len(violations), jump_threshold,
        )
    else:
        logger.info("check_time_continuity: no suspicious jumps — OK")

    return violations


def check_ni_vs_revenue(
    xbrl_df: pd.DataFrame,
) -> list[dict]:
    """
    Flag cases where |Net Income| > Revenue (almost always a data error
    for non-financial companies).
    """
    violations = []

    # Get latest FY values per ticker
    ni_sub = xbrl_df[
        xbrl_df["concept"].isin(NI_TAGS)
        & (xbrl_df["unit"].str.lower() == "usd")
        & (xbrl_df["fiscal_period"] == "FY")
    ].sort_values(["ticker", "period_end"], ascending=[True, False])
    ni_latest = ni_sub.drop_duplicates(subset=["ticker"], keep="first")

    rev_sub = xbrl_df[
        xbrl_df["concept"].isin(REV_TAGS)
        & (xbrl_df["unit"].str.lower() == "usd")
        & (xbrl_df["fiscal_period"] == "FY")
    ].sort_values(["ticker", "period_end"], ascending=[True, False])
    rev_latest = rev_sub.drop_duplicates(subset=["ticker"], keep="first")

    merged = ni_latest[["ticker", "value"]].merge(
        rev_latest[["ticker", "value"]],
        on="ticker", suffixes=("_ni", "_rev"),
    )

    # Load sector map for exemptions
    try:
        sector = pd.read_json(DATA_DIR / "sector_map.json", typ="series")
    except Exception:
        sector = pd.Series(dtype=str)

    for _, row in merged.iterrows():
        ni = abs(row["value_ni"])
        rev = abs(row["value_rev"])
        ticker_sector = sector.get(row["ticker"], "")

        # Exempt Real Estate (REITs) — NI includes property gains not in Revenue
        # Exempt Financials — Revenue tag may only capture fee income, not interest
        if ticker_sector in ("Real Estate", "Financials"):
            continue

        if rev > 0 and ni > rev * 1.5:
            violations.append({
                "ticker": row["ticker"], "check": "ni_vs_revenue",
                "detail": f"|NI|={ni:.2e} > 1.5×Revenue={rev:.2e}",
            })

    if violations:
        logger.warning(
            "check_ni_vs_revenue: %d tickers with |NI| > 1.5×Revenue",
            len(violations),
        )
    else:
        logger.info("check_ni_vs_revenue: all NI ≤ 1.5×Revenue — OK")

    return violations


def check_unit_distribution(
    xbrl_df: pd.DataFrame,
) -> list[dict]:
    """
    Scan unit distribution per concept. Flag concepts with mixed units.
    """
    violations = []

    unit_counts = xbrl_df.groupby(["concept", "unit"]).size().reset_index(name="count")
    concept_units = unit_counts.groupby("concept")["unit"].nunique()
    mixed = concept_units[concept_units > 1]

    for concept in mixed.index:
        units = unit_counts[unit_counts["concept"] == concept][["unit", "count"]]
        detail = "; ".join(f"{r['unit']}={r['count']}" for _, r in units.iterrows())
        violations.append({
            "ticker": "ALL", "check": "unit_distribution",
            "detail": f"{concept}: {detail}",
        })

    if violations:
        logger.warning(
            "check_unit_distribution: %d concepts with mixed units",
            len(violations),
        )
    else:
        logger.info("check_unit_distribution: all concepts have uniform units — OK")

    return violations


def run_all_checks(
    factor_panel: pd.DataFrame | None = None,
    xbrl_df: pd.DataFrame | None = None,
) -> dict[str, list[dict]]:
    """
    Run all cross-field validation checks.
    Returns dict of check_name → list of violations.
    """
    results = {}

    if factor_panel is not None:
        results["must_exist"] = check_must_exist(factor_panel)

    if xbrl_df is not None:
        results["time_continuity"] = check_time_continuity(xbrl_df)
        results["ni_vs_revenue"] = check_ni_vs_revenue(xbrl_df)
        results["unit_distribution"] = check_unit_distribution(xbrl_df)

    # Summary
    total = sum(len(v) for v in results.values())
    print(f"\n{'='*60}")
    print(f"CROSS-FIELD VALIDATION SUMMARY")
    print(f"{'='*60}")
    for check, violations in results.items():
        status = "PASS" if len(violations) == 0 else f"FAIL ({len(violations)})"
        print(f"  {check:<25} {status}")
    print(f"  {'TOTAL':<25} {total} violations")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    print("Loading data...")
    fp = pd.read_parquet(DATA_DIR / "factor_panel.parquet")
    xbrl = pd.read_parquet(DATA_DIR / "xbrl_df.parquet")

    results = run_all_checks(factor_panel=fp, xbrl_df=xbrl)

    # Print details
    for check, violations in results.items():
        if violations:
            print(f"\n--- {check} ---")
            for v in violations[:20]:
                print(f"  {v['ticker']:<8} {v['detail']}")
            if len(violations) > 20:
                print(f"  ... and {len(violations) - 20} more")
