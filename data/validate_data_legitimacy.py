"""
validate_data_legitimacy.py — Data Legitimacy Self-Check Script

Standalone read-only diagnostic. Reads existing parquet files, produces
a markdown report identifying data quality issues in the factor pipeline.

Usage:
    python validate_data_legitimacy.py [--output report.md]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

XBRL_PATH = RAW_DIR / "xbrl_df.parquet"
FACTOR_PATH = RAW_DIR / "factor_panel.parquet"
PRICES_PATH = RAW_DIR / "prices_cache.parquet"

BENCHMARK_TICKERS = ["AAPL", "MSFT", "GOOG", "JPM", "XOM"]
SPOT_CHECK_DATE = "2022-06-30"

# Expected ranges for S&P 500
EXPECTED_MEDIAN_EY = (0.02, 0.08)
MAX_REASONABLE_EY = 1.0
MIN_PLAUSIBLE_SHARES = 1e6
MAX_PLAUSIBLE_SHARES = 1e11
MIN_SP500_MARKET_CAP = 1e9  # $1B

# Known public data for cross-validation (approximate, FY ending in or near 2022)
# Format: {ticker: (net_income_approx, shares_approx, ey_approx)}
KNOWN_VALUES = {
    "AAPL": {"ni": 99.8e9, "shares": 15.94e9, "ey_approx": 0.035, "label": "FY2022 (Sep)"},
    "MSFT": {"ni": 72.7e9, "shares": 7.47e9, "ey_approx": 0.037, "label": "FY2022 (Jun)"},
    "GOOG": {"ni": 76.0e9, "shares": 13.2e9, "ey_approx": 0.050, "label": "FY2022 (Dec)"},
    "JPM":  {"ni": 37.7e9, "shares": 2.95e9, "ey_approx": 0.107, "label": "FY2022 (Dec)"},
    "XOM":  {"ni": 55.7e9, "shares": 4.15e9, "ey_approx": 0.130, "label": "FY2022 (Dec)"},
}

# NET_INCOME concept tags (from taxonomy_map.py)
NI_TAGS = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
SHARES_TAGS = [
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]

# Financial concepts that should have unit=USD
USD_CONCEPTS = [
    "NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic",
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet", "SalesRevenueGoodsNet",
    "GrossProfit", "Assets", "LongTermDebt",
    "CashAndCashEquivalentsAtCarryingValue", "OperatingIncomeLoss",
    "StockholdersEquity", "AssetsCurrent", "LiabilitiesCurrent",
    "NetCashProvidedByUsedInOperatingActivities",
    "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load all three parquet files."""
    xbrl = pd.read_parquet(XBRL_PATH)
    factor = pd.read_parquet(FACTOR_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    return xbrl, factor, prices


def get_latest_xbrl_value(xbrl, ticker, concept_tags, as_of_date, unit_filter=None):
    """Get the most recent value for a ticker/concept available by as_of_date."""
    mask = (
        (xbrl["ticker"] == ticker)
        & (xbrl["concept"].isin(concept_tags))
        & (xbrl["available_date"] <= as_of_date)
    )
    if unit_filter:
        mask = mask & (xbrl["unit"].str.lower() == unit_filter.lower())

    sub = xbrl.loc[mask].copy()
    if sub.empty:
        return None, None, None

    sub = sub.sort_values(["period_end", "available_date"], ascending=[False, False])
    best = sub.iloc[0]
    return best["value"], best["concept"], best["unit"]


def get_price_at_date(prices, ticker, as_of_date):
    """Get most recent price <= as_of_date."""
    if ticker not in prices.columns:
        return None
    ts = pd.Timestamp(as_of_date)
    sub = prices.loc[prices.index <= ts, ticker].dropna()
    if sub.empty:
        return None
    return float(sub.iloc[-1])


# ---------------------------------------------------------------------------
# Check 1: Spot-Check Known Companies
# ---------------------------------------------------------------------------

def check1_spot_check(xbrl, prices, report_lines):
    report_lines.append("## Check 1: Spot-Check Known Companies\n")
    report_lines.append(f"Date: {SPOT_CHECK_DATE}\n")

    header = "| Ticker | Component | Our Value | Expected | Ratio | Status |"
    sep = "|--------|-----------|-----------|----------|-------|--------|"
    rows = []

    issues = 0
    for ticker in BENCHMARK_TICKERS:
        ni_val, ni_tag, ni_unit = get_latest_xbrl_value(
            xbrl, ticker, NI_TAGS, SPOT_CHECK_DATE, unit_filter="USD"
        )
        sh_val, sh_tag, sh_unit = get_latest_xbrl_value(
            xbrl, ticker, SHARES_TAGS, SPOT_CHECK_DATE, unit_filter="shares"
        )
        price = get_price_at_date(prices, ticker, SPOT_CHECK_DATE)

        known = KNOWN_VALUES.get(ticker, {})

        # Net income check
        if ni_val is not None and known.get("ni"):
            ratio = ni_val / known["ni"]
            status = "PASS" if 0.3 < ratio < 3.0 else "**FAIL**"
            if status == "**FAIL**":
                issues += 1
            rows.append(f"| {ticker} | Net Income | {ni_val:.2e} ({ni_tag}) | {known['ni']:.2e} | {ratio:.2f}x | {status} |")
        elif ni_val is None:
            rows.append(f"| {ticker} | Net Income | MISSING | {known.get('ni', '?')} | N/A | **FAIL** |")
            issues += 1

        # Shares check
        if sh_val is not None and known.get("shares"):
            ratio = sh_val / known["shares"]
            status = "PASS" if 0.3 < ratio < 3.0 else "**FAIL**"
            if status == "**FAIL**":
                issues += 1
            rows.append(f"| {ticker} | Shares | {sh_val:.2e} ({sh_tag}) | {known['shares']:.2e} | {ratio:.2f}x | {status} |")
        elif sh_val is None:
            rows.append(f"| {ticker} | Shares | MISSING | {known.get('shares', '?')} | N/A | **FAIL** |")
            issues += 1

        # Market cap & EY check
        if ni_val is not None and sh_val is not None and price is not None and sh_val > 0:
            mkt_cap = price * sh_val
            ey = ni_val / mkt_cap

            if known.get("ey_approx"):
                ey_ratio = ey / known["ey_approx"]
                status = "PASS" if 0.3 < ey_ratio < 3.0 else "**FAIL**"
                if status == "**FAIL**":
                    issues += 1
                rows.append(
                    f"| {ticker} | EY (computed) | {ey:.4f} | ~{known['ey_approx']:.3f} | {ey_ratio:.2f}x | {status} |"
                )
            rows.append(f"| {ticker} | Market Cap | ${mkt_cap:.2e} | — | — | info |")
            rows.append(f"| {ticker} | Price (adj) | ${price:.2f} | — | — | info |")

        rows.append(f"| | | | | | |")

    report_lines.append(header)
    report_lines.append(sep)
    report_lines.extend(rows)
    report_lines.append("")

    verdict = "PASS" if issues == 0 else f"**FAIL** ({issues} issues)"
    report_lines.append(f"**Check 1 Verdict**: {verdict}\n")
    return issues


# ---------------------------------------------------------------------------
# Check 2: Distribution Analysis
# ---------------------------------------------------------------------------

def check2_distribution(factor, report_lines):
    report_lines.append("## Check 2: EarningsYield Distribution Analysis\n")

    ey = factor["EarningsYield"].dropna()
    report_lines.append(f"Total observations: {len(ey)}\n")

    # Basic stats
    stats = {
        "Mean": ey.mean(),
        "Median": ey.median(),
        "Std": ey.std(),
        "Min": ey.min(),
        "Max": ey.max(),
        "Skewness": ey.skew(),
        "Kurtosis": ey.kurtosis(),
    }

    report_lines.append("### Basic Statistics\n")
    report_lines.append("| Statistic | Value | Assessment |")
    report_lines.append("|-----------|-------|------------|")
    for name, val in stats.items():
        assessment = ""
        if name == "Median":
            if EXPECTED_MEDIAN_EY[0] <= val <= EXPECTED_MEDIAN_EY[1]:
                assessment = "OK (expected 0.02-0.08)"
            else:
                assessment = f"**ABNORMAL** (expected 0.02-0.08)"
        elif name == "Std":
            assessment = "**EXTREME**" if val > 1.0 else "OK"
        elif name == "Skewness":
            assessment = "**EXTREME**" if abs(val) > 5 else "OK"
        elif name == "Kurtosis":
            assessment = "**EXTREME**" if val > 20 else "OK"
        report_lines.append(f"| {name} | {val:.4f} | {assessment} |")
    report_lines.append("")

    # Percentiles
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    report_lines.append("### Percentiles\n")
    report_lines.append("| Percentile | Value |")
    report_lines.append("|------------|-------|")
    for p in pcts:
        report_lines.append(f"| {p}% | {ey.quantile(p/100):.6f} |")
    report_lines.append("")

    # Outlier counts
    extreme = (ey.abs() > MAX_REASONABLE_EY).sum()
    mega = (ey.abs() > 100).sum()
    negative = (ey < 0).sum()

    report_lines.append("### Outlier Counts\n")
    report_lines.append("| Category | Count | % of Total |")
    report_lines.append("|----------|-------|------------|")
    report_lines.append(f"| \\|EY\\| > 1.0 | {extreme} | {100*extreme/len(ey):.2f}% |")
    report_lines.append(f"| \\|EY\\| > 100 | {mega} | {100*mega/len(ey):.2f}% |")
    report_lines.append(f"| EY < 0 (loss) | {negative} | {100*negative/len(ey):.2f}% |")
    report_lines.append("")

    # Per-date median
    per_date = factor.groupby("signal_date")["EarningsYield"].median()
    abnormal_dates = per_date[
        (per_date < EXPECTED_MEDIAN_EY[0]) | (per_date > EXPECTED_MEDIAN_EY[1])
    ]
    report_lines.append(f"### Per-Date Median EY\n")
    report_lines.append(f"Dates with median EY outside {EXPECTED_MEDIAN_EY}: **{len(abnormal_dates)}** of {len(per_date)}\n")
    if len(abnormal_dates) > 0:
        report_lines.append("| Date | Median EY |")
        report_lines.append("|------|-----------|")
        for dt, val in abnormal_dates.head(10).items():
            report_lines.append(f"| {dt.strftime('%Y-%m-%d')} | {val:.4f} |")
        if len(abnormal_dates) > 10:
            report_lines.append(f"| ... | ({len(abnormal_dates)-10} more) |")
    report_lines.append("")

    issues = 0
    if extreme > 0:
        issues += 1
    if not (EXPECTED_MEDIAN_EY[0] <= ey.median() <= EXPECTED_MEDIAN_EY[1]):
        issues += 1
    if ey.std() > 1.0:
        issues += 1

    verdict = "PASS" if issues == 0 else f"**FAIL** ({issues} issues)"
    report_lines.append(f"**Check 2 Verdict**: {verdict}\n")
    return issues


# ---------------------------------------------------------------------------
# Check 3: Unit Consistency Audit
# ---------------------------------------------------------------------------

def check3_unit_consistency(xbrl, report_lines):
    report_lines.append("## Check 3: Unit Consistency Audit\n")

    issues = 0

    # 3a: Financial concepts should be USD
    financial_mask = xbrl["concept"].isin(USD_CONCEPTS)
    financial_rows = xbrl.loc[financial_mask]
    non_usd = financial_rows[financial_rows["unit"].str.lower() != "usd"]

    report_lines.append("### 3a: Financial Concepts with Non-USD Units\n")
    if len(non_usd) > 0:
        issues += 1
        grouped = non_usd.groupby(["ticker", "concept", "unit"]).size().reset_index(name="count")
        report_lines.append(f"Found **{len(non_usd)}** non-USD records across {grouped['ticker'].nunique()} tickers:\n")
        report_lines.append("| Ticker | Concept | Unit | Count |")
        report_lines.append("|--------|---------|------|-------|")
        for _, row in grouped.iterrows():
            report_lines.append(f"| {row['ticker']} | {row['concept']} | {row['unit']} | {row['count']} |")
    else:
        report_lines.append("All financial concept records have unit=USD. **OK**\n")
    report_lines.append("")

    # 3b: Shares concepts should be "shares"
    shares_mask = xbrl["concept"].isin(SHARES_TAGS)
    shares_rows = xbrl.loc[shares_mask]
    non_shares = shares_rows[shares_rows["unit"].str.lower() != "shares"]

    report_lines.append("### 3b: Shares Concepts with Non-Shares Units\n")
    if len(non_shares) > 0:
        issues += 1
        grouped = non_shares.groupby(["ticker", "concept", "unit"]).size().reset_index(name="count")
        report_lines.append(f"Found **{len(non_shares)}** non-shares records across {grouped['ticker'].nunique()} tickers:\n")
        report_lines.append("| Ticker | Concept | Unit | Count |")
        report_lines.append("|--------|---------|------|-------|")
        for _, row in grouped.iterrows():
            report_lines.append(f"| {row['ticker']} | {row['concept']} | {row['unit']} | {row['count']} |")
    else:
        report_lines.append("All shares concept records have unit=shares. **OK**\n")
    report_lines.append("")

    # 3c: Shares scaling consistency (max/min ratio per ticker)
    report_lines.append("### 3c: Shares Scaling Consistency\n")

    shares_valid = shares_rows[
        (shares_rows["unit"].str.lower() == "shares")
        & (shares_rows["value"] > 0)
    ]
    ticker_stats = shares_valid.groupby("ticker")["value"].agg(["min", "max"])
    ticker_stats["ratio"] = ticker_stats["max"] / ticker_stats["min"]
    suspicious = ticker_stats[ticker_stats["ratio"] > 1000].sort_values("ratio", ascending=False)

    if len(suspicious) > 0:
        issues += 1
        report_lines.append(f"Found **{len(suspicious)}** tickers with shares max/min ratio > 1000x (likely scaling errors):\n")
        report_lines.append("| Ticker | Min Shares | Max Shares | Ratio |")
        report_lines.append("|--------|-----------|-----------|-------|")
        for ticker, row in suspicious.head(20).iterrows():
            report_lines.append(f"| {ticker} | {row['min']:.2e} | {row['max']:.2e} | {row['ratio']:.0f}x |")
        if len(suspicious) > 20:
            report_lines.append(f"| ... | | | ({len(suspicious)-20} more) |")
    else:
        report_lines.append("No tickers with shares max/min ratio > 1000x. **OK**\n")
    report_lines.append("")

    verdict = "PASS" if issues == 0 else f"**FAIL** ({issues} issues)"
    report_lines.append(f"**Check 3 Verdict**: {verdict}\n")
    return issues


# ---------------------------------------------------------------------------
# Check 4: Magnitude Sanity Check
# ---------------------------------------------------------------------------

def check4_magnitude(xbrl, factor, prices, report_lines):
    report_lines.append("## Check 4: Magnitude Sanity Check\n")

    issues = 0

    # 4a: EY outliers in factor_panel
    ey = factor["EarningsYield"].dropna()
    extreme_mask = ey.abs() > MAX_REASONABLE_EY
    extreme_records = factor.loc[ey.index[extreme_mask], ["ticker", "signal_date", "EarningsYield"]]

    report_lines.append("### 4a: EarningsYield |EY| > 1.0\n")
    if len(extreme_records) > 0:
        issues += 1
        top_offenders = (
            extreme_records.groupby("ticker")["EarningsYield"]
            .agg(["count", "min", "max"])
            .sort_values("count", ascending=False)
        )
        report_lines.append(f"Found **{len(extreme_records)}** records with |EY| > 1.0 across {len(top_offenders)} tickers:\n")
        report_lines.append("| Ticker | Count | Min EY | Max EY |")
        report_lines.append("|--------|-------|--------|--------|")
        for ticker, row in top_offenders.head(20).iterrows():
            report_lines.append(f"| {ticker} | {row['count']} | {row['min']:.2f} | {row['max']:.2f} |")
        if len(top_offenders) > 20:
            report_lines.append(f"| ... | | | ({len(top_offenders)-20} more) |")
    else:
        report_lines.append("No records with |EY| > 1.0. **OK**\n")
    report_lines.append("")

    # 4b: Implausible shares values
    shares_data = xbrl[xbrl["concept"].isin(SHARES_TAGS) & (xbrl["unit"].str.lower() == "shares")]

    too_small = shares_data[shares_data["value"] < MIN_PLAUSIBLE_SHARES]
    too_large = shares_data[shares_data["value"] > MAX_PLAUSIBLE_SHARES]
    zero_shares = shares_data[shares_data["value"] == 0]

    report_lines.append("### 4b: Implausible Shares Outstanding\n")
    report_lines.append("| Category | Records | Tickers |")
    report_lines.append("|----------|---------|---------|")
    report_lines.append(f"| Shares < {MIN_PLAUSIBLE_SHARES:.0e} | {len(too_small)} | {too_small['ticker'].nunique()} |")
    report_lines.append(f"| Shares > {MAX_PLAUSIBLE_SHARES:.0e} | {len(too_large)} | {too_large['ticker'].nunique()} |")
    report_lines.append(f"| Shares == 0 | {len(zero_shares)} | {zero_shares['ticker'].nunique()} |")
    report_lines.append("")

    if len(too_small) > 0 or len(too_large) > 0 or len(zero_shares) > 0:
        issues += 1
        all_bad = pd.concat([too_small, too_large, zero_shares])
        bad_tickers = all_bad.groupby("ticker")["value"].agg(["count", "min", "max"]).sort_values("count", ascending=False)
        report_lines.append("Top offenders:\n")
        report_lines.append("| Ticker | Bad Records | Min Value | Max Value |")
        report_lines.append("|--------|------------|-----------|-----------|")
        for ticker, row in bad_tickers.head(15).iterrows():
            report_lines.append(f"| {ticker} | {row['count']} | {row['min']:.2e} | {row['max']:.2e} |")
    report_lines.append("")

    # 4c: Market cap < $1B (computed for latest date in factor_panel)
    report_lines.append("### 4c: Implausible Market Caps\n")

    latest_date = factor["signal_date"].max()
    latest_date_str = latest_date.strftime("%Y-%m-%d")

    small_mktcap_tickers = []
    for ticker in factor.loc[factor["signal_date"] == latest_date, "ticker"].unique():
        sh_val, _, _ = get_latest_xbrl_value(xbrl, ticker, SHARES_TAGS, latest_date_str, "shares")
        price = get_price_at_date(prices, ticker, latest_date_str)
        if sh_val is not None and price is not None and sh_val > 0:
            mkt_cap = price * sh_val
            if mkt_cap < MIN_SP500_MARKET_CAP:
                small_mktcap_tickers.append((ticker, mkt_cap, price, sh_val))

    if small_mktcap_tickers:
        issues += 1
        report_lines.append(f"Found **{len(small_mktcap_tickers)}** tickers with computed market_cap < $1B at {latest_date_str}:\n")
        report_lines.append("| Ticker | Market Cap | Price | Shares |")
        report_lines.append("|--------|-----------|-------|--------|")
        for t, mc, p, s in sorted(small_mktcap_tickers, key=lambda x: x[1]):
            report_lines.append(f"| {t} | ${mc:.2e} | ${p:.2f} | {s:.2e} |")
    else:
        report_lines.append(f"All tickers have computed market_cap >= $1B at {latest_date_str}. **OK**\n")
    report_lines.append("")

    verdict = "PASS" if issues == 0 else f"**FAIL** ({issues} issues)"
    report_lines.append(f"**Check 4 Verdict**: {verdict}\n")
    return issues


# ---------------------------------------------------------------------------
# Check 5: yfinance Cross-Check
# ---------------------------------------------------------------------------

def check5_yfinance_crosscheck(xbrl, prices, report_lines):
    report_lines.append("## Check 5: yfinance Cross-Check\n")

    try:
        import yfinance as yf
    except ImportError:
        report_lines.append("**SKIPPED** — yfinance not installed.\n")
        return 0

    issues = 0
    report_lines.append("| Ticker | Component | Our Value | yfinance | Ratio | Status |")
    report_lines.append("|--------|-----------|-----------|----------|-------|--------|")

    latest_price_date = prices.index.max().strftime("%Y-%m-%d")

    for ticker in BENCHMARK_TICKERS:
        try:
            info = yf.Ticker(ticker).info
            yf_mktcap = info.get("marketCap")
            yf_shares = info.get("sharesOutstanding")

            # Our shares
            our_shares, _, _ = get_latest_xbrl_value(
                xbrl, ticker, SHARES_TAGS, latest_price_date, "shares"
            )
            our_price = get_price_at_date(prices, ticker, latest_price_date)

            if our_shares and our_price:
                our_mktcap = our_price * our_shares

                if yf_mktcap and yf_mktcap > 0:
                    ratio = our_mktcap / yf_mktcap
                    status = "PASS" if 0.3 < ratio < 3.0 else "**FAIL**"
                    if status == "**FAIL**":
                        issues += 1
                    report_lines.append(f"| {ticker} | Market Cap | ${our_mktcap:.2e} | ${yf_mktcap:.2e} | {ratio:.2f}x | {status} |")

                if yf_shares and yf_shares > 0 and our_shares:
                    ratio = our_shares / yf_shares
                    status = "PASS" if 0.3 < ratio < 3.0 else "**FAIL**"
                    if status == "**FAIL**":
                        issues += 1
                    report_lines.append(f"| {ticker} | Shares | {our_shares:.2e} | {yf_shares:.2e} | {ratio:.2f}x | {status} |")
            else:
                report_lines.append(f"| {ticker} | — | MISSING | — | — | **FAIL** |")
                issues += 1
        except Exception as e:
            report_lines.append(f"| {ticker} | — | ERROR: {e} | — | — | SKIP |")

    report_lines.append("")

    verdict = "PASS" if issues == 0 else f"**FAIL** ({issues} issues)"
    report_lines.append(f"**Check 5 Verdict**: {verdict}\n")
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Data Legitimacy Validation")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output markdown file path")
    args = parser.parse_args()

    print("Loading data...")
    xbrl, factor, prices = load_data()
    print(f"  xbrl_df:       {xbrl.shape[0]:>10,} rows")
    print(f"  factor_panel:  {factor.shape[0]:>10,} rows")
    print(f"  prices_cache:  {prices.shape[0]:>10,} days x {prices.shape[1]} tickers")

    report = []
    report.append("# Data Legitimacy Validation Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"Data: xbrl_df ({xbrl.shape[0]:,} rows), "
                  f"factor_panel ({factor.shape[0]:,} rows), "
                  f"prices ({prices.shape[0]:,} days x {prices.shape[1]} tickers)\n")
    report.append("---\n")

    total_issues = 0

    print("\nRunning Check 1: Spot-Check Known Companies...")
    total_issues += check1_spot_check(xbrl, prices, report)

    print("Running Check 2: Distribution Analysis...")
    total_issues += check2_distribution(factor, report)

    print("Running Check 3: Unit Consistency Audit...")
    total_issues += check3_unit_consistency(xbrl, report)

    print("Running Check 4: Magnitude Sanity Check...")
    total_issues += check4_magnitude(xbrl, factor, prices, report)

    print("Running Check 5: yfinance Cross-Check...")
    total_issues += check5_yfinance_crosscheck(xbrl, prices, report)

    # Executive Summary
    summary = []
    summary.append("---\n")
    summary.append("## Executive Summary\n")
    summary.append(f"**Total Issues Found: {total_issues}**\n")
    if total_issues == 0:
        summary.append("All checks passed. Data appears legitimate.\n")
    else:
        summary.append("Data quality issues detected. See detailed findings above.\n")

    # Insert summary after header
    report = report[:4] + summary + report[4:]

    report_text = "\n".join(report)

    if args.output:
        Path(args.output).write_text(report_text)
        print(f"\nReport written to: {args.output}")
    else:
        print("\n" + "=" * 70)
        print(report_text)
        print("=" * 70)

    print(f"\nTotal issues: {total_issues}")
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
