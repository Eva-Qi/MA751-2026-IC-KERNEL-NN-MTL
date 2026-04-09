"""
diagnose_tickers.py — One-screen ticker debug view + NaN reason tracker

For each ticker, shows:
- Shares: value / source / plausible?
- Price: latest / source
- Market Cap: computed / >$1B?
- NI TTM: value / missing reason
- EY: value / NaN reason

Classifies NaN into:
- shares_missing: no shares tag in XBRL
- shares_filtered: shares exist but failed plausibility guard
- price_missing: no price data
- mktcap_too_small: mktcap < $1B
- ni_missing: no net income (FY or 4Q)
- ey_too_extreme: |EY| > 1.0
- should_work: all components exist, EY should be computable
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

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


def diagnose_all(date="2022-06-30"):
    xbrl = pd.read_parquet(DATA_DIR / "xbrl_df.parquet")
    prices = pd.read_parquet(DATA_DIR / "prices_cache.parquet")
    fp = pd.read_parquet(DATA_DIR / "factor_panel.parquet")
    sector = pd.read_json(DATA_DIR / "sector_map.json", typ="series")

    tickers = sorted(fp["ticker"].unique())
    date_ts = pd.Timestamp(date)
    stale = (date_ts - pd.DateOffset(months=24)).strftime("%Y-%m-%d")

    results = []

    for t in tickers:
        row = {"ticker": t, "sector": sector.get(t, "?")}

        # --- Shares ---
        sh_all = xbrl[
            (xbrl["ticker"] == t)
            & xbrl["concept"].isin(SHARES_TAGS)
            & (xbrl["unit"].str.lower() == "shares")
            & (xbrl["available_date"] <= date)
        ]
        sh_plausible = sh_all[(sh_all["value"] >= 1e6) & (sh_all["value"] <= 1e11)]

        if len(sh_plausible) > 0:
            best = sh_plausible.sort_values("period_end", ascending=False).iloc[0]
            row["shares"] = best["value"]
            row["shares_tag"] = best["concept"]
            row["shares_status"] = "OK"
        elif len(sh_all) > 0:
            row["shares"] = sh_all.sort_values("period_end", ascending=False).iloc[0]["value"]
            row["shares_tag"] = "FILTERED"
            row["shares_status"] = "filtered"
        else:
            row["shares"] = None
            row["shares_tag"] = "NONE"
            row["shares_status"] = "missing"

        # --- Price ---
        if t in prices.columns:
            psub = prices.loc[prices.index <= date_ts, t].dropna()
            row["price"] = float(psub.iloc[-1]) if len(psub) > 0 else None
        else:
            row["price"] = None

        # --- Market Cap ---
        if row.get("shares") and row.get("price") and row["shares_status"] == "OK":
            row["mktcap"] = row["price"] * row["shares"]
            row["mktcap_ok"] = row["mktcap"] > 1e9
        else:
            row["mktcap"] = None
            row["mktcap_ok"] = False

        # --- Net Income ---
        ni_fy = xbrl[
            (xbrl["ticker"] == t) & xbrl["concept"].isin(NI_TAGS)
            & (xbrl["unit"].str.lower() == "usd")
            & (xbrl["available_date"] <= date)
            & (xbrl["fiscal_period"] == "FY")
            & (xbrl["period_end"] >= stale)
        ]
        ni_q = xbrl[
            (xbrl["ticker"] == t) & xbrl["concept"].isin(NI_TAGS)
            & (xbrl["unit"].str.lower() == "usd")
            & (xbrl["available_date"] <= date)
            & xbrl["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])
        ].drop_duplicates(subset=["fiscal_year", "fiscal_period"])

        if len(ni_fy) > 0:
            row["ni"] = float(ni_fy.sort_values("period_end", ascending=False).iloc[0]["value"])
            row["ni_source"] = "FY"
        elif len(ni_q) >= 4:
            row["ni"] = float(ni_q.sort_values("period_end", ascending=False).head(4)["value"].sum())
            row["ni_source"] = "4Q"
        else:
            row["ni"] = None
            row["ni_source"] = f"Q={len(ni_q)}"

        # --- EY + NaN reason ---
        if row["shares_status"] == "missing":
            row["nan_reason"] = "shares_missing"
        elif row["shares_status"] == "filtered":
            row["nan_reason"] = "shares_filtered"
        elif row["price"] is None:
            row["nan_reason"] = "price_missing"
        elif not row["mktcap_ok"]:
            row["nan_reason"] = "mktcap_too_small"
        elif row["ni"] is None:
            row["nan_reason"] = "ni_missing"
        else:
            ey = row["ni"] / row["mktcap"]
            if abs(ey) > 1.0:
                row["nan_reason"] = f"ey_extreme({ey:.2f})"
            else:
                row["nan_reason"] = None
                row["ey"] = ey

        # Check actual factor_panel value
        fp_row = fp[(fp["ticker"] == t) & (fp["signal_date"] == date_ts)]
        row["ey_actual"] = float(fp_row["EarningsYield"].iloc[0]) if len(fp_row) > 0 and pd.notna(fp_row["EarningsYield"].iloc[0]) else None

        results.append(row)

    df = pd.DataFrame(results)

    # Summary
    print(f"\n{'='*70}")
    print(f"TICKER DIAGNOSIS @ {date}")
    print(f"{'='*70}")

    nan_reasons = df[df["nan_reason"].notna()]["nan_reason"].apply(
        lambda x: x.split("(")[0]  # group ey_extreme variants
    ).value_counts()
    print(f"\nNaN REASONS ({nan_reasons.sum()} tickers with issues):")
    for reason, count in nan_reasons.items():
        print(f"  {reason:<25} {count:>4}")

    ok = df[df["nan_reason"].isna()]
    print(f"\nOK (should have EY): {len(ok)}")

    # Show problem tickers
    problems = df[df["nan_reason"].notna()].sort_values("nan_reason")
    print(f"\n{'Ticker':<7} {'Sector':<20} {'Shares':>12} {'Price':>8} {'MktCap':>12} {'NI':>12} {'Reason':<25}")
    print("-" * 100)
    for _, r in problems.iterrows():
        sh = f"{r['shares']:.2e}" if r["shares"] else "NONE"
        p = f"${r['price']:.2f}" if r["price"] else "NONE"
        mc = f"${r['mktcap']:.2e}" if r["mktcap"] else "NONE"
        ni = f"{r['ni']:.2e}" if r["ni"] else "NONE"
        print(f"{r['ticker']:<7} {r['sector']:<20} {sh:>12} {p:>8} {mc:>12} {ni:>12} {r['nan_reason']:<25}")

    return df


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2022-06-30"
    diagnose_all(date)
