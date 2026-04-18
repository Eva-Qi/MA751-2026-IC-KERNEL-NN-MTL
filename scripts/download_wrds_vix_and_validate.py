"""
Download VIX data from CBOE and run final validation.
OptionMetrics individual stock IV is NOT accessible (BU lacks optionm_all permission).
VIX serves as market-level volatility indicator.
"""

import pandas as pd
import sqlalchemy
import time
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wrds")

def get_engine():
    return sqlalchemy.create_engine(
        "postgresql://WRDS_USER:WRDS_PASS@wrds-pgdata.wharton.upenn.edu:9737/wrds"
    )

def main():
    engine = get_engine()
    print("Connection OK\n")

    # =========================================================================
    # 1. Download VIX daily data
    # =========================================================================
    outfile_vix = os.path.join(OUTPUT_DIR, "vix_daily.parquet")
    if not os.path.exists(outfile_vix):
        print("[DOWNLOADING] VIX daily data from cboe_all.cboe...")
        t0 = time.time()
        sql = """
        SELECT date, vixo, vixh, vixl, vix
        FROM cboe_all.cboe
        WHERE date >= '2013-01-01' AND date <= '2024-12-31'
          AND vix IS NOT NULL
        ORDER BY date
        """
        df_vix = pd.read_sql_query(sql, engine)
        elapsed = time.time() - t0
        print(f"[DONE] VIX: {df_vix.shape[0]:,} rows in {elapsed:.1f}s")
        df_vix.to_parquet(outfile_vix, index=False)
        print(f"  Saved to {outfile_vix}")
    else:
        print(f"[SKIP] {outfile_vix} already exists")

    # =========================================================================
    # 2. Final validation of all parquet files
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPLETE DATA INVENTORY")
    print("=" * 80)
    print(f"{'File':<42s} {'Rows':>12s} {'Cols':>5s} {'Size':>8s} {'Date Range'}")
    print("-" * 100)

    total_size = 0
    total_rows = 0
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if not f.endswith(".parquet"):
            continue
        path = os.path.join(OUTPUT_DIR, f)
        try:
            df = pd.read_parquet(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            total_size += size_mb
            total_rows += df.shape[0]

            # Find date-like columns
            date_cols = [c for c in df.columns if any(d in c.lower() for d in
                         ["date", "pers", "fpedats", "rdate", "fdate", "datadate"])]
            date_range = ""
            if date_cols:
                col = date_cols[0]
                try:
                    vals = pd.to_datetime(df[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        date_range = f"{vals.min().strftime('%Y-%m-%d')} ~ {vals.max().strftime('%Y-%m-%d')}"
                except:
                    pass

            print(f"{f:<42s} {df.shape[0]:>12,} {df.shape[1]:>5d} {size_mb:>7.1f}M {date_range}")
        except Exception as e:
            print(f"{f:<42s} {'ERROR':>12s} {'-':>5s} {'-':>8s} {str(e)[:40]}")

    print("-" * 100)
    print(f"{'TOTAL':<42s} {total_rows:>12,} {'':>5s} {total_size:>7.1f}M")

    # =========================================================================
    # 3. Data quality checks
    # =========================================================================
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)

    checks = {
        "ibes_consensus.parquet": {"expect_cols": ["ticker", "statpers", "meanest", "medest", "numest"],
                                    "min_rows": 1_000_000},
        "inst_ownership_13f.parquet": {"expect_cols": ["cusip", "rdate", "num_inst", "total_inst_shares"],
                                        "min_rows": 500_000},
        "crsp_monthly.parquet": {"expect_cols": ["permno"],
                                  "min_rows": 400_000},
        "compustat_quarterly.parquet": {"expect_cols": ["gvkey"],
                                         "min_rows": 300_000},
        "vix_daily.parquet": {"expect_cols": ["date", "vix"],
                               "min_rows": 2500},
    }

    for fname, spec in checks.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  [MISSING] {fname}")
            continue
        df = pd.read_parquet(path)
        issues = []
        for col in spec["expect_cols"]:
            if col not in df.columns:
                issues.append(f"missing column '{col}'")
        if df.shape[0] < spec["min_rows"]:
            issues.append(f"only {df.shape[0]:,} rows (expected >= {spec['min_rows']:,})")
        if issues:
            print(f"  [WARN] {fname}: {'; '.join(issues)}")
        else:
            print(f"  [OK]   {fname}: {df.shape[0]:,} rows, all expected columns present")

    # =========================================================================
    # 4. OptionMetrics access note
    # =========================================================================
    print("\n" + "=" * 80)
    print("NOTE: OptionMetrics individual stock IV data NOT available")
    print("=" * 80)
    print("  BU subscription does NOT include optionm_all or cboe_eod schemas.")
    print("  Individual stock implied volatility must be computed from:")
    print("  - CRSP daily returns (realized vol)")
    print("  - Or obtained from alternative sources (yfinance options, etc.)")
    print("  VIX market-level data has been saved as a substitute market volatility indicator.")

if __name__ == "__main__":
    main()
