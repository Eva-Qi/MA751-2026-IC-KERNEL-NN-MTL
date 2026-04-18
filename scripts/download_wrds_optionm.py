"""
Download OptionMetrics data from WRDS:
1. secid link table (securd has cusip/permno mapping)
2. Volatility surface data from vsurfd{year} tables (2013-2024)
"""

import pandas as pd
import sqlalchemy
import time
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wrds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_engine():
    return sqlalchemy.create_engine(
        "postgresql://WRDS_USER:WRDS_PASS@wrds-pgdata.wharton.upenn.edu:9737/wrds"
    )

def run_query(engine, sql, description):
    print(f"\n{'='*60}")
    print(f"[START] {description}")
    t0 = time.time()
    try:
        df = pd.read_sql_query(sql, engine)
        elapsed = time.time() - t0
        print(f"[DONE] {df.shape[0]:,} rows x {df.shape[1]} cols in {elapsed:.1f}s")
        return df
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[ERROR] after {elapsed:.1f}s: {e}")
        return None

def main():
    engine = get_engine()
    print("Connection OK")

    # =========================================================================
    # 1. secid link table — try securd (has cusip)
    # =========================================================================
    outfile_link = os.path.join(OUTPUT_DIR, "optionm_secid_link.parquet")
    if not os.path.exists(outfile_link):
        # Check securd columns first
        cols_sql = """
        SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema = 'optionm' AND table_name = 'securd'
        ORDER BY ordinal_position
        """
        cols = pd.read_sql_query(cols_sql, engine)
        print("Columns in optionm.securd:")
        for _, r in cols.iterrows():
            print(f"  {r['column_name']}: {r['data_type']}")

        # Try to read securd
        sql_link = "SELECT * FROM optionm.securd"
        df_link = run_query(engine, sql_link, "optionm.securd (security link)")
        if df_link is not None:
            df_link.to_parquet(outfile_link, index=False)
            print(f"  Saved to {outfile_link}")
        else:
            # Fallback: try securd1
            print("Trying securd1...")
            cols_sql2 = """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'optionm' AND table_name = 'securd1'
            ORDER BY ordinal_position
            """
            cols2 = pd.read_sql_query(cols_sql2, engine)
            print(f"securd1 columns: {list(cols2['column_name'])}")

            sql_link2 = "SELECT * FROM optionm.securd1"
            df_link = run_query(engine, sql_link2, "optionm.securd1 (security link)")
            if df_link is not None:
                df_link.to_parquet(outfile_link, index=False)
                print(f"  Saved to {outfile_link}")
    else:
        print(f"[SKIP] {outfile_link} already exists")

    # =========================================================================
    # 2. Volatility surface — yearly tables vsurfd2013 to vsurfd2024
    # =========================================================================
    outfile_iv = os.path.join(OUTPUT_DIR, "optionm_ivsurface.parquet")
    if not os.path.exists(outfile_iv):
        all_dfs = []
        for year in range(2013, 2025):
            table = f"vsurfd{year}"
            sql = f"""
            SELECT secid, date, days, delta, impl_volatility, impl_strike, cp_flag
            FROM optionm.{table}
            WHERE days IN (30, 60, 91, 182, 365)
              AND delta IN (20, 25, 30, 40, 50, 60, 70, 75, 80)
            """
            df = run_query(engine, sql, f"optionm.{table}")
            if df is not None:
                all_dfs.append(df)
                print(f"  Running total: {sum(len(d) for d in all_dfs):,} rows")

        if all_dfs:
            df_iv = pd.concat(all_dfs, ignore_index=True)
            print(f"\nTotal IV surface: {df_iv.shape[0]:,} rows x {df_iv.shape[1]} cols")
            df_iv.to_parquet(outfile_iv, index=False)
            print(f"  Saved to {outfile_iv}")
    else:
        print(f"[SKIP] {outfile_iv} already exists")

    # =========================================================================
    # 3. Final validation
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL VALIDATION")
    print("="*70)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".parquet"):
            path = os.path.join(OUTPUT_DIR, f)
            df = pd.read_parquet(path)
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"  {f:40s} {df.shape[0]:>12,} rows  {df.shape[1]:>3} cols  {size_mb:>7.1f} MB")

if __name__ == "__main__":
    main()
