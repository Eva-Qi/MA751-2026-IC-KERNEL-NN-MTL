"""
Download remaining WRDS data for MA751 project:
1. Institutional Ownership (13F) from tfn.s34
2. OptionMetrics IV Surface from optionm.vsurfd (or fallback)
3. OptionMetrics secid-to-permno link
"""

import pandas as pd
import sqlalchemy
import time
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wrds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_engine():
    return sqlalchemy.create_engine(
        "postgresql://WRDS_USER:WRDS_PASS@wrds-pgdata.wharton.upenn.edu:9737/wrds"
    )

def run_query(engine, sql, description, timeout_sec=600):
    """Run a SQL query with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"[START] {description}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        df = pd.read_sql_query(sql, engine)
        elapsed = time.time() - t0
        print(f"[DONE] {description}: {df.shape[0]:,} rows x {df.shape[1]} cols in {elapsed:.1f}s")
        return df
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[ERROR] {description} after {elapsed:.1f}s: {e}")
        return None

def check_columns(engine, schema, table):
    """Check column names for a table."""
    sql = f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = '{schema}' AND table_name = '{table}'
    ORDER BY ordinal_position
    """
    df = pd.read_sql_query(sql, engine)
    print(f"\nColumns in {schema}.{table}:")
    for _, row in df.iterrows():
        print(f"  {row['column_name']}: {row['data_type']}")
    return df

def check_tables(engine, schema):
    """List tables in a schema."""
    sql = f"""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = '{schema}'
    ORDER BY table_name
    """
    df = pd.read_sql_query(sql, engine)
    print(f"\nTables in {schema}:")
    for _, row in df.iterrows():
        print(f"  {row['table_name']}")
    return df

def main():
    engine = get_engine()

    # Test connection
    print("Testing connection...")
    test = pd.read_sql_query("SELECT 1 as test", engine)
    print(f"Connection OK: {test.iloc[0,0]}")

    # =========================================================================
    # 1. Institutional Ownership (13F)
    # =========================================================================
    outfile_13f = os.path.join(OUTPUT_DIR, "inst_ownership_13f.parquet")
    if os.path.exists(outfile_13f):
        print(f"\n[SKIP] {outfile_13f} already exists")
        df_13f = pd.read_parquet(outfile_13f)
        print(f"  Shape: {df_13f.shape}")
    else:
        # First check what columns exist in tfn.s34
        print("\n--- Checking tfn.s34 schema ---")
        cols_s34 = check_columns(engine, "tfn", "s34")

        sql_13f = """
        SELECT s.cusip, s.rdate, s.fdate,
               COUNT(DISTINCT s.mgrno) AS num_inst,
               SUM(s.shares) AS total_inst_shares
        FROM tfn.s34 s
        WHERE s.rdate >= '2013-01-01'
          AND s.rdate <= '2024-12-31'
        GROUP BY s.cusip, s.rdate, s.fdate
        """
        df_13f = run_query(engine, sql_13f, "Institutional Ownership (13F)")
        if df_13f is not None:
            df_13f.to_parquet(outfile_13f, index=False)
            print(f"  Saved to {outfile_13f}")

    # =========================================================================
    # 2. OptionMetrics — check available tables
    # =========================================================================
    print("\n--- Checking OptionMetrics tables ---")
    om_tables = check_tables(engine, "optionm")
    table_list = om_tables["table_name"].tolist() if om_tables is not None else []

    # =========================================================================
    # 2a. OptionMetrics — secid-to-permno link
    # =========================================================================
    outfile_link = os.path.join(OUTPUT_DIR, "optionm_secid_link.parquet")
    if os.path.exists(outfile_link):
        print(f"\n[SKIP] {outfile_link} already exists")
        df_link = pd.read_parquet(outfile_link)
        print(f"  Shape: {df_link.shape}")
    else:
        # Check for linking table
        link_candidates = [t for t in table_list if "sec" in t.lower() or "link" in t.lower()]
        print(f"\nPossible linking tables: {link_candidates}")

        # Try secnmd first, then others
        link_table = None
        for candidate in ["secnmd", "securd", "securityd"]:
            if candidate in table_list:
                link_table = candidate
                break

        if link_table is None and link_candidates:
            link_table = link_candidates[0]

        if link_table:
            print(f"\nUsing link table: optionm.{link_table}")
            check_columns(engine, "optionm", link_table)

            # Try to get secid-permno mapping
            # First try with permno column
            try:
                sql_link = f"SELECT * FROM optionm.{link_table} LIMIT 5"
                sample = pd.read_sql_query(sql_link, engine)
                print(f"\nSample from optionm.{link_table}:")
                print(sample)
                print(f"Columns: {list(sample.columns)}")

                # Now fetch the full link table
                sql_link_full = f"SELECT * FROM optionm.{link_table}"
                df_link = run_query(engine, sql_link_full, f"OptionMetrics Link ({link_table})")
                if df_link is not None:
                    df_link.to_parquet(outfile_link, index=False)
                    print(f"  Saved to {outfile_link}")
            except Exception as e:
                print(f"Error with {link_table}: {e}")
        else:
            print("No linking table found in optionm schema")

    # =========================================================================
    # 2b. OptionMetrics — IV Surface
    # =========================================================================
    outfile_iv = os.path.join(OUTPUT_DIR, "optionm_ivsurface.parquet")
    if os.path.exists(outfile_iv):
        print(f"\n[SKIP] {outfile_iv} already exists")
        df_iv = pd.read_parquet(outfile_iv)
        print(f"  Shape: {df_iv.shape}")
    else:
        # Check if vsurfd exists
        if "vsurfd" in table_list:
            print("\n--- vsurfd found, checking columns ---")
            check_columns(engine, "optionm", "vsurfd")

            # Check row count first
            count_sql = """
            SELECT COUNT(*) as cnt FROM optionm.vsurfd
            WHERE date >= '2013-01-01' AND date <= '2024-12-31'
              AND days IN (30, 60, 91, 182, 365)
              AND delta IN (20, 25, 30, 40, 50, 60, 70, 75, 80)
            """
            count_df = run_query(engine, count_sql, "Count vsurfd rows")
            if count_df is not None:
                row_count = count_df.iloc[0, 0]
                print(f"  Expected rows: {row_count:,}")

            sql_iv = """
            SELECT secid, date, days, delta, impl_volatility, impl_strike, cp_flag
            FROM optionm.vsurfd
            WHERE date >= '2013-01-01' AND date <= '2024-12-31'
              AND days IN (30, 60, 91, 182, 365)
              AND delta IN (20, 25, 30, 40, 50, 60, 70, 75, 80)
            """
            df_iv = run_query(engine, sql_iv, "OptionMetrics IV Surface (vsurfd)")
            if df_iv is not None:
                df_iv.to_parquet(outfile_iv, index=False)
                print(f"  Saved to {outfile_iv}")

        elif "vsurfd1996" in table_list or any("vsurf" in t for t in table_list):
            vsurf_tables = [t for t in table_list if "vsurf" in t]
            print(f"\nFound vsurf-related tables: {vsurf_tables}")
            # Try the first one
            for vt in vsurf_tables:
                print(f"\nChecking optionm.{vt}:")
                check_columns(engine, "optionm", vt)

        else:
            print("\nvsurfd not found. Checking optionm_all schema...")
            try:
                om_all_tables = check_tables(engine, "optionm_all")
            except:
                print("optionm_all schema not accessible")

            # Fallback: try to get IV from option prices
            if "opprcd" in table_list:
                print("\n--- Trying opprcd for IV data ---")
                check_columns(engine, "optionm", "opprcd")

                # Get a small sample first
                sample_sql = "SELECT * FROM optionm.opprcd LIMIT 5"
                try:
                    sample = pd.read_sql_query(sample_sql, engine)
                    print(f"\nSample from opprcd:")
                    print(sample)
                except Exception as e:
                    print(f"Error: {e}")

    # =========================================================================
    # 3. Validation Summary
    # =========================================================================
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    parquet_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet")]
    parquet_files.sort()

    rows = []
    for f in parquet_files:
        path = os.path.join(OUTPUT_DIR, f)
        try:
            df = pd.read_parquet(path)
            # Find date-like columns
            date_cols = [c for c in df.columns if any(d in c.lower() for d in ["date", "pers", "fpedats", "rdate", "fdate", "datadate"])]
            date_range = ""
            if date_cols:
                col = date_cols[0]
                try:
                    vals = pd.to_datetime(df[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        date_range = f"{vals.min().strftime('%Y-%m-%d')} to {vals.max().strftime('%Y-%m-%d')}"
                except:
                    pass

            size_mb = os.path.getsize(path) / (1024*1024)
            rows.append({
                "file": f,
                "rows": f"{df.shape[0]:,}",
                "cols": df.shape[1],
                "size_mb": f"{size_mb:.1f}",
                "date_range": date_range
            })
        except Exception as e:
            rows.append({"file": f, "rows": "ERROR", "cols": "-", "size_mb": "-", "date_range": str(e)})

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
