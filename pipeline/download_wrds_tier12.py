"""
Download Tier 1 & Tier 2 WRDS data for MA751 cross-sectional return prediction.

Tier 1: IBES (analyst estimates), Fama-French 5 factors
Tier 2: Short Interest, Institutional Ownership (13F), CRSP Daily (for beta/IVOL)

Auth via ~/.pgpass or WRDS_USER/WRDS_PASS env vars
"""

import os
import pandas as pd
import sqlalchemy
from pathlib import Path

OUT = Path("data/wrds")
OUT.mkdir(parents=True, exist_ok=True)

engine = sqlalchemy.create_engine(
    f"postgresql://{os.environ.get('WRDS_USER', 'YOUR_USER')}:{os.environ.get('WRDS_PASS', 'YOUR_PASS')}@wrds-pgdata.wharton.upenn.edu:9737/wrds"
)

def query(sql):
    return pd.read_sql_query(sql, engine)

# ──────────────────────────────────────────────
# 0. Check which libraries BU has access to
# ──────────────────────────────────────────────
print("=== Checking library access ===")
libs_df = query("SELECT nspname FROM pg_namespace ORDER BY nspname")
libs = libs_df["nspname"].tolist()
for keyword in ["ibes", "tfn", "ff", "comp", "crsp", "optionm", "short"]:
    matches = [l for l in libs if keyword in l.lower()]
    if matches:
        print(f"  {keyword}: {matches[:8]}")

# ──────────────────────────────────────────────
# 1. IBES — Analyst Estimates (Tier 1)
# ──────────────────────────────────────────────

# 1a. IBES Summary Statistics — consensus EPS estimates
print("\n=== IBES: Consensus EPS Estimates ===")
try:
    ibes_stats = query("""
        SELECT ticker, statpers, fpedats, measure, fpi,
               meanest, medest, stdev, numest, actual
        FROM ibes.statsumu_epsus
        WHERE statpers >= '2013-01-01'
          AND statpers <= '2024-12-31'
          AND measure = 'EPS'
          AND fpi IN ('1', '2')
    """)
    print(f"  IBES consensus: {len(ibes_stats):,} rows")
    ibes_stats.to_parquet(OUT / "ibes_consensus.parquet", index=False)
    print(f"  Saved -> {OUT / 'ibes_consensus.parquet'}")
except Exception as e:
    print(f"  IBES consensus FAILED: {e}")
    ibes_stats = None

# 1b. IBES Surprises — actual vs estimate
print("\n=== IBES: Earnings Surprises ===")
try:
    ibes_surp = query("""
        SELECT ticker, anndats, actual, surpmean, surpstdev
        FROM ibes.surpsum
        WHERE anndats >= '2013-01-01'
          AND anndats <= '2024-12-31'
          AND measure = 'EPS'
    """)
    print(f"  IBES surprises: {len(ibes_surp):,} rows")
    ibes_surp.to_parquet(OUT / "ibes_surprises.parquet", index=False)
    print(f"  Saved -> {OUT / 'ibes_surprises.parquet'}")
except Exception as e:
    print(f"  IBES surprises FAILED: {e}")
    ibes_surp = None

# 1c. IBES ticker-to-CRSP PERMNO mapping
print("\n=== IBES-CRSP ID Mapping ===")
try:
    ibes_id = query("""
        SELECT ticker AS ibes_ticker, permno, sdate, edate
        FROM wrdsapps.ibcrsphist
        WHERE score <= 2
    """)
    print(f"  IBES-CRSP map: {len(ibes_id):,} rows")
    ibes_id.to_parquet(OUT / "ibes_crsp_link.parquet", index=False)
    print(f"  Saved -> {OUT / 'ibes_crsp_link.parquet'}")
except Exception as e:
    print(f"  IBES-CRSP map FAILED: {e}")

# ──────────────────────────────────────────────
# 2. Fama-French 5 Factors (Tier 1)
# ──────────────────────────────────────────────
print("\n=== Fama-French 5 Factors + Momentum ===")
try:
    ff = query("""
        SELECT date, mktrf, smb, hml, rf, umd
        FROM ff.factors_monthly
        WHERE date >= '2013-01-01'
          AND date <= '2024-12-31'
    """)
    print(f"  FF factors: {len(ff):,} rows")
    ff.to_parquet(OUT / "ff_factors_monthly.parquet", index=False)
    print(f"  Saved -> {OUT / 'ff_factors_monthly.parquet'}")
except Exception as e:
    print(f"  FF monthly FAILED: {e}")

# Try 5-factor model (RMW + CMA)
try:
    ff5 = query("""
        SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
        FROM ff.fivefactors_monthly
        WHERE date >= '2013-01-01'
          AND date <= '2024-12-31'
    """)
    print(f"  FF 5-factor: {len(ff5):,} rows")
    ff5.to_parquet(OUT / "ff5_factors_monthly.parquet", index=False)
    print(f"  Saved -> {OUT / 'ff5_factors_monthly.parquet'}")
except Exception as e:
    print(f"  FF 5-factor FAILED: {e}")

# ──────────────────────────────────────────────
# 3. Short Interest (Tier 2)
# ──────────────────────────────────────────────
print("\n=== Short Interest ===")
# Try Compustat supplemental short interest
try:
    short = query("""
        SELECT *
        FROM comp.sec_shortint
        WHERE datadate >= '2013-01-01'
          AND datadate <= '2024-12-31'
        LIMIT 5
    """)
    print(f"  comp.sec_shortint sample: {len(short)} rows, cols={list(short.columns)}")
    # If it works, get full data
    short_full = query("""
        SELECT gvkey, datadate, shortint, shortintadj, splitadjdate
        FROM comp.sec_shortint
        WHERE datadate >= '2013-01-01'
          AND datadate <= '2024-12-31'
    """)
    print(f"  Short interest: {len(short_full):,} rows")
    short_full.to_parquet(OUT / "short_interest.parquet", index=False)
    print(f"  Saved -> {OUT / 'short_interest.parquet'}")
except Exception as e:
    print(f"  Short interest FAILED: {e}")

# ──────────────────────────────────────────────
# 4. Institutional Ownership — 13F (Tier 2)
# ──────────────────────────────────────────────
print("\n=== Institutional Ownership (13F) ===")
# Try Thomson Reuters S34 mutual fund holdings
try:
    inst_sample = query("""
        SELECT *
        FROM tfn.s34type1
        LIMIT 5
    """)
    print(f"  tfn.s34type1 sample cols: {list(inst_sample.columns)}")
except Exception as e:
    print(f"  tfn.s34type1 FAILED: {e}")

# Try wrdsapps institutional ownership
try:
    inst = query("""
        SELECT *
        FROM wrdsapps.own_inst_conc_v2
        LIMIT 5
    """)
    print(f"  own_inst_conc_v2 sample cols: {list(inst.columns)}")
except Exception as e:
    print(f"  own_inst_conc_v2 FAILED: {e}")

# Try the 13f holdings summary
try:
    inst13f = query("""
        SELECT rdate, mgrno, fdate, cusip, shares, prc
        FROM tfn.s34
        WHERE rdate >= '2013-01-01'
          AND rdate <= '2024-12-31'
        LIMIT 100
    """)
    print(f"  tfn.s34 sample: {len(inst13f)} rows, cols={list(inst13f.columns)}")
except Exception as e:
    print(f"  tfn.s34 FAILED: {e}")

# Try the stock-level institutional ownership summary
try:
    inst_own = query("""
        SELECT permno, rdate,
               instown_perc, instown_hhi, num_inst,
               top5_perc, top10_perc
        FROM wrdsapps_fintools.v_inst_ownership
        WHERE rdate >= '2013-01-01'
          AND rdate <= '2024-12-31'
    """)
    print(f"  Inst ownership summary: {len(inst_own):,} rows")
    inst_own.to_parquet(OUT / "inst_ownership.parquet", index=False)
    print(f"  Saved -> {OUT / 'inst_ownership.parquet'}")
except Exception as e:
    print(f"  Inst ownership summary FAILED: {e}")

# ──────────────────────────────────────────────
# 5. CRSP Daily Returns (Tier 2 — for beta/IVOL)
# ──────────────────────────────────────────────
print("\n=== CRSP Daily Returns ===")
try:
    crsp_daily = query("""
        SELECT a.permno, a.date, a.ret, a.prc, a.vol, a.shrout
        FROM crsp.dsf a
        INNER JOIN crsp.msenames b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt
        WHERE a.date >= '2013-01-01'
          AND a.date <= '2024-12-31'
          AND b.shrcd IN (10, 11)
          AND b.exchcd IN (1, 2, 3)
    """)
    print(f"  CRSP daily: {len(crsp_daily):,} rows")
    crsp_daily.to_parquet(OUT / "crsp_daily.parquet", index=False)
    print(f"  Saved -> {OUT / 'crsp_daily.parquet'}")
except Exception as e:
    print(f"  CRSP daily FAILED: {e}")

# ──────────────────────────────────────────────
# 6. Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
for f in sorted(OUT.glob("*.parquet")):
    try:
        df = pd.read_parquet(f)
        print(f"  {f.name:<35} {len(df):>10,} rows  {len(df.columns):>3} cols")
    except:
        print(f"  {f.name:<35}  ERROR reading")

engine.dispose()
print("\nDone.")
