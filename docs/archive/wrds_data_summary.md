# WRDS Data Acquisition Summary

**Project**: MA751 — Cross-Sectional Stock Return Prediction (S&P 500, 2015–2024)
**Date**: April 18, 2026
**Source**: Wharton Research Data Services (WRDS), Boston University subscription

---

## 1. Data Inventory

### Tier 0 — Core (Downloaded Previously)

| File | Rows | Description |
|------|------|-------------|
| `crsp_monthly.parquet` | 505,604 | CRSP monthly stock file — unadjusted price, returns, shares outstanding, market cap |
| `compustat_annual.parquet` | 117,570 | Compustat annual fundamentals — revenue, GP, assets, earnings, cash flow, CapEx |
| `compustat_quarterly.parquet` | 471,559 | Compustat quarterly fundamentals |
| `ccm_link.parquet` | 33,324 | CRSP-Compustat Merged link table (PERMNO ↔ GVKEY) |

### Tier 1 — Analyst & Factor Data

| File | Rows | Description |
|------|------|-------------|
| `ibes_consensus.parquet` | 1,353,848 | IBES consensus EPS estimates — mean, median, std, analyst count, up/down revisions |
| `ibes_surprises.parquet` | 642,013 | IBES earnings surprises — actual vs. consensus, standardized surprise |
| `ibes_crsp_link.parquet` | 30,080 | IBES ticker → CRSP PERMNO mapping |
| `ff5_factors_monthly.parquet` | 144 | Fama-French 5 factors + momentum (MktRF, SMB, HML, RMW, CMA, UMD, RF) |
| `ff_factors_monthly.parquet` | 144 | Fama-French 3 factors + momentum |

### Tier 2 — Alternative & Risk Data

| File | Rows | Description |
|------|------|-------------|
| `short_interest.parquet` | 3,257,969 | Compustat short interest — biweekly short positions by GVKEY |
| `inst_ownership_13f.parquet` | 766,643 | Institutional ownership from 13F filings — number of institutions & total shares per CUSIP-quarter |
| `crsp_daily.parquet` | 11,516,028 | CRSP daily stock file — returns, price, volume (for beta/IVOL computation) |
| `vix_daily.parquet` | 3,039 | CBOE VIX daily values |

### Derived Features (Computed from CRSP Daily)

| File | Rows | Description |
|------|------|-------------|
| `stock_beta_monthly.parquet` | 515,439 | 252-day rolling CAPM beta per stock-month |
| `stock_ivol_monthly.parquet` | 528,878 | 60-day idiosyncratic volatility (CAPM residual std) |
| `market_regime_monthly.parquet` | 140 | Market regime indicators — volatility regime & trend regime |

**Total: 16 files, ~19.7M rows, ~251 MB**

---

## 2. New Feature Definitions

### 2.1 Analyst Features (from IBES)

**Standardized Unexpected Earnings (SUE)**
$$\text{SUE} = \frac{\text{Actual EPS} - \text{Consensus Mean EPS}}{\text{Price}}$$
Source: `ibes_surprises.surpmean`. Literature shows strong post-earnings-announcement drift (PEAD) — stocks with positive surprises continue to outperform for 1–3 months.

**Analyst Revision**
$$\text{Revision} = \frac{\text{Mean EPS}_t - \text{Mean EPS}_{t-1}}{\text{Price}}$$
Source: `ibes_consensus.meanest` month-over-month change. Analyst upgrades/downgrades are persistent return predictors.

**Analyst Dispersion**
$$\text{Dispersion} = \frac{\text{Std of EPS Estimates}}{\lvert\text{Mean EPS Estimate}\rvert}$$
Source: `ibes_consensus.stdev / |meanest|`. High dispersion → greater uncertainty → lower future returns (Diether, Malloy, Scherbina 2002).

**Revision Breadth**
$$\text{Breadth} = \frac{\text{Upgrades} - \text{Downgrades}}{\text{Total Analysts}}$$
Source: `ibes_consensus.numup, numdown, numest`. Net direction of analyst sentiment.

### 2.2 Risk Features (from CRSP Daily)

**Market Beta**
$$\beta_i = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)} \quad \text{(252-day rolling window)}$$
Source: `stock_beta_monthly.parquet`. Mean = 1.007, median = 0.976. Captures systematic risk exposure. The low-beta anomaly (Frazzini & Pedersen 2014) suggests low-beta stocks earn higher risk-adjusted returns.

**Idiosyncratic Volatility (IVOL)**
$$\text{IVOL}_i = \text{Std}(r_i - \hat{\beta}_i \cdot r_m) \quad \text{(60-day rolling)}$$
Source: `stock_ivol_monthly.parquet`. Mean = 2.98%, median = 2.20%. The IVOL puzzle (Ang et al. 2006) shows high-IVOL stocks underperform — a strong cross-sectional predictor.

### 2.3 Market Regime Variables

**Volatility Regime**
$$\text{VolRegime}_t = \mathbf{1}\left[\sigma_{63d,t}^{\text{mkt}} > \text{Median}(\sigma_{63d}^{\text{mkt}})\right]$$
High-vol months: 96, Low-vol months: 44. Factor premiums vary significantly across regimes — momentum tends to crash in high-vol environments.

**Trend Regime**
$$\text{TrendRegime}_t = \mathbf{1}\left[\sum_{k=1}^{126} r_{m,t-k} > 0\right]$$
Bull months: 98, Bear months: 42. Value factors (HML) tend to perform better in recoveries; momentum in sustained trends.

### 2.4 Alternative Data

**Short Interest Ratio**
$$\text{ShortRatio} = \frac{\text{Short Interest}}{\text{Shares Outstanding}}$$
Source: `short_interest.parquet` + `crsp_monthly.shrout`. High short interest predicts negative future returns (Dechow et al. 2001). Biweekly frequency, lagged to avoid look-ahead.

**Institutional Ownership Change**
$$\Delta\text{InstOwn} = \frac{\text{Inst Shares}_t - \text{Inst Shares}_{t-1}}{\text{Shares Outstanding}}$$
Source: `inst_ownership_13f.parquet`. Quarterly from 13F filings. Increasing institutional ownership predicts positive returns (Gompers & Metrick 2001).

---

## 3. Integration Plan

### Current Feature Set (6 factors + 9 macro)

| Factor | Source | Known Issues |
|--------|--------|-------------|
| EarningsYield | XBRL + yfinance | Stock split contamination in market cap |
| GrossProfitability | XBRL | 55 tickers missing (XBRL concept coverage), 22.4% NaN |
| AssetGrowth | XBRL | Clean |
| Accruals | XBRL | Clean |
| Momentum12_1 | yfinance | Split-adjusted (correct for returns) |
| NetDebt/EBITDA | XBRL | Type A NaN for Financials/RE |

### Upgraded Feature Set (after WRDS integration)

**Fixes:**
- EarningsYield → rebuild from `comp.ib / crsp.mktcap` (eliminates split contamination)
- GrossProfitability → rebuild from `comp.gp / comp.at` (fills 55 missing tickers)
- Momentum → compute from `crsp.ret` (no yfinance dependency)

**New features (up to 8 additional):**
1. SUE (earnings surprise)
2. Analyst revision
3. Analyst dispersion
4. Beta
5. IVOL
6. Short interest ratio
7. Institutional ownership change
8. Regime interaction terms

This expands the feature set from 15 → 23 dimensions, adding analyst sentiment, risk characteristics, and alternative data — three categories absent from the current model.

---

## 4. Data Quality Notes

### CRSP
- `prc` can be negative (bid/ask midpoint) → use `abs(prc)`
- `shrout` is in thousands → multiply by 1,000 for actual shares
- Market cap = `abs(prc) * shrout * 1000`

### Compustat
- SIC code column is `sich`, not `sic`
- Quarterly cash flow is `oancfy` (YTD), not `oancfq`
- Quarterly CapEx is `capxy`, not `capxq`

### IBES
- IBES ticker ≠ CRSP ticker — must use `ibes_crsp_link.parquet` to map via PERMNO
- Consensus `statpers` (statistics period) is the forecast date; `fpedats` is the fiscal period end date
- `fpi = '1'` is current fiscal year, `fpi = '2'` is next fiscal year

### OptionMetrics
- BU does **not** have access to individual stock implied volatility (OptionMetrics `optionm_all` permission denied)
- Workaround: use IVOL from CRSP daily as a risk measure, VIX for market-level vol

### Identifiers
- CRSP uses PERMNO, Compustat uses GVKEY, IBES uses its own ticker
- Join path: IBES ticker → `ibes_crsp_link` → PERMNO → `ccm_link` → GVKEY
- All joins must respect date validity windows (link start/end dates)

---

## 5. Access & Reproducibility

- **WRDS credentials**: `~/.pgpass` (PostgreSQL passfile, auto-authenticated)
- **Connection**: `postgresql://evanolott@wrds-pgdata.wharton.upenn.edu:9737/wrds`
- **Download scripts**: `download_wrds_tier12.py`, `scripts/download_wrds_*.py`
- **Derived features**: `compute_derived_features.py`
- **All data**: `data/wrds/*.parquet`
