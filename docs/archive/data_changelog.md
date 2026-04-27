# MA751 Data Pipeline — Changelog & Results

**Date**: April 18, 2026  
**Project**: Cross-Sectional Stock Return Prediction (S&P 500, 2015–2024)

---

## 1. Data Source Migration

| Component | Before (V1) | After (V2) | Why |
|-----------|------------|------------|-----|
| Prices / Market Cap | yfinance (split-adjusted) | CRSP monthly (`prc`, `shrout`) | Eliminates stock split contamination in EY |
| Fundamentals | SEC XBRL filings (parsed) | Compustat annual (`gp`, `at`, `ib`, etc.) | GP/A NaN: 22.4% → 0%. Covers all tickers |
| Forward Returns | yfinance price changes | CRSP monthly `ret` | No split/dividend ambiguity |
| Realized Vol | yfinance 21-day rolling std | CRSP daily 21-day rolling std | Direct from 11.5M daily return rows |
| Analyst Data | *none* | IBES (consensus + surprises) | SUE, revision, dispersion, breadth |
| Risk Factors | *none* | CRSP daily (computed) | 252-day beta, 60-day IVOL |
| Short Interest | *none* | Compustat short interest | Biweekly short positions |
| Institutional | *none* | Thomson Reuters 13F | Quarterly ownership changes |
| Fama-French | *none* | WRDS FF library | 5 factors + momentum |
| CCM Link | *none* | CRSP-Compustat merged | PERMNO ↔ GVKEY mapping |

**Total WRDS data: 16 parquet files, 19.7M rows, 251 MB**

---

## 2. Pipeline Fixes (Audit-Driven)

| Fix | Bug | Impact | Resolution |
|-----|-----|--------|------------|
| **Z-score universe** | Z-scored on 6,900 CRSP stocks, filtered to 442 S&P 500 after | EY mean=+0.37 (should be 0), LASSO all-zero | Z-score AFTER filtering to S&P 500 |
| **Zero-variance features** | 9 macro + 2 regime features had std=0 within each month | 44% of features = dead weight in cross-section | Removed from model features; kept as metadata |
| **Realized vol proxy** | Used IVOL (CAPM residual) as realized vol target | Wrong concept: IVOL ≠ total vol; 36 months 0% coverage | Computed true rv from CRSP daily returns |
| **NaN semantics** | SUE 72.9% NaN → fillna(0) = "no surprise" | Missing coverage ≠ zero surprise | Added `has_sue` binary indicator |
| **Type A expansion** | Accruals not flagged for Financials | Wrong imputation for banks | Added Financials to Accruals Type A |
| **Pandas nullable dtype** | WRDS parquet uses `Float64` → sklearn crash | `NAType` can't convert to float | Convert all columns to `float64` before save |
| **Date alignment** | Daily rv snapped to calendar month-end (Jan 31) vs panel (Jan 30) | 36 months of rv = NaN after join | Snap to last trading day per month |

---

## 3. Feature Set Change

| V1 (15 features) | V2 (14 features) |
|-------------------|-------------------|
| 6 factor z-scores | 6 factor z-scores (same names, WRDS source) |
| 9 macro variables | *removed* (zero CS variance) |
| | + SUE_zscore |
| | + AnalystRevision_zscore |
| | + AnalystDispersion_zscore |
| | + RevisionBreadth_zscore |
| | + Beta_zscore |
| | + IVOL_zscore |
| | + ShortInterestRatio_zscore |
| | + InstOwnershipChg_zscore |

Macro/regime kept as panel columns for conditioning, not model input.

---

## 4. Results — 5-Rung Complexity Ladder

All models: 14 stock-level features, 58 test months (2020-02 to 2024-11), S&P 500 universe.

| Rung | Model | IC | Sharpe | Hit% |
|------|-------|----|--------|------|
| 1a | OLS | -0.012 | 0.055 | 50.0% |
| 1b | IC-weighted Ensemble | -0.002 | 0.271 | 48.3% |
| 2 | LASSO | -0.009 | 0.143 | 50.0% |
| 3 | GAM (8 splines) | -0.014 | -0.062 | 46.6% |
| 5a | MLP single-task | -0.007 | 0.037 | 43.1% |
| **5b** | **MTL (ret + ret3m)** | **-0.000** | **0.417** | **43.1%** |
| 5c | MTL (ret + vol) | -0.015 | -0.448 | 41.4% |
| 5d | MTL (ret + ret3m + vol) | -0.020 | -0.359 | 34.5% |

**Best model: 5b (MTL with 3-month return auxiliary) — Sharpe 0.417**

Key observations:
- All ICs near zero — consistent with literature for S&P 500 cross-section
- Nonlinearity (GAM) does not help over linear models
- Multi-task learning with ret3m auxiliary improves Sharpe from 0.037 → 0.417
- Vol auxiliary hurts return prediction despite good vol forecasting (corr=0.44)
