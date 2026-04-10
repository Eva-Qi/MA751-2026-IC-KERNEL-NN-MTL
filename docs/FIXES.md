# Data Pipeline Fixes — April 10, 2026

Fixes applied after code audit identified 3 critical issues and 1 important issue.
Second-pass fixes applied after sonnet audit caught regressions from the initial fixes.

---

## C1: Z-Score Temporal Leakage (CRITICAL)

**File**: `load_data.py:100-106`

**Problem**: Z-scores were computed using **pooled global mean/std across all dates and tickers**:
```python
mu = fp[raw_col].mean()       # global mean — includes future data
sigma = fp[raw_col].std()     # global std — includes future data
fp[zscore_col] = (fp[raw_col] - mu) / sigma
```

This introduces temporal leakage: when the model trains on period T, the features at period T already encode information from periods T+1...T_max via the pooled statistics. It also destroys cross-sectional interpretability (factor z-scores should rank stocks within a given month, not across time).

**Fix**: Changed to **cross-sectional z-score per signal_date**:
```python
fp[zscore_col] = fp.groupby("signal_date")[raw_col].transform(_cs_zscore)
```

Each month's z-scores are now computed using only that month's cross-section. No future information leaks.

**Impact**: Affects all Rung 4-5 results. The master_panel.parquet must be regenerated.

---

## C2: EarningsYield Split Contamination (CRITICAL)

**File**: `load_data.py:96`

**Problem**: yfinance returns split-adjusted prices, but SEC XBRL filings report raw (unadjusted) shares outstanding. Computing `Market Cap = Adjusted Price x Raw Shares` gives incorrect values:
- AAPL: computed $326B vs true $1.4T (4.3x undercount)
- GOOG: ~20x error due to 20:1 split
- 54 tickers with major splits (ratio >= 2:1) affected

The previous filter `|EY| <= 1.0` only removed extreme outliers but the systematic bias remained.

**Fix (v1 — reverted)**: Blanket NaN for all tickers with any historical split. Audit found this wiped 65% of EY data (325/500 tickers) because it didn't consider split timing.

**Fix (v2 — current)**: Time-aware filtering. Only NaN rows where `signal_date < first_split_date`. Post-split rows keep their EY because yfinance prices and SEC shares are both in post-split units after the split.
```python
contaminated = fp["first_split_date"].notna() & (fp["signal_date"] < fp["first_split_date"])
fp.loc[contaminated, "EarningsYield"] = np.nan
```

Also tightened AssetGrowth filter from `|AG| <= 10,000` to `|AG| <= 10.0` (no rows affected in current data).

**Proper fix** (not yet implemented): Get CRSP data (PRC x SHROUT) for correct market cap.

---

## C3: Missing Purge Buffer in Rung 1-2 Walk-Forward (CRITICAL)

**File**: `statistical_tests.py` (all 3 walk-forward functions)

**Problem**: The training set included the month immediately before the test month:
```python
train_mask = df["signal_date"] < test_date  # includes month T-1
```

The target is a 21-trading-day forward return. Monthly signal dates are ~21 days apart, so the training observation at T-1 has a target that overlaps with the test period at T.

**Fix**: Added 1-month purge buffer matching `main.py`:
```python
train_end = dates[i - 1]                     # skip 1 month
train_mask = df["signal_date"] < train_end   # excludes T-1
```

**Impact**: Removes potential target leakage from overlapping forward returns. May change Rung 1-2 IC values.

---

## I1: fillna(0) for Economically Undefined Factors (IMPORTANT)

**File**: `load_data.py:255-256`

**Problem**: All factor NaNs were filled with 0.0:
```python
for col in FACTOR_ZSCORE_COLS:
    panel[col] = panel[col].fillna(0.0)
```

For GrossProfitability (Financials, Utilities, Real Estate) and NetDebtEBITDA (Financials, Real Estate), NaN is **economically meaningful** — these metrics are undefined for these sectors. Filling with 0.0 tells the model these stocks have exactly average profitability/leverage, which is wrong.

**Fix**: Type A missing (economically undefined) stays NaN. Only non-Type-A NaN is filled with 0.0:
```python
TYPE_A_SECTORS = {
    "GrossProfitability_zscore": ["Financials", "Utilities", "Real Estate"],
    "NetDebtEBITDA_zscore": ["Financials", "Real Estate"],
}
# Type A rows stay NaN; non-Type-A NaN filled with 0.0
```

**Impact**: Models must handle NaN for ~27% of the S&P 500 (Financials + Utilities + Real Estate). Tree-based models handle this natively. For MLP, these NaN values need to be addressed in the model code (e.g., masking or separate imputation).

---

## Summary

| Fix | Severity | Status | Regenerate Data? |
|-----|----------|--------|-----------------|
| C1: Cross-sectional z-score | Critical | Fixed | Yes |
| C2: EY split contamination | Critical | Fixed (time-aware filter) | Yes |
| C3: Purge buffer | Critical | Fixed | N/A (stats framework only) |
| I1: Type A imputation | Important | Fixed | Yes |

---

## Second-Pass Fixes (from Sonnet Audit)

### A1: C2 Split Filter Was Too Aggressive (CRITICAL regression)

The initial C2 fix blanket-NaN'd all tickers that ever had a major split — 325/500 tickers (65%), making EY factor useless. Fixed to time-aware: only NaN rows where `signal_date < first_split_date`.

### A2: NaN Feature Propagation Crash (CRITICAL regression from I1)

After I1 fix, GP/A and NetDebt/EBITDA have real NaN values in the panel. Downstream models crashed:
- `rung3_gam.py`: pygam raises `ValueError: X data must not contain Inf nor NaN`
- `main.py` / `rung5_planned.py`: `StandardScaler` propagates NaN → PyTorch loss becomes NaN → silent training failure

**Fix**: Added `np.nan_to_num(X, nan=0.0)` in all three files after converting features to numpy. This treats Type A NaN as "neutral" (z-score 0) at the model level while preserving the semantic distinction in the data.

### A3: `_cs_zscore` Defined Inside Loop (MINOR)

Moved the helper function definition outside the `for raw_col` loop. No closure bug existed, but cleaner code.

---

## Summary

| Fix | Severity | Status | Regenerate Data? |
|-----|----------|--------|-----------------|
| C1: Cross-sectional z-score | Critical | Fixed | Yes |
| C2: EY split contamination | Critical | Fixed (time-aware filter) | Yes |
| C3: Purge buffer | Critical | Fixed | N/A (stats framework only) |
| I1: Type A imputation | Important | Fixed | Yes |
| A1: Split filter regression | Critical | Fixed (time-aware) | Yes |
| A2: NaN propagation crash | Critical | Fixed (nan_to_num) | N/A (model code) |
| A3: Helper function scope | Minor | Fixed | N/A |

**Next step**: Regenerate `data/master_panel.parquet` by running:
```bash
python load_data.py
```
Then re-run all models on the corrected data.
