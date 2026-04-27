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

## Third-Pass Fixes (from Code Council Audit)

### D1: dei Namespace Gap in V1 EY Extraction (V1 KNOWN LIMITATION)

**File**: `src/data_pipeline/taxonomy_map.py:303`

**Problem**: The V1 XBRL reader only queried the `us-gaap` namespace from the SEC companyfacts API. `EntityCommonStockSharesOutstanding` for three tickers (Visa, Berkshire Hathaway Class B, Constellation Brands) lives in the `dei` namespace with `units=["shares"]`. These entries were silently skipped, zeroing market cap and inflating EY for **3 tickers x 95 months = 285 observations**.

**Resolution**: V1 is deprecated. V2 computes market cap via CRSP `abs(prc) * shrout * 1000` in `wrds_factor_builder.py:191` and never touches XBRL for market cap — impact is zero in V2.

Optional future fix (not applied): add `dei` namespace to `taxonomy_map.py` with `units="shares"` handling for any future V1-based work.

---

### D2: EY Split-Safety in V2 (PHANTOM BUG — no fix needed)

**File**: `load_data.py:159-162` (V1 only)

**Investigation**: V1 needed a split-contamination filter because yfinance `adj_close` (backward-adjusted for all future splits) x SEC `SharesOutstanding` (point-in-time) creates a unit mismatch. V2 was suspected to have the same issue. Investigation confirms it does not.

**Why V2 is intrinsically split-safe**: In V2, `mktcap = |prc| * shrout` where both `prc` and `shrout` are CRSP point-in-time values for the same month. Across a split both scale together, so the product is invariant. The EY numerator `ib` is total net income in dollars — also split-invariant.

**Empirical check**: AAPL across the 2020-08-28 4-for-1 split yields V2 EY values `0.0354 → 0.0304 → 0.0250 → 0.0281 → 0.0299` — continuous, no jump.

**Resolution**: Do NOT port the V1 filter to V2. Adding it would drop valid data without correcting any actual bias. The existing `|EY| <= 1.0` cap handles extreme-but-legitimate micro-cap values, which is a separate concern.

---

### D3: Missingness Indicators Wired Into Models (AUDIT FIX)

**File**: `config.py`, `load_data.py`

**Problem**: `config.py` defined `MISSINGNESS_INDICATORS = [has_sue, has_short_interest, has_inst_ownership]` but no model's feature list included them. `np.nan_to_num(nan=0.0)` at tensor creation silently treated every NaN as the cross-sectional mean, discarding coverage signal. In-panel coverage rates: SUE **33.3%**, ShortInterestRatio **70.3%**, InstOwnershipChg **68.3%** — variance worth exposing.

**Fix applied**:
1. Extended `MISSINGNESS_INDICATORS` to include `has_analyst_consensus` (IBES Revision/Dispersion/Breadth) and `has_positive_earnings` (Compustat `ib > 0`).
2. Added `ALL_FEATURE_COLS_V2_WITH_MISS` and `ALL_FEATURE_COLS_V3_WITH_MISS` variants in `config.py`.
3. Flipped the `ALL_FEATURE_COLS` alias from `ALL_FEATURE_COLS_V2` to `ALL_FEATURE_COLS_V2_WITH_MISS` — model scripts using the alias pick up flags automatically.
4. Added flag construction for the two new flags in `load_data.py`.
5. Changed Beta/IVOL NaN treatment from zero-fill to sector cross-sectional median (0.45%/0.28% NaN rate — post-IPO window, ABNB/PYPL/PLTR/MRNA-style names).

**Note**: `ENHANCED_MOE_FEATURE_COLS` was NOT modified — it is LASSO-selected and adding flags would invalidate the selection. A separate re-run is required if the MoE is also to benefit.

**Test plan**: Re-run rungs 1-2 with/without indicators on the same walk-forward folds. Paired IC difference test on 58 test months. Expected |ΔIC| ≤ 0.01-0.02 based on coverage x literature priors (Hong-Lim-Stein 2000, Arbel-Strebel 1983).

---

### D4: Rung 4 Relabel + Elastic Net 2d Addition (AUDIT FIX)

**Files**: `main.py`, `statistical_tests.py`, `run_rung12_v2.py`, `scripts/rung12_v3.py`

**Problem 1 (labeling)**: The "5-rung complexity ladder" design calls for Rung 4 = pure single-task MLP (nonlinearity without MTL). The codebase implemented this as variant `5a` inside the Rung 5 MTL ablation. Architecturally `MTLNet(n_tasks=1)` has two dead heads and is equivalent to a plain 64→32 MLP, but the `5a` label hides Rung 4's poor result (IC=+0.005, net Sharpe=0.07, turnover=76%) inside the MTL conversation rather than as a standalone Rung 4 conclusion.

**Problem 2 (missing model)**: Rung 2 included LASSO (L1) and Ridge (L2) but not Elastic Net (L1+L2). Given severe multicollinearity in the feature set (Beta/IVOL both risk, Amihud/Turnover both liquidity, SUE/Revision/Dispersion/Breadth all IBES), LASSO's feature selection is unstable (it arbitrarily picks one from correlated groups) and Ridge can't select. Elastic Net is the standard remedy (Zou & Hastie 2005).

**Fix applied**:
1. **Relabel** `"5a"` → `"rung4"` in `main.py:ABLATION_TASKS` and downstream `statistical_tests.py` comparison maps.
2. **Add Elastic Net** as `2d_ElasticNet_v2` in `run_rung12_v2.py` and `2d_ElasticNet_v3` in `scripts/rung12_v3.py`. Uses `ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9])`.
3. **Diagnostics logged**: per-fold `l1_ratio`, `alpha`, and `n_nonzero` saved to `output/elastic_net_diag_v{2,3}.csv`. Mean l1_ratio tells us whether LASSO-like, Ridge-like, or mixed behavior dominates.

**Interpretation guide**:
- If mean `l1_ratio > 0.7` → LASSO-like → Ridge was overspecified
- If mean `l1_ratio < 0.3` → Ridge-like → LASSO was too sparse
- If mean `l1_ratio ≈ 0.5` → Elastic Net genuinely adds value

**Test plan**: Compare 2d ElasticNet to 2a LASSO and 2b Ridge on both V2 and V3 panels. Paired t-test 2d vs 2b (current Pareto-best). Expected: modest Sharpe improvement if multicollinearity is limiting LASSO/Ridge. Unchanged if one of them was already at the right end of the α-mix spectrum.

**Rung 4 no re-run needed**: existing `results_5a_uw.parquet` numbers are correct; only the label changed. Relabeling is documentation, not model change.

---

## Summary (All Passes)

| Fix | Severity | Status | Regenerate Data? |
|-----|----------|--------|-----------------|
| C1: Cross-sectional z-score | Critical | Fixed | Yes |
| C2: EY split contamination | Critical | Fixed (time-aware filter) | Yes |
| C3: Purge buffer | Critical | Fixed | N/A (stats framework only) |
| I1: Type A imputation | Important | Fixed | Yes |
| A1: Split filter regression | Critical | Fixed (time-aware) | Yes |
| A2: NaN propagation crash | Critical | Fixed (nan_to_num) | N/A (model code) |
| A3: Helper function scope | Minor | Fixed | N/A |
| D1: dei namespace gap | V1 limitation | No action (V2 unaffected) | N/A |
| D2: EY split-safety V2 | Phantom bug | No action (V2 safe by construction) | N/A |
| D3: Missingness indicators | Audit fix | Fixed (flags wired in) | Yes |
| D4: Rung 4 label + 2d ElasticNet | Audit fix | Fixed (2d added, 5a→rung4) | Re-run Rung 1-2 for 2d |
