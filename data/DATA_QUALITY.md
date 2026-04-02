# Data Quality Notes

## Known Issues (as of 2026-04-02)

The pre-computed `factor_panel.parquet` contains data quality issues in the **EarningsYield** column:

| Issue | Impact | Affected Records |
|-------|--------|-----------------|
| XBRL shares scaling errors (119 tickers) | EY values up to +277,686 | 806 records with \|EY\| > 1.0 |
| Stock split mismatch (yfinance adj prices vs SEC raw shares) | Median EY = 0.09 vs expected 0.03-0.05 | Systematic bias across 73/95 months |
| Zero/implausible shares outstanding | Division errors | 337 records (shares < 1M), 41 records (shares = 0) |
| XBRL assets scaling errors (AMCR=$130, ARES=$1000) | AssetGrowth up to +132,038,461 | 37 records with \|AG\| > 10,000 |
| GrossProfit negative values | min = -$9.05B | Likely accounting (WARN, not FAIL) |

## Validation Script

Run `validate_data_legitimacy.py` in this directory to generate a full diagnostic:

```bash
python3 validate_data_legitimacy.py -o validation_report.md
```

## Recommended Handling

When using `factor_panel.parquet` for model training:

1. **Filter EarningsYield outliers**: Drop rows where `|EarningsYield| > 1.0`
2. **Or use z-scored column**: `EarningsYield_zscore` has winsorization applied (but based on contaminated distribution)
3. **Best**: Recompute from source data with plausibility guards (see AI-QC-RESUME fixes)

## Source Code (with fixes)

Fixed computation code is in `src/`:
- `src/factor_library/academic_factors.py` — all 6 factor functions with plausibility guards
- `src/data_pipeline/data_quality.py` — unit consistency + value magnitude checks
- `src/data_pipeline/taxonomy_map.py` — XBRL concept mapping

### Fixes applied:
- Shares plausibility guard [1M, 100B] in `_get_shares_outstanding()`
- EY magnitude guard |EY| < 1.0 + market_cap > $1B in `compute_earnings_yield()`
- Assets plausibility guard > $100M in `compute_asset_growth()`, `compute_accruals()`, `compute_gross_profitability()`
- AG magnitude guard |AG| > 10 in `compute_asset_growth()`
- Unit filter (USD only) in `compute_ttm()`
- Diagnostic logging in `compute_earnings_yield()` and `compute_asset_growth()`
