# Data Pipeline

This document describes the current baseline data pipeline from raw inputs to the final complete modeling dataset.

## Layout

Raw / intermediate inputs live under `data/raw/`:
- `factor_panel.parquet`
- `xbrl_df.parquet`
- `prices_cache.parquet`
- `prices_unadj_cache.parquet`
- `volume_cache.parquet`
- `macro_features.parquet`
- `split_history.parquet`
- `sector_map.json`

Final baseline modeling outputs live under `data/baseline/`:
- `model_dataset.parquet`

## Goal

The current baseline dataset is designed for the first-pass predictive models:
- LASSO
- kernel methods
- basic MLP-style models

The immediate objective is a complete supervised dataset `T = (X, y)` with:
- no missing predictor entries
- no missing target entries
- cross-sectionally standardized predictors
- cross-sectionally standardized target

This baseline intentionally uses only the 4 broadly valid factors:
- `EarningsYield`
- `AssetGrowth`
- `Accruals`
- `Momentum12_1`

The two structurally problematic factors are excluded for now:
- `GrossProfitability`
- `NetDebtEBITDA`

Those two remain for later work once the baseline modeling path is stable.

## Pipeline

The baseline builder is:
- `python3 data/build_baseline_dataset.py`

Source code:
- [data/build_baseline_dataset.py](/home/hsi/school/MA751-2026-IC-KERNEL-NN-MTL/data/build_baseline_dataset.py)

The pipeline is:

1. Load the raw factor panel from `data/raw/factor_panel.parquet`.

2. Keep only the 4 baseline raw predictors:
   - `EarningsYield`
   - `AssetGrowth`
   - `Accruals`
   - `Momentum12_1`

3. Apply raw-value plausibility caps before any standardization.
   Current caps:
   - `EarningsYield`: `[-0.5, 1.0]`
   - `AssetGrowth`: `[-1.0, 10.0]`
   - `Accruals`: `[-1.5, 1.5]`
   - `Momentum12_1`: `[-0.95, 5.0]`

   Values outside those bounds are set to `NaN`.

4. Median-impute post-existence gaps by month.
   Rule:
   - for each factor and `signal_date`, compute the monthly median across tickers that already "exist" in the sample
   - fill missing values only for those post-existence rows

   Pre-existence gaps are not preserved in the final baseline dataset; rows that still cannot be completed are dropped later.

5. Drop any row that still has a missing predictor after capping and median imputation.

6. Recompute predictor z-scores cross-sectionally by `signal_date`.
   For each month and factor:
   - remove `NaN`s
   - compute mean and std
   - winsorize at `mean ± 3 * std`
   - recompute mean and std on the clipped cross section
   - convert to z-scores

7. Compute the raw target as 21-trading-day forward return from `data/raw/prices_cache.parquet`.

8. Drop rows with missing raw target.

9. Standardize the target cross-sectionally by `signal_date` using the same winsorize + z-score procedure.

10. Write the final dataset to `data/baseline/model_dataset.parquet`.

## Final Dataset

Path:
- [data/baseline/model_dataset.parquet](/home/hsi/school/MA751-2026-IC-KERNEL-NN-MTL/data/baseline/model_dataset.parquet)

Columns:
- `ticker`
- `signal_date`
- `EarningsYield_zscore`
- `AssetGrowth_zscore`
- `Accruals_zscore`
- `Momentum12_1_zscore`
- `target_forward_return_21d_zscore`

Shape:
- `45,201` rows x `7` columns
- `494` unique tickers
- `95` monthly signal dates
- no missing values anywhere

Important structural note:
- this is a complete dataset, but not a perfectly balanced rectangular panel
- monthly row counts range from `456` to `494`
- early months have fewer names because some securities do not yet have enough valid history to survive the baseline filters

## Target Distribution

For `target_forward_return_21d_zscore`:
- Skew: `0.15`
- Kurtosis: `0.99`
- Range: `-3.46` to `+3.96` std
- `1.52%` of rows lie beyond `±3` std
- `0.00%` of rows lie beyond `±4` std
- `0.00%` of rows lie beyond `±5` std

Interpretation:
- the target is nearly symmetric
- tails are controlled
- this is much more coherent for cross-sectional stock prediction than using raw absolute forward returns

## Predictor Notes

The 4 predictors are all standardized cross-sectionally, but they are not equally Gaussian:
- `AssetGrowth_zscore` has the heaviest positive tail
- current max is about `5.50`
- this comes from a small set of very large raw `AssetGrowth` observations plus the current winsorize-then-restandardize procedure

This is not currently treated as a blocker for the baseline LASSO path, but it is worth monitoring in experiments.

## Intended Use

When training baseline models:
- use only the 4 z-scored predictor columns as `X`
- use `target_forward_return_21d_zscore` as `y`
- treat `ticker` and `signal_date` as identifiers / keys, not as model inputs
- split train / validation / test by time, not by random row shuffle

## Rebuild

To rebuild the baseline dataset from the raw inputs:

```bash
python3 data/build_baseline_dataset.py
```

This will overwrite:
- `data/baseline/model_dataset.parquet`
