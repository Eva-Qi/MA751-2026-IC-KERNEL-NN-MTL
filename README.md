# MA751 Final Project — Complexity Ladder with Multi-Task Learning

Cross-sectional stock return prediction using a 5-rung complexity ladder: IC-weighted linear → Lasso → Kernel Ridge → MLP → Multi-Task MLP.

## Data

~500 S&P 500 stocks × 95 months (2016–2024), 13 fundamental factors + FRED macro data.

Raw inputs live under `data/raw/`:
- `factor_panel.parquet` — raw factor panel with missing values
- `xbrl_df.parquet` — standardized SEC XBRL facts
- `prices_cache.parquet` — adjusted close history
- `prices_unadj_cache.parquet` — unadjusted close history
- `volume_cache.parquet` — daily volume
- `macro_features.parquet` — FRED macro indicators
- `split_history.parquet` — split-history cache
- `sector_map.json` — sector classification

Baseline modeling outputs live under `data/baseline/`:
- `model_dataset.parquet` — complete 4-factor z-scored dataset with forward-return target

## Structure

```
├── data/
│   ├── raw/        Source / intermediate datasets
│   └── baseline/   Final complete modeling datasets
├── docs/           Project plan
├── src/            Model code (5 rungs + evaluation)
└── notebooks/      Exploration & visualization
```

## Setup

```bash
pip install -r requirements.txt
```

## Building Baseline Data

Build the complete 4-factor baseline dataset:

```bash
python3 data/build_baseline_dataset.py
```
