# MA751 Final Project — Complexity Ladder with Multi-Task Learning

Cross-sectional stock return prediction using a 5-rung complexity ladder: IC-weighted linear → Lasso → Kernel Ridge → MLP → Multi-Task MLP.

## Data

~500 S&P 500 stocks × 95 months (2016–2024), 13 fundamental factors + FRED macro data.

| File | Size | Source | Description |
|------|------|--------|-------------|
| `xbrl_df.parquet` | 3.1MB | SEC EDGAR | Standardized financials (346K rows, 13 concepts) |
| `prices_cache.parquet` | 9.1MB | yfinance | Daily adjusted close (2831 days × 501 tickers) |
| `volume_cache.parquet` | 8.4MB | yfinance | Daily volume |
| `factor_panel.parquet` | 3.6MB | Computed | 6 academic factors × 95 months × ~500 stocks |
| `macro_features.parquet` | 30KB | FRED | Macro indicators (yield curve, VIX, sentiment, etc.) |
| `sector_map.json` | 13KB | Wikipedia/SEC | Sector classification |

## Structure

```
├── data/           Pre-computed datasets (committed)
├── docs/           Project plan
├── src/            Model code (5 rungs + evaluation)
└── notebooks/      Exploration & visualization
```

## Setup

```bash
pip install -r requirements.txt
```
