"""
Market regime detection via Hidden Markov Model (HMM).

Provides utility functions for building market features, fitting HMM,
and predicting regime posteriors. Designed to be called INSIDE the
walk-forward loop to prevent look-ahead bias.

Usage (standalone — offline, full-sample, for EDA only):
    python regime.py

Usage (correct — inside walk-forward):
    from regime import build_market_monthly_features, fit_and_predict_regime
    market_features = build_market_monthly_features(...)
    regime_df = fit_and_predict_regime(market_features, train_end=train_end)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM

from config import (
    N_REGIMES, HMM_FEATURE_COLS,
    DATE_COL, MACRO_COLS,
)

# ── Paths (for standalone mode) ─────────────────────────────────────────

DATA_DIR = Path("data/raw/")
WRDS_DIR = Path("data/wrds/")
OUTPUT_DIR = Path("data/")

PRICES_CACHE = DATA_DIR / "prices_cache.parquet"
MACRO_FEATURES = DATA_DIR / "macro_features.parquet"

# ── 1. Build market-level monthly features ──────────────────────────────


def build_market_monthly_features(
    prices_path: Path = PRICES_CACHE,
    macro_path: Path = MACRO_FEATURES,
    market_ticker: str = "SPY",
    rv_window_days: int = 21,
) -> pd.DataFrame:
    """
    Build monthly market features for HMM input.

    Returns DataFrame with columns: date, mkt_ret_1m, mkt_rv_1m, VIXCLS, T10Y2Y, BAMLH0A0HYM2
    """
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)

    if market_ticker not in prices.columns:
        # Equal-weighted proxy
        daily_rets = prices.pct_change()
        ew_ret = daily_rets.mean(axis=1)
        ew_price = (1 + ew_ret).cumprod()
        ew_price.iloc[0] = 1.0
        spy = ew_price.rename("price").to_frame()
    else:
        spy = prices[[market_ticker]].copy().rename(columns={market_ticker: "price"})
    spy = spy.sort_index()

    spy["ret_daily"] = spy["price"].pct_change()
    spy["rv_daily"] = spy["ret_daily"].rolling(rv_window_days).std()

    monthly_price = spy["price"].resample("ME").last()
    monthly_ret = monthly_price.pct_change(1)
    monthly_rv = spy["rv_daily"].resample("ME").last()

    market_df = pd.DataFrame({
        "date": monthly_price.index,
        "mkt_ret_1m": monthly_ret.values,
        "mkt_rv_1m": monthly_rv.values,
    })

    # Merge macro features
    macro = pd.read_parquet(macro_path)
    macro.index = pd.DatetimeIndex(macro.index)
    macro_monthly = macro.resample("ME").last().reset_index()

    keep_macro = ["date"] + [c for c in ["VIXCLS", "T10Y2Y", "BAMLH0A0HYM2"]
                             if c in macro_monthly.columns]
    macro_monthly = macro_monthly[keep_macro].copy()

    out = pd.merge(market_df, macro_monthly, on="date", how="left")
    out = out.sort_values("date").reset_index(drop=True)

    return out


# ── 2. Fit HMM and predict regimes (expanding window) ──────────────────


def fit_and_predict_regime(
    monthly_features: pd.DataFrame,
    train_end: pd.Timestamp,
    n_regimes: int = N_REGIMES,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
    n_iter: int = 300,
) -> pd.DataFrame:
    """
    Fit HMM on data up to train_end, predict regime posteriors for ALL months.

    This is the CORRECT way to use HMM in walk-forward: only train data
    informs the model, but we predict on all months (including test) to get
    regime labels for the test month.

    Returns DataFrame with columns: date, regime_label, regime_p0, ..., regime_p{K-1}
    All values are LAGGED by 1 month (month t's regime uses features from month t-1).
    """
    if feature_cols is None:
        feature_cols = [c for c in HMM_FEATURE_COLS if c in monthly_features.columns]
        if len(feature_cols) < 2:
            raise ValueError(
                f"Need ≥2 HMM features. Available: {list(monthly_features.columns)}"
            )

    df = monthly_features.copy().sort_values("date").reset_index(drop=True)

    # Impute missing (forward-fill then back-fill)
    for c in feature_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Split into train (up to train_end) and full
    train_mask = df["date"] <= train_end
    df_train = df[train_mask].copy()

    if len(df_train) < 24:
        raise ValueError(
            f"Only {len(df_train)} train months for HMM (need ≥24). "
            f"train_end={train_end}"
        )

    # Fit scaler + HMM on TRAIN data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)

    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    hmm.fit(X_train)

    # ── Decode TRAIN sequence only (forward-backward within train) ──
    # This avoids the smoother using future test months to revise
    # train months' posteriors — the key look-ahead fix.
    train_states = hmm.predict(X_train)
    train_probs = hmm.predict_proba(X_train)

    out_train = df_train[["date"]].copy().reset_index(drop=True)
    out_train["raw_state"] = train_states
    for k in range(n_regimes):
        out_train[f"raw_p{k}"] = train_probs[:, k]

    # ── For months AFTER train_end: 1-step transition from last state ──
    # Use HMM transition matrix to predict forward without future data.
    df_future = df[~train_mask].copy().reset_index(drop=True)
    if len(df_future) > 0:
        last_probs = train_probs[-1]  # posterior at train_end
        T = hmm.transmat_             # T[i,j] = P(s_{t+1}=j | s_t=i)
        future_entries = []
        current_probs = last_probs
        for idx in range(len(df_future)):
            # 1-step forward: marginalize over current state
            next_probs = current_probs @ T
            entry = {"date": df_future.loc[idx, "date"],
                     "raw_state": int(np.argmax(next_probs))}
            for k in range(n_regimes):
                entry[f"raw_p{k}"] = next_probs[k]
            future_entries.append(entry)
            current_probs = next_probs  # iterate forward
        out_future = pd.DataFrame(future_entries)
        out = pd.concat([out_train, out_future], ignore_index=True)
    else:
        out = out_train

    # Reorder states: regime 0 = calm (low vol), regime 2 = stressed (high vol)
    out = _reorder_states_by_risk(out, df, n_regimes)

    # LAG by 1 month: month t's regime comes from month t-1's features
    out = out.sort_values("date").reset_index(drop=True)
    out["regime_label"] = out["regime_label"].shift(1)
    for k in range(n_regimes):
        out[f"regime_p{k}"] = out[f"regime_p{k}"].shift(1)

    keep = ["date", "regime_label"] + [f"regime_p{k}" for k in range(n_regimes)]
    return out[keep].copy()


def _reorder_states_by_risk(
    regimes_df: pd.DataFrame,
    features_df: pd.DataFrame,
    n_regimes: int,
) -> pd.DataFrame:
    """Re-label states so regime 0 = calm, regime 2 = stressed."""
    df = pd.merge(
        regimes_df,
        features_df[["date", "mkt_ret_1m", "mkt_rv_1m"]],
        on="date",
        how="left",
    )

    stats = (
        df.groupby("raw_state")[["mkt_rv_1m", "mkt_ret_1m"]]
        .mean()
        .reset_index()
        .sort_values(["mkt_rv_1m"], ascending=[True])
        .reset_index(drop=True)
    )

    mapping = {int(row["raw_state"]): i for i, (_, row) in enumerate(stats.iterrows())}

    out = regimes_df.copy()
    out["regime_label"] = out["raw_state"].map(mapping)

    for new_k in range(n_regimes):
        old_k = [old for old, new in mapping.items() if new == new_k][0]
        out[f"regime_p{new_k}"] = out[f"raw_p{old_k}"]

    return out


# ── 3. Merge regime into stock panel ────────────────────────────────────


def merge_regime_into_panel(
    panel_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_regimes: int = N_REGIMES,
) -> pd.DataFrame:
    """
    Merge monthly regime posteriors into stock-level panel.
    Fills missing regime months with uniform prior (1/N).
    """
    panel = panel_df.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    regime = regime_df.copy()
    regime["date"] = pd.to_datetime(regime["date"])

    merged = pd.merge(panel, regime, on="date", how="left")

    # Fill missing with uniform prior (NOT 0.0 — that would bias the gate)
    uniform = 1.0 / n_regimes
    for k in range(n_regimes):
        col = f"regime_p{k}"
        merged[col] = merged[col].fillna(uniform)

    if "regime_label" in merged.columns:
        merged["regime_label"] = merged["regime_label"].fillna(-1)

    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)


# ── Standalone mode (EDA only — NOT for model training) ─────────────────


def main():
    """
    Standalone: fit HMM on FULL sample for exploratory analysis.
    WARNING: This produces look-ahead-biased labels. For model training,
    use fit_and_predict_regime() inside the walk-forward loop.
    """
    print("=" * 60)
    print("WARNING: Standalone mode fits HMM on full sample.")
    print("These labels have look-ahead bias — use for EDA only.")
    print("=" * 60)

    print("\nBuilding monthly market features...")
    features = build_market_monthly_features()

    available = [c for c in HMM_FEATURE_COLS if c in features.columns]
    print(f"HMM features: {available}")

    # Fit on full sample (EDA mode)
    train_end = features["date"].max()
    regime = fit_and_predict_regime(features, train_end=train_end)

    # Report
    print(f"\nDate range: {regime['date'].min().date()} → {regime['date'].max().date()}")
    print(f"Months with regime: {regime['regime_label'].notna().sum()}")
    print("\nCounts by regime:")
    print(regime["regime_label"].value_counts(dropna=False).sort_index())

    df = pd.merge(regime, features[["date", "mkt_ret_1m", "mkt_rv_1m"]], on="date", how="left")
    summary = (
        df.dropna(subset=["regime_label"])
        .groupby("regime_label")[["mkt_ret_1m", "mkt_rv_1m"]]
        .mean()
        .round(4)
    )
    print("\nAvg market features by regime:")
    print(summary)

    # Save for EDA
    regime.to_parquet(OUTPUT_DIR / "monthly_regimes_eda.parquet", index=False)
    print(f"\nSaved → {OUTPUT_DIR / 'monthly_regimes_eda.parquet'} (EDA only)")


if __name__ == "__main__":
    main()
