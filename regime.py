import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM

DATA_DIR = Path("data/raw/")
OUTPUT_DIR = Path("data/")

PRICES_CACHE = DATA_DIR / "prices_cache.parquet"
MACRO_FEATURES = DATA_DIR / "macro_features.parquet"
MASTER_PANEL = OUTPUT_DIR / "master_panel.parquet"

REGIME_MONTHLY_OUT = OUTPUT_DIR / "monthly_regimes.parquet"
MASTER_PANEL_WITH_REGIMES_OUT = OUTPUT_DIR / "master_panel_with_regimes.parquet"

MARKET_TICKER = "SPY"
N_REGIMES = 3
RV_WINDOW_DAYS = 21

# Features used for the HMM
HMM_FEATURE_COLS = [
    "mkt_ret_1m",
    "mkt_rv_1m",
    "VIXCLS",
    "T10Y2Y",
    "BAMLH0A0HYM2",
]

def build_market_monthly_features(
    prices_path: Path,
    macro_path: Path,
    market_ticker: str = "SPY",
    rv_window_days: int = 21,
) -> pd.DataFrame:
    prices = pd.read_parquet(prices_path)
    prices.index = pd.DatetimeIndex(prices.index)

    if market_ticker not in prices.columns:
        print(
            f"  Warning: {market_ticker} not in price file — "
            "constructing equal-weighted market proxy from all constituents."
        )
        # Equal-weighted daily price index: start each series at 1, then compound
        daily_rets = prices.pct_change()
        ew_ret = daily_rets.mean(axis=1)
        ew_price = (1 + ew_ret).cumprod()
        ew_price.iloc[0] = 1.0
        spy = ew_price.rename("price").to_frame()
    else:
        spy = prices[[market_ticker]].copy().rename(columns={market_ticker: "price"})
    spy = spy.sort_index()

    # Daily returns and rolling realized vol
    spy["ret_daily"] = spy["price"].pct_change()
    spy["rv_daily"] = spy["ret_daily"].rolling(rv_window_days).std()

    # Monthly end-of-month features
    monthly_price = spy["price"].resample("ME").last()
    monthly_ret = monthly_price.pct_change(1)

    # Use last available rolling vol in each month
    monthly_rv = spy["rv_daily"].resample("ME").last()

    market_df = pd.DataFrame({
        "date": monthly_price.index,
        "mkt_ret_1m": monthly_ret.values,
        "mkt_rv_1m": monthly_rv.values,
    })

    macro = pd.read_parquet(macro_path)
    macro.index = pd.DatetimeIndex(macro.index)
    macro_monthly = macro.resample("ME").last().reset_index()

    # keep only cols that exist
    keep_macro = ["date"] + [c for c in ["VIXCLS", "T10Y2Y", "BAMLH0A0HYM2"] if c in macro_monthly.columns]
    macro_monthly = macro_monthly[keep_macro].copy()

    # Merge
    out = pd.merge(market_df, macro_monthly, on="date", how="left")
    out = out.sort_values("date").reset_index(drop=True)

    # Shift macro by 0 or 1?
    # To be conservative for prediction at end of month t for month t+1,
    # use only values known by end of month t. Monthly-resampled macro is fine
    # if the source is contemporaneously available. If you want stricter timing,
    # shift by 1 below:
    # macro_cols = [c for c in out.columns if c not in ["date", "mkt_ret_1m", "mkt_rv_1m"]]
    # out[macro_cols] = out[macro_cols].shift(1)

    return out


def fit_hmm_and_predict_posteriors(
    monthly_df: pd.DataFrame,
    feature_cols: list[str],
    n_regimes: int = 3,
    random_state: int = 42,
    n_iter: int = 300,
) -> tuple[pd.DataFrame, GaussianHMM, StandardScaler]:
    df = monthly_df.copy().sort_values("date").reset_index(drop=True)

    # Drop months without sufficient info
    use_cols = ["date"] + feature_cols
    df = df[use_cols].copy()

    # modest imputation
    for c in feature_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].ffill().bfill()

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if len(df) < 24:
        raise ValueError("Too few monthly observations to fit a stable HMM.")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)

    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    hmm.fit(X)

    hidden_states = hmm.predict(X)
    state_probs = hmm.predict_proba(X)

    out = df[["date"]].copy()
    out["raw_state"] = hidden_states

    for k in range(n_regimes):
        out[f"raw_p{k}"] = state_probs[:, k]

    return out, hmm, scaler


def reorder_states_by_risk(monthly_regimes: pd.DataFrame, monthly_features: pd.DataFrame) -> pd.DataFrame:
    """
    Re-label states so that:
      regime 0 = calm / low-vol
      regime 1 = middle
      regime 2 = stressed / high-vol

    Ordering uses mean market realized vol and then mean return as tie-breaker.
    """
    df = pd.merge(
        monthly_regimes,
        monthly_features[["date", "mkt_ret_1m", "mkt_rv_1m"]],
        on="date",
        how="left",
    )

    stats = (
        df.groupby("raw_state")[["mkt_rv_1m", "mkt_ret_1m"]]
        .mean()
        .reset_index()
        .sort_values(["mkt_rv_1m", "mkt_ret_1m"], ascending=[True, True])
        .reset_index(drop=True)
    )

    mapping = {int(row["raw_state"]): i for i, (_, row) in enumerate(stats.iterrows())}

    out = monthly_regimes.copy()
    out["regime_label"] = out["raw_state"].map(mapping)

    # remap probabilities
    n_regimes = len(mapping)
    for new_k in range(n_regimes):
        old_k = [old for old, new in mapping.items() if new == new_k][0]
        out[f"regime_p{new_k}"] = out[f"raw_p{old_k}"]

    keep_cols = ["date", "regime_label"] + [f"regime_p{k}" for k in range(n_regimes)]
    out = out[keep_cols].copy()

    # lag regime by 1 month so month t features generate regime used for t+1 prediction
    out = out.sort_values("date").reset_index(drop=True)
    out["regime_label"] = out["regime_label"].shift(1)
    for k in range(n_regimes):
        out[f"regime_p{k}"] = out[f"regime_p{k}"].shift(1)

    return out


def merge_regimes_into_panel(
    panel_path: Path,
    regimes_monthly_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])

    regimes = pd.read_parquet(regimes_monthly_path)
    regimes["date"] = pd.to_datetime(regimes["date"])

    merged = pd.merge(panel, regimes, on="date", how="left")
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)

    merged.to_parquet(output_path, index=False)
    return merged


def print_regime_report(monthly_regimes: pd.DataFrame, market_features: pd.DataFrame) -> None:
    df = pd.merge(monthly_regimes, market_features, on="date", how="left")

    print("\n" + "=" * 72)
    print("MONTHLY REGIME REPORT")
    print("=" * 72)

    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Months with regimes: {df['regime_label'].notna().sum()}")

    print("\nCounts by regime:")
    print(df["regime_label"].value_counts(dropna=False).sort_index().to_string())

    print("\nAverage market features by regime:")
    summary = (
        df.dropna(subset=["regime_label"])
        .groupby("regime_label")[["mkt_ret_1m", "mkt_rv_1m"]]
        .mean()
        .round(4)
    )
    print(summary.to_string())

    print("\nSample:")
    print(df[["date", "regime_label", "regime_p0", "regime_p1", "regime_p2"]].tail(12).to_string(index=False))

    print("=" * 72 + "\n")

def main():
    print("Building monthly market regime features...")
    monthly_features = build_market_monthly_features(
        prices_path=PRICES_CACHE,
        macro_path=MACRO_FEATURES,
        market_ticker=MARKET_TICKER,
        rv_window_days=RV_WINDOW_DAYS,
    )

    available_hmm_features = [c for c in HMM_FEATURE_COLS if c in monthly_features.columns]
    if len(available_hmm_features) < 2:
        raise ValueError(
            f"Not enough HMM features available. Requested {HMM_FEATURE_COLS}, "
            f"available: {list(monthly_features.columns)}"
        )

    print(f"Using HMM features: {available_hmm_features}")

    print("Fitting HMM...")
    raw_regimes, hmm_model, scaler = fit_hmm_and_predict_posteriors(
        monthly_df=monthly_features,
        feature_cols=available_hmm_features,
        n_regimes=N_REGIMES,
    )

    print("Reordering states into calm -> middle -> stressed...")
    monthly_regimes = reorder_states_by_risk(raw_regimes, monthly_features)

    monthly_regimes.to_parquet(REGIME_MONTHLY_OUT, index=False)
    print(f"Saved monthly regimes -> {REGIME_MONTHLY_OUT}")

    print_regime_report(monthly_regimes, monthly_features)

    print("Merging regimes into master panel...")
    merged = merge_regimes_into_panel(
        panel_path=MASTER_PANEL,
        regimes_monthly_path=REGIME_MONTHLY_OUT,
        output_path=MASTER_PANEL_WITH_REGIMES_OUT,
    )
    print(f"Saved master panel with regimes -> {MASTER_PANEL_WITH_REGIMES_OUT}")
    print(f"Rows: {len(merged):,}")


if __name__ == "__main__":
    main()