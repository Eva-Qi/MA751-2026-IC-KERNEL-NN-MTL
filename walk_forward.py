"""
Shared walk-forward evaluation framework for MA751.

Provides the date-splitting logic used by all rungs. Each rung implements
its own model training but reuses the same temporal split to ensure
comparable results.
"""

import numpy as np
import pandas as pd
from config import DATE_COL, STOCK_COL, DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS


def walk_forward_splits(
    dates: np.ndarray,
    min_train_months: int = DEFAULT_MIN_TRAIN_MONTHS,
    purge_months: int = DEFAULT_PURGE_MONTHS,
):
    """
    Yield (train_end_date, test_date, fold_index) tuples for expanding-window
    walk-forward evaluation.

    Parameters
    ----------
    dates : array-like of unique sorted month-end dates
    min_train_months : minimum number of months in training window
    purge_months : gap between train end and test month (avoids target overlap)

    Yields
    ------
    (train_end_date, test_date, fold_index)
        train_end_date : last date included in training set
        test_date : the month being predicted
        fold_index : integer index of the fold
    """
    months = sorted(dates)
    start = min_train_months + purge_months

    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - purge_months - 1]
        yield train_end, test_month, i


def split_train_test(
    df: pd.DataFrame,
    train_end,
    test_date,
    date_col: str = DATE_COL,
):
    """
    Split dataframe into train and test sets based on dates.

    Returns
    -------
    (df_train, df_test)
    """
    df_tr = df[df[date_col] <= train_end].copy()
    df_te = df[df[date_col] == test_date].copy()
    return df_tr, df_te


def run_walk_forward(
    df: pd.DataFrame,
    fit_predict_fn,
    min_train_months: int = DEFAULT_MIN_TRAIN_MONTHS,
    purge_months: int = DEFAULT_PURGE_MONTHS,
    verbose: bool = True,
):
    """
    Generic walk-forward loop. Delegates model fitting/prediction to
    `fit_predict_fn`.

    Parameters
    ----------
    df : panel DataFrame with DATE_COL, STOCK_COL, features, and target
    fit_predict_fn : callable(df_train, df_test, fold_index) -> pd.DataFrame
        Must return a DataFrame with at least: DATE_COL, STOCK_COL, "y_true", "y_pred"
    min_train_months : int
    purge_months : int
    verbose : bool

    Returns
    -------
    pd.DataFrame of concatenated fold results
    """
    from scipy.stats import spearmanr

    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []

    for train_end, test_month, fold_idx in walk_forward_splits(
        months, min_train_months, purge_months
    ):
        df_tr, df_te = split_train_test(df, train_end, test_month)

        if df_te.empty:
            continue

        out = fit_predict_fn(df_tr, df_te, fold_idx)
        results.append(out)

        if verbose and "y_true" in out.columns and "y_pred" in out.columns:
            ic = spearmanr(out["y_true"], out["y_pred"]).statistic
            print(
                f"fold {fold_idx:3d} | {str(test_month)[:7]} | "
                f"n_train={len(df_tr):6,d} | n_test={len(df_te):4d} | IC={ic:+.4f}"
            )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
