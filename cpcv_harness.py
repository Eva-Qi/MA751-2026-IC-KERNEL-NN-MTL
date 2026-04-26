"""
Combinatorial Purged K-fold Cross-Validation (CPCV) harness.

Per López de Prado, Advances in Financial Machine Learning (2018), Chapter 12.

Key properties:
  - Split panel into N_blocks contiguous time blocks
  - For each C(N_blocks, k_test) combo: train on remaining blocks, test on k_test blocks
  - Purge: remove embargo_months of train data adjacent to any test block
  - Returns list of path dicts — each path is one complete backtest
  - C(6, 2) = 15 paths total (default config)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from config import TARGET_COL, DATE_COL, STOCK_COL, ALL_FEATURE_COLS_V3_WITH_MISS
from metrics import compute_long_short_sharpe, compute_ic_ir


# ─────────────────────────────────────────────────────────────────────────────
# Block construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_blocks(months_sorted, N_blocks):
    """Split sorted month array into N_blocks contiguous groups.

    Returns: list of sets, each containing the month timestamps in that block.
    Block sizes differ by at most 1 (distributes remainder across early blocks).
    """
    n = len(months_sorted)
    base = n // N_blocks
    remainder = n % N_blocks
    blocks = []
    start = 0
    for i in range(N_blocks):
        size = base + (1 if i < remainder else 0)
        end = start + size
        blocks.append(set(months_sorted[start:end]))
        start = end
    return blocks  # list of N_blocks sets of month timestamps


def _purge_adjacent(train_months, test_months_set, all_months_sorted, embargo_months):
    """Remove embargo_months adjacent to test blocks from train_months.

    For each test block boundary (first and last month), find the neighboring
    months in all_months_sorted and exclude them from train set.
    Purge is bidirectional: removes embargo months before AND after each test block.
    """
    test_sorted = sorted(test_months_set)
    if not test_sorted:
        return train_months

    all_arr = np.array(all_months_sorted)
    purge_set = set()

    for t_month in test_months_set:
        idx = np.searchsorted(all_arr, t_month)
        # embargo forward (months after this test month — future leakage)
        for k in range(1, embargo_months + 1):
            if idx + k < len(all_arr):
                purge_set.add(all_arr[idx + k])
        # embargo backward (months before this test month — label overlap)
        for k in range(1, embargo_months + 1):
            if idx - k >= 0:
                purge_set.add(all_arr[idx - k])

    return [m for m in train_months if m not in purge_set]


# ─────────────────────────────────────────────────────────────────────────────
# Per-path prediction
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_path(df, model_fn, features, train_months, test_months_sorted):
    """Train model on train_months data, predict on each test month.

    Returns DataFrame with columns: [DATE_COL, STOCK_COL, y_pred, y_true]
    """
    df_train = df[df[DATE_COL].isin(set(train_months))].copy()
    if len(df_train) < 50:
        return pd.DataFrame()

    X_tr = np.nan_to_num(df_train[features].values, nan=0.0)
    y_tr = df_train[TARGET_COL].values
    train_dates = df_train[DATE_COL].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    # Collect test predictions across all test months
    all_rows = []
    for test_month in test_months_sorted:
        df_te = df[df[DATE_COL] == test_month].copy()
        if len(df_te) < 5:
            continue

        X_te = np.nan_to_num(df_te[features].values, nan=0.0)
        y_te = df_te[TARGET_COL].values
        tickers = df_te[STOCK_COL].values

        X_te_s = scaler.transform(X_te)

        try:
            y_pred = model_fn(X_tr_s, y_tr, X_te_s, features, train_dates=train_dates)
        except TypeError:
            # Some adapters don't accept train_dates kwarg
            y_pred = model_fn(X_tr_s, y_tr, X_te_s, features)

        if y_pred is None or len(y_pred) == 0:
            continue

        for ticker, yp, yt in zip(tickers, y_pred, y_te):
            all_rows.append({
                DATE_COL: test_month,
                STOCK_COL: ticker,
                "y_pred": float(yp),
                "y_true": float(yt),
            })

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Monthly IC from expanded predictions
# ─────────────────────────────────────────────────────────────────────────────

def _monthly_ic(expanded_df, min_stocks=5):
    """Compute monthly cross-sectional Spearman IC from expanded predictions."""
    rows = []
    for date, g in expanded_df.groupby(DATE_COL):
        g = g.dropna(subset=["y_pred", "y_true"])
        if len(g) < min_stocks:
            continue
        ic = spearmanr(g["y_true"], g["y_pred"]).statistic
        rows.append({DATE_COL: date, "IC": ic, "n_test": len(g)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main CPCV entry point
# ─────────────────────────────────────────────────────────────────────────────

def cpcv_paths(
    df,
    model_fn,
    features,
    target_col=TARGET_COL,
    date_col=DATE_COL,
    stock_col=STOCK_COL,
    N_blocks=6,
    k_test=2,
    embargo_months=1,
    verbose=True,
):
    """
    Run CPCV: split panel into N_blocks contiguous time blocks,
    run C(N_blocks, k_test) train/test combos.

    Each combo:
      - train on (N_blocks - k_test) blocks (with purge)
      - test on k_test blocks
      - predict all stocks in each test month using a single trained model

    Returns
    -------
    paths : list of dicts, each with:
        - 'path_id'       : int (0-indexed)
        - 'train_blocks'  : tuple of block indices used for training
        - 'test_blocks'   : tuple of block indices used for testing
        - 'train_months'  : list of month timestamps (after purge)
        - 'test_months'   : list of month timestamps (sorted)
        - 'expanded'      : DataFrame [date, ticker, y_pred, y_true]
        - 'monthly_results': DataFrame [date, IC, n_test]
        - 'IC_mean'       : float
        - 'IC_std'        : float
        - 'Sharpe'        : float (annualized LS Sharpe)
        - 'n_obs'         : int (number of test-month ICs)
    """
    df = df.sort_values([date_col, stock_col]).reset_index(drop=True)
    all_months = sorted(df[date_col].unique())
    n_months_total = len(all_months)

    if verbose:
        print(f"  [CPCV] {n_months_total} months → {N_blocks} blocks, "
              f"k_test={k_test}, C({N_blocks},{k_test})={len(list(combinations(range(N_blocks), k_test)))} paths, "
              f"embargo={embargo_months}mo")

    blocks = _build_blocks(all_months, N_blocks)
    block_sizes = [len(b) for b in blocks]
    if verbose:
        print(f"  [CPCV] block sizes: {block_sizes}")

    test_combos = list(combinations(range(N_blocks), k_test))
    paths = []

    for path_id, test_block_ids in enumerate(test_combos):
        train_block_ids = tuple(i for i in range(N_blocks) if i not in test_block_ids)

        # Collect months for train and test
        test_months_set = set()
        for bid in test_block_ids:
            test_months_set.update(blocks[bid])

        raw_train_months = []
        for bid in train_block_ids:
            raw_train_months.extend(sorted(blocks[bid]))
        raw_train_months.sort()

        # Purge embargo months adjacent to test blocks
        purged_train_months = _purge_adjacent(
            raw_train_months, test_months_set, all_months, embargo_months
        )
        test_months_sorted = sorted(test_months_set)

        if verbose:
            print(f"  [CPCV] path {path_id:2d}/{len(test_combos)-1} | "
                  f"test_blocks={test_block_ids} train_blocks={train_block_ids} | "
                  f"train={len(purged_train_months)}mo test={len(test_months_sorted)}mo", flush=True)

        if len(purged_train_months) < 12:
            if verbose:
                print(f"    SKIP: too few train months ({len(purged_train_months)})")
            continue

        # Run model
        expanded = _run_one_path(df, model_fn, features, purged_train_months, test_months_sorted)

        if expanded.empty:
            ic_mean = np.nan
            ic_std = np.nan
            sharpe = np.nan
            monthly = pd.DataFrame()
            n_obs = 0
        else:
            monthly = _monthly_ic(expanded)
            ic_series = monthly["IC"].dropna()
            ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
            ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else np.nan
            sharpe = compute_long_short_sharpe(expanded)
            n_obs = len(ic_series)

        paths.append({
            "path_id": path_id,
            "train_blocks": train_block_ids,
            "test_blocks": test_block_ids,
            "train_months": purged_train_months,
            "test_months": test_months_sorted,
            "expanded": expanded,
            "monthly_results": monthly,
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "Sharpe": sharpe,
            "n_obs": n_obs,
        })

        if verbose:
            print(f"    → IC={ic_mean:+.4f} Sharpe={sharpe:.3f} n_obs={n_obs}", flush=True)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def summarise_paths(paths, label=""):
    """Compute cross-path statistics from list of path dicts."""
    sharpes = np.array([p["Sharpe"] for p in paths if pd.notna(p["Sharpe"])])
    ics = np.array([p["IC_mean"] for p in paths if pd.notna(p["IC_mean"])])

    if len(sharpes) == 0:
        return {"label": label, "n_paths": 0}

    return {
        "label": label,
        "n_paths": len(paths),
        "n_valid_paths": len(sharpes),
        "Sharpe_mean": float(np.mean(sharpes)),
        "Sharpe_std": float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else np.nan,
        "Sharpe_p5": float(np.percentile(sharpes, 5)),
        "Sharpe_p25": float(np.percentile(sharpes, 25)),
        "Sharpe_p50": float(np.percentile(sharpes, 50)),
        "Sharpe_p75": float(np.percentile(sharpes, 75)),
        "Sharpe_p95": float(np.percentile(sharpes, 95)),
        "Prob_neg_Sharpe": float((sharpes < 0).mean()),
        "IC_mean": float(np.mean(ics)) if len(ics) else np.nan,
        "IC_std": float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan,
        "IC_p5": float(np.percentile(ics, 5)) if len(ics) else np.nan,
        "IC_p95": float(np.percentile(ics, 95)) if len(ics) else np.nan,
    }


def paths_to_long_df(paths, label):
    """Convert list of path dicts to long-format DataFrame row(s) for storage."""
    rows = []
    for p in paths:
        rows.append({
            "model": label,
            "path_id": p["path_id"],
            "train_blocks": str(p["train_blocks"]),
            "test_blocks": str(p["test_blocks"]),
            "IC_mean": p["IC_mean"],
            "IC_std": p["IC_std"],
            "Sharpe": p["Sharpe"],
            "n_obs": p["n_obs"],
        })
    return pd.DataFrame(rows)
