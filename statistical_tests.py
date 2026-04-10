"""
Statistical Testing Framework for MA751 Cross-Sectional Return Prediction
==========================================================================

Implements the statistical tests needed to evaluate whether rung transitions
in the 5-rung complexity ladder yield significant improvements.

Tests:
  1. Paired t-test on monthly IC differences
  2. Diebold-Mariano test with Newey-West HAC standard errors
  3. Benjamini-Hochberg-Yekutieli (BHY) FDR correction for multiple comparisons
  4. Ljung-Box test on prediction residuals for autocorrelation diagnostics

Rungs:
  1a  - OLS (plain multivariate linear regression)
  1b  - IC-weighted linear ensemble (univariate OLS combined with IC-weights)
  2   - LASSO (LassoCV with walk-forward)
  5a  - MLP single-task (ret only)
  5b  - MTL (ret + ret3m)
  5c  - MTL (ret + vol)
  5d  - MTL (ret + ret3m + vol)

Usage:
  python statistical_tests.py
"""

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DATA = PROJECT_ROOT / "data" / "baseline" / "model_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Rung 1-2 config (matches notebooks) ─────────────────────────────────────
FEATURES_R12 = [
    "EarningsYield_zscore",
    "AssetGrowth_zscore",
    "Accruals_zscore",
    "Momentum12_1_zscore",
]
TARGET_R12 = "target_forward_return_21d_zscore"
MIN_TRAIN_PERIODS = 12
IC_WEIGHT_WINDOW = 12
LASSO_ALPHAS = np.logspace(-4, 1, 50)
LASSO_CV_FOLDS = 5


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Reproduce Rung 1-2 Results (walk-forward)
# ═══════════════════════════════════════════════════════════════════════════

def run_ols_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    """Rung 1a: Plain OLS walk-forward. Returns per-month IC and RMSE."""
    dates = df["signal_date"].sort_values().unique()
    records = []
    model = LinearRegression()

    for i, test_date in enumerate(dates):
        if i < MIN_TRAIN_PERIODS:
            continue

        # Purge buffer: skip 1 month to avoid target overlap (21d fwd return)
        train_end = dates[i - 1] if i >= 1 else dates[0]
        train_mask = df["signal_date"] < train_end
        test_mask = df["signal_date"] == test_date

        X_train = df.loc[train_mask, FEATURES_R12].values
        y_train = df.loc[train_mask, TARGET_R12].values
        X_test = df.loc[test_mask, FEATURES_R12].values
        y_test = df.loc[test_mask, TARGET_R12].values

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ic, _ = spearmanr(y_pred, y_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        se = (y_test - y_pred) ** 2  # per-stock squared errors

        records.append({
            "date": pd.Timestamp(test_date),
            "IC": ic,
            "RMSE": rmse,
            "mean_se": float(np.mean(se)),
        })

    return pd.DataFrame(records)


def run_ic_ensemble_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    """Rung 1b: IC-weighted linear ensemble walk-forward."""
    dates = df["signal_date"].sort_values().unique()
    ic_history = {f: [] for f in FEATURES_R12}
    records = []

    for i, test_date in enumerate(dates):
        if i < MIN_TRAIN_PERIODS:
            continue

        # Purge buffer: skip 1 month to avoid target overlap
        train_end = dates[i - 1] if i >= 1 else dates[0]
        train_mask = df["signal_date"] < train_end
        test_mask = df["signal_date"] == test_date
        y_train = df.loc[train_mask, TARGET_R12].values
        y_test = df.loc[test_mask, TARGET_R12].values

        if len(y_train) == 0 or len(y_test) == 0:
            continue

        preds_by_feature = {}
        current_ics = {}
        for feat in FEATURES_R12:
            X_tr = df.loc[train_mask, [feat]].values
            X_te = df.loc[test_mask, [feat]].values
            pred_f = LinearRegression().fit(X_tr, y_train).predict(X_te)
            ic_f, _ = spearmanr(pred_f, y_test)
            preds_by_feature[feat] = pred_f
            current_ics[feat] = ic_f

        # Weights from prior IC history (no lookahead)
        weights = {}
        for feat in FEATURES_R12:
            recent = ic_history[feat][-IC_WEIGHT_WINDOW:]
            weights[feat] = max(0.0, np.mean(recent)) if recent else 0.0

        total_w = sum(weights.values())
        if total_w == 0:
            weights = {f: 1 / len(FEATURES_R12) for f in FEATURES_R12}
        else:
            weights = {f: w / total_w for f, w in weights.items()}

        y_pred_ens = sum(weights[f] * preds_by_feature[f] for f in FEATURES_R12)
        ic_ens, _ = spearmanr(y_pred_ens, y_test)
        rmse_ens = np.sqrt(np.mean((y_test - y_pred_ens) ** 2))

        records.append({
            "date": pd.Timestamp(test_date),
            "IC": ic_ens,
            "RMSE": rmse_ens,
            "mean_se": float(np.mean((y_test - y_pred_ens) ** 2)),
        })

        # Update IC history after using it
        for feat in FEATURES_R12:
            ic_history[feat].append(current_ics[feat])

    return pd.DataFrame(records)


def run_lasso_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    """Rung 2: LASSO walk-forward."""
    dates = df["signal_date"].sort_values().unique()
    records = []

    for i, test_date in enumerate(dates):
        if i < MIN_TRAIN_PERIODS:
            continue

        # Purge buffer: skip 1 month to avoid target overlap
        train_end = dates[i - 1] if i >= 1 else dates[0]
        train_mask = df["signal_date"] < train_end
        test_mask = df["signal_date"] == test_date

        X_train = df.loc[train_mask, FEATURES_R12].values
        y_train = df.loc[train_mask, TARGET_R12].values
        X_test = df.loc[test_mask, FEATURES_R12].values
        y_test = df.loc[test_mask, TARGET_R12].values

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = LassoCV(
            alphas=LASSO_ALPHAS, cv=LASSO_CV_FOLDS,
            max_iter=5000, fit_intercept=True,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # When LASSO shrinks all coefficients to zero, predictions are constant
        # (= intercept). spearmanr is undefined for constant arrays.
        if np.std(y_pred) < 1e-12:
            ic = 0.0  # no discriminative signal
        else:
            ic, _ = spearmanr(y_pred, y_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        records.append({
            "date": pd.Timestamp(test_date),
            "IC": ic,
            "RMSE": rmse,
            "mean_se": float(np.mean((y_test - y_pred) ** 2)),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Load Rung 4-5 Results
# ═══════════════════════════════════════════════════════════════════════════

def load_rung5_monthly(variant: str) -> pd.DataFrame:
    """
    Load a Rung 5 results parquet and compute per-month IC and RMSE.
    Returns DataFrame with columns: date, IC, RMSE, mean_se.
    """
    path = OUTPUT_DIR / f"results_{variant}_uw.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    records = []

    for date_val, g in df.groupby("date"):
        g = g.dropna(subset=["y_true", "y_pred"])
        if len(g) < 5:
            continue
        ic = spearmanr(g["y_true"], g["y_pred"]).statistic
        se = (g["y_true"] - g["y_pred"]) ** 2
        rmse = np.sqrt(se.mean())
        records.append({
            "date": pd.Timestamp(date_val),
            "IC": ic,
            "RMSE": rmse,
            "mean_se": float(se.mean()),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════

def paired_ic_ttest(ic_a: np.ndarray, ic_b: np.ndarray) -> dict:
    """
    Paired t-test on monthly IC differences.
    H₀: mean(IC_a - IC_b) = 0
    Returns t-stat, p-value, and mean difference.
    """
    diff = ic_a - ic_b
    n = len(diff)
    if n < 3:
        return {"t_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan, "n": n}

    t_stat, p_value = stats.ttest_rel(ic_a, ic_b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "n": n,
    }


def newey_west_se(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Newey-West HAC standard error for the mean of a time series.
    Uses Bartlett kernel with automatic bandwidth selection.
    """
    n = len(x)
    if n < 3:
        return np.nan

    x_demeaned = x - x.mean()

    if max_lag is None:
        # Andrews (1991) rule-of-thumb: floor(n^(1/3))
        max_lag = max(1, int(np.floor(n ** (1 / 3))))

    # Gamma_0
    gamma_0 = float(np.dot(x_demeaned, x_demeaned)) / n

    # Sum weighted autocovariances (Bartlett kernel)
    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        gamma_j = float(np.dot(x_demeaned[j:], x_demeaned[:-j])) / n
        weight = 1.0 - j / (max_lag + 1)
        nw_var += 2.0 * weight * gamma_j

    # Variance of the mean
    var_mean = nw_var / n
    return np.sqrt(max(var_mean, 0.0))


def diebold_mariano_test(
    se_a: np.ndarray,
    se_b: np.ndarray,
    max_lag: int | None = None,
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Uses squared errors as the loss function.
    H₀: E[L_a - L_b] = 0  (no difference in predictive accuracy)

    Accounts for autocorrelation in the loss differential series
    using Newey-West HAC standard errors.

    Parameters
    ----------
    se_a, se_b : arrays of per-period mean squared errors
    max_lag : maximum lag for Newey-West (None = auto)

    Returns
    -------
    dict with DM_stat, p_value, mean_loss_diff
    """
    d = se_a - se_b  # loss differential
    n = len(d)

    if n < 3:
        return {"DM_stat": np.nan, "p_value": np.nan, "mean_loss_diff": np.nan, "n": n}

    d_bar = d.mean()
    se = newey_west_se(d, max_lag=max_lag)

    if se < 1e-12:
        return {"DM_stat": np.nan, "p_value": np.nan, "mean_loss_diff": float(d_bar), "n": n}

    dm_stat = d_bar / se

    # Two-sided p-value using normal approximation (standard for DM)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return {
        "DM_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_bar),
        "n": n,
    }


def bhy_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg-Yekutieli procedure for FDR control.

    Unlike standard BH, BHY controls FDR under arbitrary dependence
    by multiplying BH thresholds by c(m) = sum(1/k for k=1..m).

    Parameters
    ----------
    p_values : array of raw p-values

    Returns
    -------
    Array of BHY-adjusted p-values (same order as input)
    """
    m = len(p_values)
    if m == 0:
        return np.array([])

    # c(m) = harmonic number
    c_m = sum(1.0 / k for k in range(1, m + 1))

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # BHY adjusted p-values
    adjusted = np.zeros(m)
    for i in range(m):
        rank = i + 1
        adjusted[i] = sorted_pvals[i] * m * c_m / rank

    # Enforce monotonicity (from largest rank down)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Unsort back to original order
    result = np.zeros(m)
    result[sorted_indices] = adjusted

    return result


def ljung_box_test(
    residuals: np.ndarray,
    max_lag: int = 10,
) -> dict:
    """
    Ljung-Box test for autocorrelation in residuals.

    H₀: residuals are independently distributed (no autocorrelation up to lag h).

    Parameters
    ----------
    residuals : array of prediction residuals (per-period, e.g. monthly IC or errors)
    max_lag : maximum lag to test

    Returns
    -------
    dict with Q_stat, p_value, max_lag
    """
    n = len(residuals)
    if n < max_lag + 2:
        max_lag = max(1, n - 2)

    r = residuals - residuals.mean()
    gamma_0 = np.dot(r, r) / n

    if gamma_0 < 1e-12:
        return {"Q_stat": 0.0, "p_value": 1.0, "max_lag": max_lag}

    Q = 0.0
    for k in range(1, max_lag + 1):
        rho_k = np.dot(r[k:], r[:-k]) / (n * gamma_0)
        Q += rho_k ** 2 / (n - k)

    Q *= n * (n + 2)

    p_value = 1.0 - stats.chi2.cdf(Q, df=max_lag)

    return {
        "Q_stat": float(Q),
        "p_value": float(p_value),
        "max_lag": max_lag,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Unified Summary + Pairwise Tests
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary_row(label: str, rung: str, monthly: pd.DataFrame) -> dict:
    """Compute a unified summary row from monthly IC/RMSE data."""
    ic = monthly["IC"].values
    n = len(ic)

    if n < 2:
        return {
            "Rung": rung, "Model": label, "Mean_IC": np.nan,
            "IC_t_stat": np.nan, "IC_gt_0_pct": np.nan,
            "ICIR": np.nan, "Mean_RMSE": np.nan, "n_months": n,
        }

    mean_ic = float(ic.mean())
    ic_std = float(ic.std(ddof=1))
    t_stat = mean_ic / (ic_std / np.sqrt(n)) if ic_std > 1e-8 else np.nan
    icir = mean_ic / ic_std if ic_std > 1e-8 else np.nan

    return {
        "Rung": rung,
        "Model": label,
        "Mean_IC": round(mean_ic, 4),
        "IC_t_stat": round(t_stat, 3),
        "IC_gt_0_pct": round(float((ic > 0).mean()) * 100, 1),
        "ICIR": round(icir, 4),
        "Mean_RMSE": round(float(monthly["RMSE"].mean()), 4),
        "n_months": n,
    }


def _target_group(label: str) -> str:
    """
    Models using different target columns have incomparable squared errors.
    Rung 1-2 predict target_forward_return_21d_zscore (z-scored, RMSE ~1.0).
    Rung 5 predicts fwd_ret_1m (raw returns, RMSE ~0.11).
    DM test on squared errors is only valid within the same target group.
    """
    if label.startswith("1") or label.startswith("2"):
        return "zscore"
    return "raw"


def run_pairwise_tests(
    all_monthly: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Run all pairwise statistical tests between models.
    Returns a DataFrame with one row per pair.

    Notes:
      - IC (rank correlation) is scale-free, so paired IC t-tests are valid
        across all model pairs even with different target definitions.
      - DM test compares squared errors, which are NOT comparable across
        different target scales. Cross-group DM results are flagged.
    """
    labels = sorted(all_monthly.keys())
    rows = []

    for label_a, label_b in combinations(labels, 2):
        ma = all_monthly[label_a]
        mb = all_monthly[label_b]

        same_target = _target_group(label_a) == _target_group(label_b)

        # Align on common dates
        merged = pd.merge(
            ma[["date", "IC", "mean_se"]].rename(columns={"IC": "IC_a", "mean_se": "se_a"}),
            mb[["date", "IC", "mean_se"]].rename(columns={"IC": "IC_b", "mean_se": "se_b"}),
            on="date",
            how="inner",
        )

        n_common = len(merged)
        if n_common < 3:
            rows.append({
                "model_a": label_a,
                "model_b": label_b,
                "n_common_months": n_common,
                "same_target_scale": same_target,
                "ic_ttest_t": np.nan,
                "ic_ttest_p": np.nan,
                "ic_mean_diff": np.nan,
                "dm_stat": np.nan,
                "dm_p": np.nan,
                "dm_mean_loss_diff": np.nan,
            })
            continue

        # Paired t-test on IC (always valid — IC is scale-free)
        tt = paired_ic_ttest(merged["IC_a"].values, merged["IC_b"].values)

        # Diebold-Mariano on squared errors (only meaningful within same target)
        if same_target:
            dm = diebold_mariano_test(merged["se_a"].values, merged["se_b"].values)
        else:
            dm = {"DM_stat": np.nan, "p_value": np.nan, "mean_loss_diff": np.nan}

        rows.append({
            "model_a": label_a,
            "model_b": label_b,
            "n_common_months": n_common,
            "same_target_scale": same_target,
            "ic_ttest_t": round(tt["t_stat"], 3) if pd.notna(tt["t_stat"]) else np.nan,
            "ic_ttest_p": round(tt["p_value"], 4) if pd.notna(tt["p_value"]) else np.nan,
            "ic_mean_diff": round(tt["mean_diff"], 4) if pd.notna(tt["mean_diff"]) else np.nan,
            "dm_stat": round(dm["DM_stat"], 3) if pd.notna(dm["DM_stat"]) else np.nan,
            "dm_p": round(dm["p_value"], 4) if pd.notna(dm["p_value"]) else np.nan,
            "dm_mean_loss_diff": round(dm["mean_loss_diff"], 6) if pd.notna(dm["mean_loss_diff"]) else np.nan,
        })

    pairwise_df = pd.DataFrame(rows)

    # Apply BHY correction to both p-value columns
    for col in ["ic_ttest_p", "dm_p"]:
        raw_p = pairwise_df[col].values
        valid_mask = ~np.isnan(raw_p)
        if valid_mask.sum() > 0:
            corrected = np.full_like(raw_p, np.nan)
            corrected[valid_mask] = bhy_correction(raw_p[valid_mask])
            pairwise_df[f"{col}_bhy"] = np.round(corrected, 4)
        else:
            pairwise_df[f"{col}_bhy"] = np.nan

    return pairwise_df


def run_ljung_box_all(
    all_monthly: dict[str, pd.DataFrame],
    max_lag: int = 10,
) -> pd.DataFrame:
    """Run Ljung-Box on IC residuals for each model."""
    rows = []
    for label, monthly in sorted(all_monthly.items()):
        ic = monthly["IC"].values
        if len(ic) < 5:
            continue
        lb = ljung_box_test(ic, max_lag=min(max_lag, len(ic) - 2))
        rows.append({
            "model": label,
            "LB_Q_stat": round(lb["Q_stat"], 3),
            "LB_p_value": round(lb["p_value"], 4),
            "LB_max_lag": lb["max_lag"],
            "n_months": len(ic),
            "autocorrelated": lb["p_value"] < 0.05,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_monthly: dict[str, pd.DataFrame] = {}

    # ── Rung 1-2: Run walk-forward from baseline data ────────────────────
    if BASELINE_DATA.exists():
        print("Loading baseline data for Rung 1-2...")
        df_base = pd.read_parquet(BASELINE_DATA)
        df_base = df_base.sort_values("signal_date").reset_index(drop=True)

        print("  Running Rung 1a (OLS)...")
        all_monthly["1a_OLS"] = run_ols_walkforward(df_base)

        print("  Running Rung 1b (IC-weighted ensemble)...")
        all_monthly["1b_IC_Ensemble"] = run_ic_ensemble_walkforward(df_base)

        print("  Running Rung 2 (LASSO)...")
        all_monthly["2_LASSO"] = run_lasso_walkforward(df_base)
    else:
        print(f"WARNING: Baseline data not found at {BASELINE_DATA}")

    # ── Rung 4-5: Load from saved parquets ───────────────────────────────
    for variant in ["5a", "5b", "5c", "5d"]:
        label_map = {
            "5a": "5a_MLP_ret",
            "5b": "5b_MTL_ret_ret3m",
            "5c": "5c_MTL_ret_vol",
            "5d": "5d_MTL_ret_ret3m_vol",
        }
        monthly = load_rung5_monthly(variant)
        if not monthly.empty:
            all_monthly[label_map[variant]] = monthly
            print(f"  Loaded Rung {variant}: {len(monthly)} months")
        else:
            print(f"  WARNING: Rung {variant} results not found")

    if len(all_monthly) < 2:
        print("ERROR: Need at least 2 models to run pairwise tests.")
        sys.exit(1)

    # ── Unified comparison table ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("UNIFIED COMPARISON TABLE")
    print("=" * 80)

    rung_map = {
        "1a_OLS": "1",
        "1b_IC_Ensemble": "1",
        "2_LASSO": "2",
        "5a_MLP_ret": "4-5",
        "5b_MTL_ret_ret3m": "4-5",
        "5c_MTL_ret_vol": "4-5",
        "5d_MTL_ret_ret3m_vol": "4-5",
    }

    summary_rows = []
    for label in sorted(all_monthly.keys()):
        row = compute_summary_row(label, rung_map.get(label, "?"), all_monthly[label])
        summary_rows.append(row)

    comparison_df = pd.DataFrame(summary_rows)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(OUTPUT_DIR / "statistical_comparison.csv", index=False)
    print(f"\nSaved -> {OUTPUT_DIR / 'statistical_comparison.csv'}")

    # ── Pairwise statistical tests ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("PAIRWISE STATISTICAL TESTS")
    print("=" * 80)

    pairwise_df = run_pairwise_tests(all_monthly)

    # Print readable version
    print(f"\nTotal pairs: {len(pairwise_df)}")
    print(f"  (C({len(all_monthly)},2) = {len(all_monthly) * (len(all_monthly)-1) // 2})")
    print()

    display_cols = [
        "model_a", "model_b", "n_common_months", "same_target_scale",
        "ic_mean_diff", "ic_ttest_t", "ic_ttest_p", "ic_ttest_p_bhy",
        "dm_stat", "dm_p", "dm_p_bhy",
    ]
    print(pairwise_df[display_cols].to_string(index=False))

    cross_target = pairwise_df[~pairwise_df["same_target_scale"]]
    if len(cross_target) > 0:
        print(f"\nNOTE: {len(cross_target)} pair(s) compare models with different target scales")
        print("      (zscore vs raw returns). DM test is N/A for these — only IC t-test applies.")

    pairwise_df.to_csv(OUTPUT_DIR / "pairwise_tests.csv", index=False)
    print(f"\nSaved -> {OUTPUT_DIR / 'pairwise_tests.csv'}")

    # ── Significant results ──────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("SIGNIFICANT RESULTS (BHY-corrected p < 0.10)")
    print("-" * 80)

    sig_ic = pairwise_df[pairwise_df["ic_ttest_p_bhy"] < 0.10]
    sig_dm = pairwise_df[pairwise_df["dm_p_bhy"] < 0.10]

    if len(sig_ic) > 0:
        print("\nIC t-test significant pairs:")
        for _, r in sig_ic.iterrows():
            direction = ">" if r["ic_mean_diff"] > 0 else "<"
            print(f"  {r['model_a']} {direction} {r['model_b']}: "
                  f"diff={r['ic_mean_diff']:.4f}, t={r['ic_ttest_t']:.3f}, "
                  f"p_bhy={r['ic_ttest_p_bhy']:.4f}")
    else:
        print("\nNo IC t-test pairs significant at BHY-corrected 0.10 level.")

    if len(sig_dm) > 0:
        print("\nDiebold-Mariano significant pairs:")
        for _, r in sig_dm.iterrows():
            better = r["model_a"] if r["dm_mean_loss_diff"] < 0 else r["model_b"]
            print(f"  {r['model_a']} vs {r['model_b']}: "
                  f"DM={r['dm_stat']:.3f}, p_bhy={r['dm_p_bhy']:.4f} "
                  f"(lower loss: {better})")
    else:
        print("\nNo Diebold-Mariano pairs significant at BHY-corrected 0.10 level.")

    # ── Ljung-Box autocorrelation diagnostics ────────────────────────────
    print("\n" + "=" * 80)
    print("LJUNG-BOX AUTOCORRELATION TEST ON IC SERIES")
    print("=" * 80)

    lb_df = run_ljung_box_all(all_monthly)
    print(lb_df.to_string(index=False))

    autocorr_models = lb_df[lb_df["autocorrelated"]]
    if len(autocorr_models) > 0:
        print(f"\nWARNING: {len(autocorr_models)} model(s) show significant IC autocorrelation:")
        for _, r in autocorr_models.iterrows():
            print(f"  {r['model']}: Q={r['LB_Q_stat']:.3f}, p={r['LB_p_value']:.4f}")
        print("  -> Standard errors may be understated; consider HAC adjustments.")
    else:
        print("\nNo models show significant IC autocorrelation at 5% level.")

    # ── Overall narrative ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_ic = comparison_df.loc[comparison_df["Mean_IC"].idxmax()]
    print(f"\nBest mean IC: {best_ic['Model']} ({best_ic['Mean_IC']:.4f})")

    best_rmse = comparison_df.loc[comparison_df["Mean_RMSE"].idxmin()]
    print(f"Best RMSE   : {best_rmse['Model']} ({best_rmse['Mean_RMSE']:.4f})")

    # Date range note
    print("\nNOTE: Rung 1-2 models have 83 test months (2016-01 to 2022-11).")
    print("      Rung 5 models have 34 test months (2020-02 to 2022-11).")
    print("      Pairwise tests use only overlapping months for fair comparison.")

    n_sig = len(sig_ic) + len(sig_dm)
    if n_sig == 0:
        print("\nCONCLUSION: No pairwise differences are statistically significant")
        print("            after BHY correction. The complexity ladder does not")
        print("            yield reliably detectable improvements on this dataset.")
    else:
        print(f"\nCONCLUSION: {n_sig} pairwise comparison(s) significant at BHY 10% level.")

    print()


if __name__ == "__main__":
    main()
