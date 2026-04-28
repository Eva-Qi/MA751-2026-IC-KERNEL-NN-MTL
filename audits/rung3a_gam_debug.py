"""
Rung 3a GAM Debug Audit
=======================

Checks why GAM performs best at n_splines = 4/5.

Diagnostics:
1. Partial dependence plots for n_splines in {4, 5, 10, 20}
2. Fold-by-fold selected lambda
3. Train IC vs test IC by n_splines
4. Feature-level effective degrees of freedom

Run from project root:
    python audits/rung3a_gam_debug.py

Outputs:
    output/gam_debug_fold_metrics.csv
    output/gam_debug_feature_edof.csv
    output/gam_debug_summary.csv
    output/gam_debug_partial_dependence_*.pdf
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s

# ---------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS,
    TARGET_COL,
    DATE_COL,
    STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS,
    DEFAULT_PURGE_MONTHS,
)

from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

FEATURES = ALL_FEATURE_COLS_V3_WITH_MISS

N_SPLINES_GRID = [4, 5, 10, 20]
LAM_GRID = np.logspace(-3, 3, 11)

# Pick interpretable high-signal features if available
FEATURES_TO_PLOT = [
    "IVOL_zscore",
    "GrossProfitability_zscore",
    "Beta_zscore",
    "Momentum12_1_zscore",
    "ShortInterestRatio_zscore",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def mean_monthly_ic(df: pd.DataFrame, pred_col: str, y_col: str, date_col: str) -> float:
    """
    Compute average month-by-month Spearman IC.
    This is better than one pooled Spearman over all rows.
    """
    ics = []
    for _, g in df.groupby(date_col):
        if len(g) < 10:
            continue
        if g[pred_col].std() < 1e-12:
            continue
        ic = spearmanr(g[y_col], g[pred_col]).statistic
        if not np.isnan(ic):
            ics.append(ic)

    return float(np.mean(ics)) if ics else np.nan


def build_gam(n_features: int, n_splines: int) -> LinearGAM:
    """
    Build additive GAM with one spline term per feature.
    """
    terms = s(0, n_splines=n_splines)
    for j in range(1, n_features):
        terms += s(j, n_splines=n_splines)
    return LinearGAM(terms)


def average_lambda(gam: LinearGAM) -> float:
    """
    pygam stores lam as nested lists/arrays. Flatten and average.
    """
    vals = []

    def collect(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            for z in x:
                collect(z)
        else:
            try:
                vals.append(float(x))
            except Exception:
                pass

    collect(gam.lam)
    return float(np.mean(vals)) if vals else np.nan


def feature_edof(gam: LinearGAM, features: list[str]) -> pd.DataFrame:
    """
    Approximate feature-level effective degrees of freedom.

    pygam stores edof per coefficient. We sum edof_per_coef over each term's
    coefficient indices. The last term is usually the intercept, so we only
    loop over feature terms.
    """
    rows = []

    if "edof_per_coef" not in gam.statistics_:
        return pd.DataFrame()

    edof_per_coef = np.asarray(gam.statistics_["edof_per_coef"])

    for j, feat in enumerate(features):
        try:
            coef_idx = gam.terms.get_coef_indices(j)
            edof_j = float(np.sum(edof_per_coef[coef_idx]))
        except Exception:
            edof_j = np.nan

        rows.append({
            "feature": feat,
            "term_index": j,
            "edof": edof_j,
        })

    return pd.DataFrame(rows)


def partial_dependence_curves(
    gam: LinearGAM,
    scaler: StandardScaler,
    X_train_raw: np.ndarray,
    features: list[str],
    n_splines: int,
    fold_label: str,
):
    """
    Save partial dependence plots for selected features.

    We use standardized model space because the GAM is trained after StandardScaler.
    The x-axis is shown in standardized units.
    """
    plot_features = [f for f in FEATURES_TO_PLOT if f in features]

    if not plot_features:
        plot_features = features[:5]

    for feat in plot_features:
        j = features.index(feat)

        # Work in standardized feature space
        X_train_s = scaler.transform(np.nan_to_num(X_train_raw, nan=0.0))
        xj = X_train_s[:, j]

        grid_j = np.linspace(np.nanpercentile(xj, 1), np.nanpercentile(xj, 99), 100)

        # Baseline row = median standardized feature vector
        base = np.nanmedian(X_train_s, axis=0)
        X_grid = np.tile(base, (len(grid_j), 1))
        X_grid[:, j] = grid_j

        try:
            pdp = gam.partial_dependence(term=j, X=X_grid)
        except Exception as e:
            print(f"    [skip PDP] n={n_splines}, {feat}: {e}")
            continue

        plt.figure(figsize=(7, 4))
        plt.plot(grid_j, pdp)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.title(f"GAM partial dependence: {feat}\n"
                  f"n_splines={n_splines}, fold={fold_label}")
        plt.xlabel(f"{feat} standardized value")
        plt.ylabel("partial effect on predicted return")
        plt.tight_layout()

        safe_feat = feat.replace("/", "_").replace(" ", "_")
        out = OUTPUT / f"gam_debug_partial_dependence_n{n_splines}_{safe_feat}_{fold_label}.pdf"
        plt.savefig(out)
        plt.close()


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def run_debug():
    print("Loading panel...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    months = sorted(df[DATE_COL].unique())
    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS

    fold_metric_rows = []
    edof_rows = []

    # For partial dependence, use only a few representative folds
    # Last fold is useful because it has the most training data.
    representative_fold_indices = {
        start,
        start + 12,
        start + 36,
        len(months) - 1,
    }

    for n_splines in N_SPLINES_GRID:
        print("\n" + "=" * 80)
        print(f"Running GAM debug for n_splines={n_splines}")
        print("=" * 80)

        expanded_test_rows = []

        for i, test_month in enumerate(months[start:], start=start):
            train_end = months[i - DEFAULT_PURGE_MONTHS - 1]

            df_tr = df[df[DATE_COL] <= train_end].copy()
            df_te = df[df[DATE_COL] == test_month].copy()

            if len(df_te) < 10:
                continue

            X_tr_raw = df_tr[FEATURES].values.astype(float)
            y_tr = df_tr[TARGET_COL].values.astype(float)

            X_te_raw = df_te[FEATURES].values.astype(float)
            y_te = df_te[TARGET_COL].values.astype(float)

            X_tr_clean = np.nan_to_num(X_tr_raw, nan=0.0)
            X_te_clean = np.nan_to_num(X_te_raw, nan=0.0)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_clean)
            X_te_s = scaler.transform(X_te_clean)

            gam = build_gam(n_features=len(FEATURES), n_splines=n_splines)

            try:
                gam.gridsearch(
                    X_tr_s,
                    y_tr,
                    lam=LAM_GRID,
                    progress=False,
                )
            except Exception as e:
                print(f"  fold {i}, n={n_splines} failed: {e}")
                continue

            # Predictions
            y_tr_pred = gam.predict(X_tr_s)
            y_te_pred = gam.predict(X_te_s)

            # Train IC: average monthly IC on training months
            df_tr_pred = df_tr[[DATE_COL, STOCK_COL, TARGET_COL]].copy()
            df_tr_pred["y_pred"] = y_tr_pred
            train_ic = mean_monthly_ic(
                df_tr_pred,
                pred_col="y_pred",
                y_col=TARGET_COL,
                date_col=DATE_COL,
            )

            # Test IC: current test month Spearman
            test_ic = spearmanr(y_te, y_te_pred).statistic
            if np.isnan(test_ic):
                test_ic = np.nan

            # Total edof
            total_edof = float(gam.statistics_.get("edof", np.nan))

            # Lambda
            lam_avg = average_lambda(gam)

            fold_metric_rows.append({
                "n_splines": n_splines,
                "fold": i,
                "test_month": pd.to_datetime(test_month),
                "train_end": pd.to_datetime(train_end),
                "n_train": len(df_tr),
                "n_test": len(df_te),
                "lambda_avg": lam_avg,
                "total_edof": total_edof,
                "train_ic": train_ic,
                "test_ic": test_ic,
                "pred_std_test": float(np.std(y_te_pred)),
                "y_std_test": float(np.std(y_te)),
            })

            # Feature-level edof
            edof_df = feature_edof(gam, FEATURES)
            if len(edof_df):
                edof_df["n_splines"] = n_splines
                edof_df["fold"] = i
                edof_df["test_month"] = pd.to_datetime(test_month)
                edof_rows.append(edof_df)

            # Save expanded test rows for Sharpe
            tmp = pd.DataFrame({
                DATE_COL: df_te[DATE_COL].values,
                STOCK_COL: df_te[STOCK_COL].values,
                "y_true": y_te,
                "y_pred": y_te_pred,
                "n_splines": n_splines,
                "fold": i,
            })
            expanded_test_rows.append(tmp)

            # Partial dependence on representative folds
            if i in representative_fold_indices:
                fold_label = str(pd.to_datetime(test_month).date())[:7]
                print(f"  Saving PDPs for n={n_splines}, fold={i}, month={fold_label}")
                partial_dependence_curves(
                    gam=gam,
                    scaler=scaler,
                    X_train_raw=X_tr_raw,
                    features=FEATURES,
                    n_splines=n_splines,
                    fold_label=fold_label,
                )

            if i % 12 == 0:
                print(
                    f"  fold {i:3d} | {str(test_month)[:7]} | "
                    f"lambda={lam_avg:.4g} | edof={total_edof:.1f} | "
                    f"train IC={train_ic:+.4f} | test IC={test_ic:+.4f}"
                )

        # Per-n_splines Sharpe summary
        if expanded_test_rows:
            expanded = pd.concat(expanded_test_rows, ignore_index=True)
            expanded.to_parquet(
                OUTPUT / f"gam_debug_expanded_n{n_splines}.parquet",
                index=False,
            )

    # Save fold metrics
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(OUTPUT / "gam_debug_fold_metrics.csv", index=False)

    # Save edof
    if edof_rows:
        edof_all = pd.concat(edof_rows, ignore_index=True)
        edof_all.to_csv(OUTPUT / "gam_debug_feature_edof.csv", index=False)
    else:
        edof_all = pd.DataFrame()

    # Build summary by n_splines
    summary_rows = []

    for n_splines in N_SPLINES_GRID:
        sub = fold_metrics[fold_metrics["n_splines"] == n_splines].copy()

        expanded_path = OUTPUT / f"gam_debug_expanded_n{n_splines}.parquet"
        if expanded_path.exists():
            expanded = pd.read_parquet(expanded_path)
            sharpe = compute_long_short_sharpe(expanded)
        else:
            sharpe = np.nan

        summary_rows.append({
            "n_splines": n_splines,
            "n_folds": len(sub),
            "train_ic_mean": sub["train_ic"].mean(),
            "test_ic_mean": sub["test_ic"].mean(),
            "test_ic_std": sub["test_ic"].std(),
            "ic_ir": compute_ic_ir(sub["test_ic"].dropna()),
            "hit_rate": compute_hit_rate(sub["test_ic"].dropna()),
            "ls_sharpe": sharpe,
            "lambda_avg_mean": sub["lambda_avg"].mean(),
            "lambda_avg_median": sub["lambda_avg"].median(),
            "total_edof_mean": sub["total_edof"].mean(),
            "total_edof_median": sub["total_edof"].median(),
            "pred_std_test_mean": sub["pred_std_test"].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "gam_debug_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("GAM DEBUG SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nWrote:")
    print(f"  {OUTPUT / 'gam_debug_fold_metrics.csv'}")
    print(f"  {OUTPUT / 'gam_debug_feature_edof.csv'}")
    print(f"  {OUTPUT / 'gam_debug_summary.csv'}")
    print(f"  {OUTPUT / 'gam_debug_partial_dependence_*.pdf'}")


if __name__ == "__main__":
    run_debug()