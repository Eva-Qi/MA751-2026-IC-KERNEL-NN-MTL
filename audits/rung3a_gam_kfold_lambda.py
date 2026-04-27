"""
Rung 3a GAM — K-fold λ selection (TimeSeriesSplit inner CV).

Replaces pygam.gridsearch (single-shot GCV) with a 5-fold month-aware
inner TimeSeriesSplit to pick λ per outer walk-forward fold.

Key settings
------------
  n_splines = 5          (sweet spot from rung3a_gam_audit.py)
  λ grid    = logspace(-3, 3, 11)
  Inner CV  = K=5, gap_months=1, month-boundary splits
              (same _month_aware_cv pattern as run_rung12_v2.py)
  Outer     = expanding window, min 60 months train, 1-month purge, 58 test months
  Scoring   = Spearman IC (matches evaluation metric)

Outputs
-------
  output/rung3a_gam_kfold_lambda_summary.csv   — aggregate comparison row
  output/rung3a_gam_kfold_lambda_diag.csv      — per-fold: chosen λ + inner CV IC
  output/rung3a_kfold_lambda_log.txt           — stdout tee target
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

# ── Hyperparameters ──────────────────────────────────────────────────────────
FEATURES   = ALL_FEATURE_COLS_V3_WITH_MISS
N_SPLINES  = 5                           # sweet spot from audit
LAM_GRID   = np.logspace(-3, 3, 11)     # 11 candidates: 0.001 … 1000
K_INNER    = 5                           # inner CV folds
GAP_MONTHS = 1                           # 1-month gap between inner train/val

DATA   = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)


# ── Month-aware inner CV ─────────────────────────────────────────────────────
def _month_aware_cv(train_dates: np.ndarray, n_splits: int = 5, gap_months: int = 1):
    """Return list of (train_idx, val_idx) with month-aligned boundaries.

    Identical pattern to run_rung12_v2.py._month_aware_cv.
    Splits distinct months into (n_splits+1) equal segments.
    Fold k: train on segments [0..k], val on segment [k+1] (after gap_months).
    """
    unique_months = np.array(sorted(np.unique(train_dates)))
    n_months = len(unique_months)
    seg = max(1, n_months // (n_splits + 1))
    splits = []
    for k in range(n_splits):
        train_end_idx = (k + 1) * seg - 1
        val_start_idx = train_end_idx + 1 + gap_months
        val_end_idx   = min(val_start_idx + seg - 1, n_months - 1)
        if val_start_idx > val_end_idx or val_start_idx >= n_months:
            continue
        train_end_month = unique_months[train_end_idx]
        val_start_month = unique_months[val_start_idx]
        val_end_month   = unique_months[val_end_idx]
        train_idx = np.where(train_dates <= train_end_month)[0]
        val_idx   = np.where(
            (train_dates >= val_start_month) & (train_dates <= val_end_month)
        )[0]
        if len(train_idx) < 30 or len(val_idx) < 10:
            continue
        splits.append((train_idx, val_idx))
    return splits if splits else None


def _build_gam(n_features: int, n_splines: int, lam: float) -> LinearGAM:
    """Construct LinearGAM with fixed λ across all terms."""
    terms = s(0, n_splines=n_splines)
    for j in range(1, n_features):
        terms += s(j, n_splines=n_splines)
    gam = LinearGAM(terms)
    # Set lam as a list of length n_features (pygam expects per-term lam)
    gam.set_params(lam=[lam] * n_features)
    return gam


def _inner_cv_ic(
    X_tr_s: np.ndarray,
    y_tr: np.ndarray,
    train_dates: np.ndarray,
    lam: float,
    n_features: int,
) -> float:
    """Score λ via inner K-fold Spearman IC. Returns mean IC across folds."""
    splits = _month_aware_cv(train_dates, n_splits=K_INNER, gap_months=GAP_MONTHS)
    if splits is None:
        return np.nan

    fold_ics = []
    for tr_idx, val_idx in splits:
        X_inner_tr = X_tr_s[tr_idx]
        y_inner_tr = y_tr[tr_idx]
        X_inner_val = X_tr_s[val_idx]
        y_inner_val = y_tr[val_idx]

        try:
            gam = _build_gam(n_features, N_SPLINES, lam)
            gam.fit(X_inner_tr, y_inner_tr)
            y_val_pred = gam.predict(X_inner_val)

            # Compute Spearman IC by month in the val window
            val_dates = train_dates[val_idx]
            unique_val_months = np.unique(val_dates)
            month_ics = []
            for m in unique_val_months:
                mask = val_dates == m
                if mask.sum() < 5:
                    continue
                ic_m = float(spearmanr(y_inner_val[mask], y_val_pred[mask]).statistic)
                if not np.isnan(ic_m):
                    month_ics.append(ic_m)

            if month_ics:
                fold_ics.append(float(np.mean(month_ics)))
        except Exception:
            continue

    return float(np.nanmean(fold_ics)) if fold_ics else np.nan


# ── Main walk-forward ────────────────────────────────────────────────────────
def run_kfold_lambda(df: pd.DataFrame):
    """Full walk-forward with inner K-fold λ selection per outer fold."""
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS

    n_features = len(FEATURES)
    fold_records = []
    ic_list      = []
    diag_rows    = []

    total_folds = len(months) - start
    t_run_start = time.time()

    for fold_idx, i in enumerate(range(start, len(months))):
        test_month = months[i]
        train_end  = months[i - DEFAULT_PURGE_MONTHS - 1]

        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]

        if len(df_te) < 10:
            continue

        # Features — NaN → 0
        X_tr = np.nan_to_num(df_tr[FEATURES].values.astype(np.float64), nan=0.0)
        y_tr = df_tr[TARGET_COL].values.astype(np.float64)
        X_te = np.nan_to_num(df_te[FEATURES].values.astype(np.float64), nan=0.0)
        y_te = df_te[TARGET_COL].values.astype(np.float64)

        # StandardScaler — fit on outer train ONLY
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        train_dates = df_tr[DATE_COL].values

        # ── Inner CV: pick best λ ────────────────────────────────────────
        lam_scores = []
        for lam in LAM_GRID:
            score = _inner_cv_ic(X_tr_s, y_tr, train_dates, lam, n_features)
            lam_scores.append(score)

        lam_scores = np.array(lam_scores)
        valid_mask = ~np.isnan(lam_scores)

        if valid_mask.any():
            best_lam = float(LAM_GRID[np.nanargmax(lam_scores)])
            inner_cv_ic_mean = float(np.nanmax(lam_scores))
        else:
            # Fallback: use λ=1.0 (neutral midpoint of grid)
            best_lam = 1.0
            inner_cv_ic_mean = np.nan

        # ── Refit on full outer train with chosen λ ──────────────────────
        gam = _build_gam(n_features, N_SPLINES, best_lam)
        gam.fit(X_tr_s, y_tr)

        y_pred = gam.predict(X_te_s)
        ic = float(spearmanr(y_te, y_pred).statistic)
        ic_list.append(ic)

        fold_records.append({
            DATE_COL:  test_month,
            STOCK_COL: df_te[STOCK_COL].values.tolist(),
            "y_true":  y_te.tolist(),
            "y_pred":  y_pred.tolist(),
            "ic":      ic,
        })

        diag_rows.append({
            "fold":              fold_idx + 1,
            "test_month":        str(test_month)[:7],
            "chosen_lambda":     best_lam,
            "inner_cv_ic_mean":  round(inner_cv_ic_mean, 6) if not np.isnan(inner_cv_ic_mean) else np.nan,
        })

        elapsed = time.time() - t_run_start
        print(
            f"  fold {fold_idx+1:3d}/{total_folds} | "
            f"{str(test_month)[:7]} | n_train={len(df_tr):6,d} | "
            f"n_test={len(df_te):4d} | "
            f"λ={best_lam:.4g} | IC={ic:+.4f} | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    # ── Build expanded frame for Sharpe ──────────────────────────────────
    rows = []
    for r in fold_records:
        for t, yp, yt in zip(r[STOCK_COL], r["y_pred"], r["y_true"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t,
                         "y_pred": yp, "y_true": yt})
    expanded = pd.DataFrame(rows)

    ic_series = pd.Series(ic_list, dtype=float).dropna()
    sharpe    = compute_long_short_sharpe(expanded) if len(expanded) else float("nan")

    summary = {
        "model":     "GAM_kfold_lambda",
        "n_splines": N_SPLINES,
        "IC_mean":   round(float(ic_series.mean()), 4) if len(ic_series) else float("nan"),
        "IC_std":    round(float(ic_series.std(ddof=1)), 4) if len(ic_series) > 1 else float("nan"),
        "IC_IR":     round(compute_ic_ir(ic_series), 4),
        "Hit_Rate":  round(compute_hit_rate(ic_series), 3),
        "LS_Sharpe": round(sharpe, 3),
        "n_months":  len(ic_series),
    }

    diag_df = pd.DataFrame(diag_rows)
    return summary, diag_df, ic_series


# ── Sanity check ─────────────────────────────────────────────────────────────
def sanity_check(summary: dict):
    flags = []
    ic = summary["IC_mean"]
    sharpe = summary["LS_Sharpe"]
    if not np.isnan(ic) and not (0.005 <= ic <= 0.025):
        flags.append(
            f"IC={ic:.4f} outside expected [+0.005, +0.025] range — "
            "borderline OK if within [-0.02, +0.05], else BUG"
        )
    if not np.isnan(sharpe) and not (0.4 <= sharpe <= 1.0):
        flags.append(
            f"Sharpe={sharpe:.3f} outside expected [+0.4, +1.0] range — "
            "borderline OK if within [-0.5, +1.5], else BUG"
        )
    return flags


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("RUNG 3A GAM — K-fold λ selection (inner TimeSeriesSplit CV)")
    print(f"  n_splines = {N_SPLINES}  |  K_inner = {K_INNER}  |  gap = {GAP_MONTHS} month")
    print(f"  λ grid: {LAM_GRID[0]:.1e} … {LAM_GRID[-1]:.1e}  ({len(LAM_GRID)} candidates)")
    print("=" * 80, flush=True)

    print(f"\nLoading panel: {DATA}")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(
        f"  {len(df):,} rows | {df[DATE_COL].nunique()} months | "
        f"{df[STOCK_COL].nunique()} tickers | {len(FEATURES)} features",
        flush=True,
    )

    t0 = time.time()
    summary, diag_df, ic_series = run_kfold_lambda(df)
    elapsed = time.time() - t0
    summary["wall_sec"] = round(elapsed, 1)

    flags = sanity_check(summary)
    summary["flags"] = " | ".join(flags) if flags else "OK"

    # ── Save outputs ──────────────────────────────────────────────────────
    summary_path = OUTPUT / "rung3a_gam_kfold_lambda_summary.csv"
    diag_path    = OUTPUT / "rung3a_gam_kfold_lambda_diag.csv"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved diagnostics -> {diag_path}")

    # ── Comparison table ──────────────────────────────────────────────────
    # Baseline numbers from rung3a_gam_audit.py (n_splines=5, gridsearch GCV)
    baseline = {
        "model":     "GAM_gridsearch_GCV",
        "n_splines": 5,
        "IC_mean":   0.0146,
        "IC_std":    float("nan"),
        "IC_IR":     float("nan"),
        "Hit_Rate":  float("nan"),
        "LS_Sharpe": 0.792,
        "n_months":  58,
    }

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    fmt = "{:<28s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}"
    print(fmt.format("Model", "IC_mean", "IC_std", "IC_IR", "Hit_Rate", "LS_Sharpe", "n_months"))
    print("-" * 80)

    def _fmt_row(s):
        ic_std   = f"{s['IC_std']:.4f}"   if not (isinstance(s['IC_std'],   float) and np.isnan(s['IC_std']))   else "   N/A"
        ic_ir    = f"{s['IC_IR']:.4f}"    if not (isinstance(s['IC_IR'],    float) and np.isnan(s['IC_IR']))    else "   N/A"
        hit_rate = f"{s['Hit_Rate']:.3f}" if not (isinstance(s['Hit_Rate'], float) and np.isnan(s['Hit_Rate'])) else "   N/A"
        print(fmt.format(
            s["model"],
            f"{s['IC_mean']:+.4f}",
            ic_std,
            ic_ir,
            hit_rate,
            f"{s['LS_Sharpe']:.3f}",
            str(s["n_months"]),
        ))

    _fmt_row(baseline)
    _fmt_row(summary)

    # ── λ distribution diagnostics ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("λ SELECTION DIAGNOSTICS (across outer folds)")
    print("=" * 80)
    lam_vals = diag_df["chosen_lambda"].values
    print(f"  mean   = {lam_vals.mean():.4g}")
    print(f"  median = {np.median(lam_vals):.4g}")
    print(f"  std    = {lam_vals.std():.4g}")
    print(f"  min    = {lam_vals.min():.4g}")
    print(f"  max    = {lam_vals.max():.4g}")

    # Frequency count per λ value
    lam_counts = pd.Series(lam_vals).value_counts().sort_index()
    print("\n  λ value   | count | fraction")
    print("  ----------|-------|----------")
    for lv, cnt in lam_counts.items():
        bar = "#" * int(cnt / max(lam_counts) * 20)
        edge = ""
        if lv == LAM_GRID[0]:
            edge = " ← LEFT EDGE (under-smoothed?)"
        elif lv == LAM_GRID[-1]:
            edge = " ← RIGHT EDGE (over-smoothed?)"
        print(f"  {lv:9.4g} | {cnt:5d} | {cnt/len(lam_vals):6.1%}  {bar}{edge}")

    edge_frac = (
        (lam_vals == LAM_GRID[0]).sum() + (lam_vals == LAM_GRID[-1]).sum()
    ) / len(lam_vals)
    if edge_frac > 0.3:
        print(f"\n  *** WARNING: {edge_frac:.0%} of folds hit grid edge — consider widening LAM_GRID ***")
    else:
        print(f"\n  Grid coverage OK ({edge_frac:.0%} at edges)")

    # ── Verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    delta_ic = summary["IC_mean"] - baseline["IC_mean"]
    print(f"  K-fold λ IC_mean:    {summary['IC_mean']:+.4f}")
    print(f"  GCV baseline IC:     {baseline['IC_mean']:+.4f}")
    print(f"  ΔIC (kfold - GCV):   {delta_ic:+.4f}")

    if abs(delta_ic) < 0.003:
        print("  IC delta within noise range — K-fold λ did NOT materially change IC.")
    elif delta_ic > 0:
        print("  K-fold λ IMPROVED IC.")
    else:
        print("  K-fold λ HURT IC.")

    if not np.isnan(summary["IC_IR"]):
        print(f"\n  IC_IR (K-fold):  {summary['IC_IR']:.4f}")
        print("  (Baseline IC_IR not available — run rung3a_gam_audit.py with IC_IR output)")
        if summary["IC_IR"] > 0.4:
            print("  IC_IR looks reasonable — consistent month-to-month signal.")
        else:
            print("  IC_IR below 0.4 — signal consistency is low.")

    delta_sharpe = summary["LS_Sharpe"] - baseline["LS_Sharpe"]
    print(f"\n  ΔSharpe (kfold - GCV): {delta_sharpe:+.3f}")
    if delta_sharpe > 0.05:
        print("  Sharpe improved — better λ selection OR noise.")
    elif delta_sharpe < -0.05:
        print("  Sharpe degraded — K-fold λ may have over-tuned on IC rather than tail ordering.")
    else:
        print("  Sharpe roughly equivalent.")

    if flags:
        print("\n  SANITY FLAGS:")
        for f in flags:
            print(f"  *** {f} ***")
    else:
        print("\n  All sanity checks PASSED.")

    print("=" * 80)
    print(f"Outputs: {summary_path}")
    print(f"         {diag_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
