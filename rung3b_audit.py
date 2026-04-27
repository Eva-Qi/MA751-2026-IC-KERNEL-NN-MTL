"""
Rung 3b Audit — 3-stage diagnostic per teammate's PDE audit meta-learnings.

STAGE 1 — Shape diagnostic on current Quantile config
    Rule 2/6: per-fold pred_std / pred_mean / feature importance.
    Checks if -0.0029 IC is real negative signal or pure noise
    around a collapsed prediction distribution.

STAGE 2 — Sensitivity grid (MSE, 3×3 depth × lr)
    Rule 3: if IC sign reverses across reasonable hyperparameter configs,
    solver is unstable in this noise regime — individual results are noise.
    If IC is monotone / same-sign, thesis "GBM can't extract signal" is earned.

STAGE 3 — Gu-Kelly-Xiu 2020 RFS tuned variant
    Rule 5/7: Gu, Kelly, Xiu "Empirical Asset Pricing via ML" found
    gradient boosting DOES work on similar cross-sectional equity panels
    with heavy regularization: max_depth=1 (stumps), n_est=1000, lr=0.01.
    Our default (depth=4, n=200, lr=0.05) is generic ML, not finance-tuned.
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
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V3_WITH_MISS, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from metrics import compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

FEATURES = ALL_FEATURE_COLS_V3_WITH_MISS
DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)


def run_xgb_config(df, objective, n_estimators, max_depth, learning_rate,
                   label, collect_shape=False):
    """Walk-forward with one XGB config. Returns (summary dict, shape records list)."""
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []
    shape_records = []

    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS

    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    if objective == "reg:quantileerror":
        kwargs["quantile_alpha"] = 0.5

    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]
        if len(df_te) < 10:
            continue

        X_tr = np.nan_to_num(df_tr[FEATURES].values, nan=0.0)
        y_tr = df_tr[TARGET_COL].values
        X_te = np.nan_to_num(df_te[FEATURES].values, nan=0.0)
        y_te = df_te[TARGET_COL].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = XGBRegressor(objective=objective, **kwargs)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        ic = spearmanr(y_te, y_pred).statistic

        if collect_shape:
            imp = model.feature_importances_
            top5_idx = np.argsort(imp)[-5:][::-1]
            shape_records.append({
                "fold": i,
                "month": str(test_month)[:7],
                "pred_std": float(np.std(y_pred)),
                "pred_mean": float(np.mean(y_pred)),
                "y_te_std": float(np.std(y_te)),
                "ic": float(ic),
                "top1_feat_name": FEATURES[top5_idx[0]],
                "top1_feat_imp": float(imp[top5_idx[0]]),
                "top5_feat_names": [FEATURES[k] for k in top5_idx],
                "top5_feat_imps": [float(imp[k]) for k in top5_idx],
            })

        results.append({
            DATE_COL: test_month,
            "IC": ic,
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
        })

    monthly = pd.DataFrame(results)
    ic_series = monthly["IC"].dropna()

    rows = []
    for _, r in monthly.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t, "y_pred": yp, "y_true": yt})
    expanded = pd.DataFrame(rows)
    sharpe = compute_long_short_sharpe(expanded)

    return {
        "label": label,
        "objective": objective,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "IC_mean": ic_series.mean(),
        "IC_std": ic_series.std(),
        "IC_IR": compute_ic_ir(ic_series),
        "Hit_Rate": compute_hit_rate(ic_series),
        "LS_Sharpe": sharpe,
        "n_months": len(ic_series),
    }, shape_records


def main():
    print("=" * 80)
    print("RUNG 3B AUDIT — Sensitivity + GKX + Shape Diagnostic")
    print("=" * 80, flush=True)

    print("Loading V3 panel...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, {len(FEATURES)} features", flush=True)

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Quantile shape diagnostic on current config
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 1: Quantile XGB shape diagnostic (current config d=4, lr=0.05, n=200)")
    print("=" * 80, flush=True)
    t0 = time.time()
    s1, shape = run_xgb_config(
        df, objective="reg:quantileerror",
        n_estimators=200, max_depth=4, learning_rate=0.05,
        label="quantile_current", collect_shape=True,
    )
    print(f"  [done in {time.time()-t0:.1f}s]  IC={s1['IC_mean']:+.4f}, "
          f"Sharpe={s1['LS_Sharpe']:.3f}", flush=True)

    shape_df = pd.DataFrame(shape)
    print(f"\n  Per-fold diagnostic (n={len(shape_df)}):")
    print(f"    pred_std:  mean={shape_df['pred_std'].mean():.5f}, "
          f"min={shape_df['pred_std'].min():.5f}, max={shape_df['pred_std'].max():.5f}")
    print(f"    pred_mean: mean={shape_df['pred_mean'].mean():+.5f}, "
          f"min={shape_df['pred_mean'].min():+.5f}, max={shape_df['pred_mean'].max():+.5f}")
    print(f"    y_te_std:  mean={shape_df['y_te_std'].mean():.5f}  (for scale reference)")

    ratio = shape_df['pred_std'].mean() / shape_df['y_te_std'].mean()
    print(f"    [SHAPE] pred_std / y_te_std = {ratio:.4f}")
    if ratio < 0.05:
        print("    🚨 pred_std is <5% of y_te_std — predictions nearly COLLAPSED to constant")
        print("       Negative IC is likely noise in a flat predictor, not real anti-signal")

    n_collapse = (shape_df['pred_std'] < 0.001).sum()
    print(f"    folds with pred_std<0.001: {n_collapse}/{len(shape_df)}")

    top1 = shape_df['top1_feat_name'].value_counts()
    print(f"\n  Top-1 feature across {len(shape_df)} folds:")
    for name, cnt in top1.head(5).items():
        print(f"    {name}: {cnt} folds")

    shape_df.to_csv(OUTPUT / "rung3b_quantile_shape_diag.csv", index=False)

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Sensitivity grid
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 2: Sensitivity grid — MSE objective, depth ∈ {1,4,6} × lr ∈ {0.01,0.05,0.1}")
    print("  n_estimators=300 fixed. 9 configs.")
    print("=" * 80, flush=True)

    grid_results = []
    depths = [1, 4, 6]
    lrs = [0.01, 0.05, 0.1]
    n_est = 300

    for depth in depths:
        for lr in lrs:
            label = f"mse_d{depth}_lr{lr}_n{n_est}"
            t0 = time.time()
            s, _ = run_xgb_config(
                df, objective="reg:squarederror",
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                label=label,
            )
            elapsed = time.time() - t0
            print(f"  d={depth} lr={lr}  [{elapsed:.1f}s]  "
                  f"IC={s['IC_mean']:+.4f}, IC_IR={s['IC_IR']:.3f}, "
                  f"Sharpe={s['LS_Sharpe']:.3f}, Hit={s['Hit_Rate']:.1%}", flush=True)
            grid_results.append(s)

    grid_df = pd.DataFrame(grid_results)
    print("\n  === Sensitivity grid summary ===")
    print(grid_df[['max_depth', 'learning_rate', 'n_estimators',
                   'IC_mean', 'IC_IR', 'LS_Sharpe']].to_string(index=False))

    ic_vals = grid_df['IC_mean'].values
    n_pos = (ic_vals > 0).sum()
    n_neg = (ic_vals < 0).sum()
    ic_range = ic_vals.max() - ic_vals.min()
    print(f"\n  [SIGN CHECK] positive IC: {n_pos}/{len(grid_df)}, "
          f"negative IC: {n_neg}/{len(grid_df)}")
    print(f"  [RANGE]      IC spans {ic_vals.min():+.4f} to {ic_vals.max():+.4f} "
          f"(Δ={ic_range:.4f})")
    if n_pos > 0 and n_neg > 0:
        print("  🚨 SIGN REVERSAL — solver unstable. -0.0029 Quantile / +0.0018 MSE likely noise.")
    elif ic_range > 0.01:
        print("  ⚠️  Large spread — IC is sensitive to hyperparams, individual result unreliable")
    else:
        print("  ✅ Consistent sign + small range — 'no signal at this complexity' thesis earned")

    grid_df.to_csv(OUTPUT / "rung3b_sensitivity_grid.csv", index=False)

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Gu-Kelly-Xiu 2020 RFS tuned variant
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 3: Gu-Kelly-Xiu 2020 RFS tuned (depth=1 stumps, n=1000, lr=0.01)")
    print("=" * 80, flush=True)
    t0 = time.time()
    gkx, _ = run_xgb_config(
        df, objective="reg:squarederror",
        n_estimators=1000, max_depth=1, learning_rate=0.01,
        label="gkx_tuned",
    )
    print(f"  [done in {time.time()-t0:.1f}s]")
    print(f"  GKX: IC={gkx['IC_mean']:+.4f}, IC_IR={gkx['IC_IR']:.3f}, "
          f"Sharpe={gkx['LS_Sharpe']:.3f}, Hit={gkx['Hit_Rate']:.1%}", flush=True)

    # Compare to our current default
    print("\n  === GKX vs current default ===")
    default_row = grid_df[(grid_df['max_depth'] == 4) &
                          (grid_df['learning_rate'] == 0.05)]
    if len(default_row):
        d = default_row.iloc[0]
        d_ic = gkx['IC_mean'] - d['IC_mean']
        d_sharpe = gkx['LS_Sharpe'] - d['LS_Sharpe']
        print(f"    Current (d=4, lr=0.05, n=300): IC={d['IC_mean']:+.4f}, Sharpe={d['LS_Sharpe']:.3f}")
        print(f"    GKX     (d=1, lr=0.01, n=1000): IC={gkx['IC_mean']:+.4f}, Sharpe={gkx['LS_Sharpe']:.3f}")
        print(f"    Δ:                              ΔIC={d_ic:+.4f}, ΔSharpe={d_sharpe:+.3f}")
        if gkx['IC_mean'] > 0.01:
            print("  🔔 GKX tuned IC > 0.01 — hyperparameter tuning MATTERS, retract 'GBM fails' claim")
        elif abs(d_ic) > 0.005:
            print("  ⚠️  Material difference but still weak — tuning helps but won't overturn thesis")
        else:
            print("  ✓ Minimal improvement — even finance-tuned GBM can't extract signal here")

    pd.DataFrame([gkx]).to_csv(OUTPUT / "rung3b_gkx_tuned.csv", index=False)

    # ─────────────────────────────────────────────────────────────
    # Final verdict
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"  Stage 1 (shape):       pred_std / y_te_std = {ratio:.4f}")
    print(f"  Stage 2 (sensitivity): sign reversal = {'YES' if (n_pos>0 and n_neg>0) else 'NO'}, "
          f"IC range = {ic_range:.4f}")
    print(f"  Stage 3 (GKX tuned):   IC = {gkx['IC_mean']:+.4f}, Sharpe = {gkx['LS_Sharpe']:.3f}")

    print("\nAll outputs:")
    print("  output/rung3b_quantile_shape_diag.csv")
    print("  output/rung3b_sensitivity_grid.csv")
    print("  output/rung3b_gkx_tuned.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
