"""
Rung 1-2 on V2 data: OLS, IC-Ensemble, LASSO, Ridge, ElasticNet,
plus new: Fama-MacBeth (1c), Barra-adjusted IC-Ensemble (1d), Adaptive LASSO (2e).

Uses the V2 panel master_panel_v2.parquet (WITH_MISS flags included).

Audit fixes applied 2026-04-24:
  - cv=5 → TimeSeriesSplit(5, gap=1) everywhere
  - LASSO α grid: logspace(-4, 1, 50) [reverted from logspace(-4, 2, 80)
    because extended ceiling caused α-plateau selection on winsorized + low-p V2]
  - ElasticNet l1_ratio refined [0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99]
  - Ridge scoring → Spearman IC (align with evaluation metric)
  - Missingness indicators wired into feature set
  - REVERTED: global y_tr winsorize (was asymmetric w/ X per-month winsorize,
    broke monotonic tail ordering needed for IC metric)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.covariance import LedoitWolf

from config import (
    ALL_FEATURE_COLS_V2_WITH_MISS, TARGET_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
ALL_FEATURE_COLS_V2 = ALL_FEATURE_COLS_V2_WITH_MISS  # audit fix: include missingness indicators
from metrics import compute_monthly_ic, compute_long_short_sharpe, compute_ic_ir, compute_hit_rate

DATA = Path("data/master_panel_v2.parquet")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

IC_WEIGHT_WINDOW = 12
LASSO_ALPHAS = np.logspace(-4, 1, 50)  # audit 2026-04-24: reverted from (-4,2,80) after α-plateau problem
RIDGE_ALPHAS = np.logspace(-4, 4, 50)

# Spearman scorer for Ridge CV (loss-metric alignment)
def _spearman_scorer(y_true, y_pred):
    r = spearmanr(y_true, y_pred).statistic
    return 0.0 if np.isnan(r) else r

SPEARMAN_SCORER = make_scorer(_spearman_scorer)


# Month-aware CV splitter (audit 2026-04-24)
# Replaces sklearn TimeSeriesSplit(gap=1) which gaps by ROW (= 0 months in stacked panel).
# This splits on full month boundaries with a real month-level gap.
def _month_aware_cv(train_dates, n_splits=5, gap_months=1):
    """Return list of (train_idx, val_idx) tuples with month-aligned boundaries.

    Splits the training window's distinct months into (n_splits + 1) segments.
    Fold k: train on segments [0..k], validation on segment [k+1] (after gap_months).
    """
    if train_dates is None:
        return None
    unique_months = np.array(sorted(np.unique(train_dates)))
    n_months = len(unique_months)
    seg = max(1, n_months // (n_splits + 1))
    splits = []
    for k in range(n_splits):
        train_end_idx = (k + 1) * seg - 1
        val_start_idx = train_end_idx + 1 + gap_months
        val_end_idx = min(val_start_idx + seg - 1, n_months - 1)
        if val_start_idx > val_end_idx or val_start_idx >= n_months:
            continue
        train_end_month = unique_months[train_end_idx]
        val_start_month = unique_months[val_start_idx]
        val_end_month = unique_months[val_end_idx]
        train_idx = np.where(train_dates <= train_end_month)[0]
        val_idx = np.where((train_dates >= val_start_month) & (train_dates <= val_end_month))[0]
        if len(train_idx) < 30 or len(val_idx) < 10:
            continue
        splits.append((train_idx, val_idx))
    return splits if splits else None


def _cv_for(train_dates):
    """Return month-aware splits if dates known; fallback to TimeSeriesSplit row-based."""
    m = _month_aware_cv(train_dates, n_splits=5, gap_months=1)
    return m if m is not None else TimeSeriesSplit(n_splits=5, gap=1)


def run_walk_forward(df, model_fn, label, features=None):
    """Generic walk-forward for sklearn-style models.

    Passes train_dates to model_fn via kwargs so models needing per-month
    grouping (e.g. Fama-MacBeth) can access them.
    """
    if features is None:
        features = ALL_FEATURE_COLS_V2

    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []

    start = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS
    for i, test_month in enumerate(months[start:], start=start):
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]

        if len(df_te) < 10:
            continue

        X_tr = np.nan_to_num(df_tr[features].values, nan=0.0)
        y_tr = df_tr[TARGET_COL].values  # raw forward return (NOT winsorized)
        # Rationale (audit 2026-04-24 reversal): X is already per-month winsorized
        # at panel-construction time (_cs_zscore_winsorized in load_data.py). Adding
        # a GLOBAL y winsorize here was asymmetric AND broke monotonic tail ordering
        # (tied ranks at boundary → breaks IC signal at tails). Canonical quant
        # practice: winsorize features X, NOT target y. Fat-tail defense for models
        # that need it lives in the loss function (Huber in MLP/MTL) or model
        # regularization (L2 in Ridge), not at the target level.
        X_te = np.nan_to_num(df_te[features].values, nan=0.0)
        y_te = df_te[TARGET_COL].values
        train_dates = df_tr[DATE_COL].values

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        y_pred = model_fn(X_tr_s, y_tr, X_te_s, features, train_dates=train_dates)

        ic = spearmanr(y_te, y_pred).statistic
        results.append({
            DATE_COL: test_month,
            "fold": i,
            "IC": ic,
            "n_test": len(df_te),
            "y_pred_list": y_pred.tolist(),
            "y_true_list": y_te.tolist(),
            "tickers": df_te[STOCK_COL].values.tolist(),
        })

        if i % 12 == 0:
            print(f"  {label} fold {i} | {str(test_month)[:7]} | IC={ic:+.4f}")

    return pd.DataFrame(results)


def ols_model(X_tr, y_tr, X_te, features, **kwargs):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _recent_window_by_month(X_tr, y_tr, train_dates, n_months=IC_WEIGHT_WINDOW):
    """Return last n_months of data based on DATE, not row-count proxy (audit 2026-04-24)."""
    if train_dates is None:
        # Fallback to row proxy when dates not passed
        n_recent = min(len(X_tr), n_months * 400)
        return X_tr[-n_recent:], y_tr[-n_recent:]
    unique_months = np.sort(np.unique(train_dates))
    if len(unique_months) <= n_months:
        return X_tr, y_tr
    cutoff = unique_months[-n_months]
    mask = train_dates >= cutoff
    return X_tr[mask], y_tr[mask]


def ic_ensemble_model(X_tr, y_tr, X_te, features, train_dates=None, **kwargs):
    """IC-weighted univariate ensemble. Naive Σ|IC| weighting over last 12 months."""
    n_features = X_tr.shape[1]
    X_recent, y_recent = _recent_window_by_month(X_tr, y_tr, train_dates, n_months=IC_WEIGHT_WINDOW)

    ics = []
    preds_per_feature = []
    for j in range(n_features):
        model = LinearRegression()
        model.fit(X_tr[:, j:j+1], y_tr)
        preds_per_feature.append(model.predict(X_te[:, j:j+1]))
        ic_j = abs(np.corrcoef(X_recent[:, j], y_recent)[0, 1])
        ics.append(ic_j if not np.isnan(ic_j) else 0.0)

    ics = np.array(ics)
    total = ics.sum()
    weights = np.ones(n_features) / n_features if total < 1e-8 else ics / total
    return sum(w * p for w, p in zip(weights, preds_per_feature))


def lasso_model(X_tr, y_tr, X_te, features, train_dates=None, **kwargs):
    """[CAVEAT-unreliable_SNR] LASSO standalone unreliable at IC~0.01.
    Friend's critique: L1 often zeroes out; preserved for ensemble feature-selection use.
    """
    model = LassoCV(alphas=LASSO_ALPHAS, cv=_cv_for(train_dates), max_iter=10000)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if np.std(pred) < 1e-10:
        fallback = LinearRegression()
        fallback.fit(X_tr, y_tr)
        pred = fallback.predict(X_te)
    return pred


def ridge_model(X_tr, y_tr, X_te, features, train_dates=None, **kwargs):
    """Ridge with Spearman-IC scoring + month-aware CV."""
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=_cv_for(train_dates), scoring=SPEARMAN_SCORER)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


_ELASTIC_NET_DIAG = []


def elastic_net_model(X_tr, y_tr, X_te, features, train_dates=None, **kwargs):
    """[CAVEAT-unreliable_SNR] ElasticNet. V3 empirical l1_ratio ≈ 0.07 → basically Ridge.

    Month-aware CV + refined l1_ratio grid.
    """
    model = ElasticNetCV(
        alphas=LASSO_ALPHAS,
        l1_ratio=[0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99],
        cv=_cv_for(train_dates),
        max_iter=10000,
    )
    model.fit(X_tr, y_tr)
    _ELASTIC_NET_DIAG.append({
        "l1_ratio": float(model.l1_ratio_),
        "alpha": float(model.alpha_),
        "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-10)),
    })
    pred = model.predict(X_te)
    if np.std(pred) < 1e-10:
        fallback = LinearRegression()
        fallback.fit(X_tr, y_tr)
        pred = fallback.predict(X_te)
    return pred


# ───────────────────────────────────────────────────────────────────────
# Tier A additions (audit 2026-04-24)
# ───────────────────────────────────────────────────────────────────────

_FM_DIAG = []  # per-fold: number of monthly regressions, avg R², etc.


def fama_macbeth_model(X_tr, y_tr, X_te, features, train_dates=None, **kwargs):
    """1c: Fama-MacBeth per-month cross-sectional regression (Fama & MacBeth 1973).

    Uses np.linalg.lstsq (SVD-based) to handle rank-deficiency from constant
    flag columns (has_analyst_consensus ≈ 99%, has_positive_earnings ≈ 92%)
    without introducing Ridge shrinkage bias.
    """
    if train_dates is None:
        return ols_model(X_tr, y_tr, X_te, features)

    n_features = X_tr.shape[1]
    min_obs = n_features + 5

    betas = []
    intercepts = []
    unique_months = np.unique(train_dates)

    for m in unique_months:
        mask = train_dates == m
        if mask.sum() < min_obs:
            continue
        X_m = X_tr[mask]
        y_m = y_tr[mask]
        # Use SVD lstsq with intercept column for rank-deficiency handling
        X_with_intercept = np.column_stack([np.ones(len(X_m)), X_m])
        try:
            coef_full, *_ = np.linalg.lstsq(X_with_intercept, y_m, rcond=None)
            intercept = float(coef_full[0])
            beta = coef_full[1:]
            # Sanity check: if any coefficient is extreme (>1e4), skip this month
            if np.abs(beta).max() > 1e4:
                continue
            betas.append(beta)
            intercepts.append(intercept)
        except np.linalg.LinAlgError:
            continue

    if not betas:
        return ols_model(X_tr, y_tr, X_te, features)

    beta_mean = np.mean(betas, axis=0)
    intercept_mean = float(np.mean(intercepts))
    pred = X_te @ beta_mean + intercept_mean

    _FM_DIAG.append({
        "n_monthly_regressions": len(betas),
        "beta_mean_L2": float(np.linalg.norm(beta_mean)),
        "beta_max_abs": float(np.max(np.abs(beta_mean))),
    })
    return pred


def corr_adj_ic_ensemble_model(X_tr, y_tr, X_te, features, train_dates=None, **kwargs):
    """1d REVISED 2026-04-24: TRUE Barra factor model (Grinold-Kahn 2000, Ch.7).

    Replaces the previous heuristic (Corr(X)^{-1} · IC_vec) which was NOT actually
    Barra. This is the real thing:

    Step 1: Per-month cross-sectional regression y_m = X_m β + ε_m → factor
            return f_m (Fama-MacBeth-style slope vector, one per feature per month)
    Step 2: Stack f_m into factor return time series F (T × p)
    Step 3: Ledoit-Wolf shrunk covariance Σ_f (automatic λ selection)
            + time-series mean μ_f
    Step 4: Optimal factor weights w* = Σ_f^{-1} μ_f (max-Sharpe combination)
    Step 5: Predict y_test = X_test @ w*

    Distinguishes from:
      - 1b IC-Ensemble: univariate IC |r| weighted sum (no joint adjustment)
      - 1c Fama-MacBeth: w = mean(f_m) directly (no covariance adjustment)
      - This (Barra): w = Σ_f^{-1} μ_f (cov-adjusted, max-Sharpe over factors)
    """
    if train_dates is None:
        return ols_model(X_tr, y_tr, X_te, features)

    n_features = X_tr.shape[1]
    min_obs = n_features + 5
    unique_months = np.unique(train_dates)

    # Step 1-2: extract monthly factor returns via cross-sectional regression
    factor_returns = []
    for m in unique_months:
        mask = train_dates == m
        if mask.sum() < min_obs:
            continue
        X_m = X_tr[mask]
        y_m = y_tr[mask]
        X_with_intercept = np.column_stack([np.ones(len(X_m)), X_m])
        try:
            coef_full, *_ = np.linalg.lstsq(X_with_intercept, y_m, rcond=None)
            beta = coef_full[1:]
            if np.abs(beta).max() > 1e4:
                continue
            factor_returns.append(beta)
        except np.linalg.LinAlgError:
            continue

    # Need enough months for stable Σ_f (at minimum p+5)
    if len(factor_returns) < n_features + 5:
        return ols_model(X_tr, y_tr, X_te, features)

    F = np.array(factor_returns)

    # Step 3: Ledoit-Wolf shrunk covariance + mean
    mu_f = F.mean(axis=0)
    try:
        lw = LedoitWolf()
        lw.fit(F)
        Sigma_shrunk = lw.covariance_
    except Exception:
        Sigma_shrunk = np.cov(F, rowvar=False) + 0.01 * np.eye(n_features)

    # Step 4: optimal factor weights w* = Σ_f^{-1} μ_f
    try:
        w_star = np.linalg.solve(Sigma_shrunk, mu_f)
    except np.linalg.LinAlgError:
        w_star = mu_f  # fallback to pure mean (reduces to FM)

    # Step 5: predict — no intercept needed (z-scored features), scale irrelevant for IC
    return X_te @ w_star


def adaptive_lasso_model(X_tr, y_tr, X_te, features, train_dates=None, **kwargs):
    """[CAVEAT-unreliable_SNR] 2e Adaptive LASSO (Zou 2006). Same L1 failure mode at IC~0.01.

    Uses same-fold OLS for adaptive weights (not test leakage, but in-fold overfit risk).
    """
    ols = LinearRegression()
    ols.fit(X_tr, y_tr)
    beta_ols = ols.coef_

    eps = 1e-3
    w_adapt = 1.0 / (np.abs(beta_ols) + eps)

    X_tr_r = X_tr / w_adapt
    X_te_r = X_te / w_adapt

    lasso = LassoCV(alphas=LASSO_ALPHAS, cv=_cv_for(train_dates), max_iter=10000)
    lasso.fit(X_tr_r, y_tr)
    pred = lasso.predict(X_te_r)

    if np.std(pred) < 1e-10:
        return ols.predict(X_te)
    return pred


def expand_results(monthly_df):
    rows = []
    for _, r in monthly_df.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({DATE_COL: r[DATE_COL], STOCK_COL: t, "y_pred": yp, "y_true": yt})
    return pd.DataFrame(rows)


def main():
    print("Loading V2 panel...")
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, {len(ALL_FEATURE_COLS_V2)} features")

    models = [
        ("1a_OLS_v2", ols_model),
        ("1b_IC_Ensemble_v2", ic_ensemble_model),
        ("1c_FamaMacBeth_v2", fama_macbeth_model),
        ("1d_Barra_v2", corr_adj_ic_ensemble_model),  # True Barra (Grinold-Kahn)
        ("2a_LASSO_v2", lasso_model),                  # [unreliable_SNR]
        ("2b_Ridge_v2", ridge_model),
        ("2d_ElasticNet_v2", elastic_net_model),       # [unreliable_SNR]
        ("2e_AdaptiveLASSO_v2", adaptive_lasso_model), # [unreliable_SNR]
    ]

    all_results = {}
    for label, fn in models:
        print(f"\nRunning {label}...")
        monthly = run_walk_forward(df, fn, label)
        all_results[label] = monthly

        ic_series = monthly["IC"].dropna()
        expanded = expand_results(monthly)
        sharpe = compute_long_short_sharpe(expanded)

        print(f"  {label}: IC={ic_series.mean():+.4f} (std={ic_series.std():.4f}), "
              f"IC_IR={compute_ic_ir(ic_series):.4f}, "
              f"Hit={compute_hit_rate(ic_series):.1%}, "
              f"Sharpe={sharpe:.3f}, "
              f"n_months={len(ic_series)}")

        expanded.to_parquet(OUTPUT / f"results_{label}.parquet", index=False)
        monthly.to_csv(OUTPUT / f"monthly_{label}.csv", index=False)

    print("\n" + "=" * 80)
    print("RUNG 1-2 SUMMARY (V2 DATA — TimeSeriesSplit + winsorized y_tr)")
    print("=" * 80)
    rows = []
    for label, monthly in all_results.items():
        ic = monthly["IC"].dropna()
        expanded = expand_results(monthly)
        rows.append({
            "Model": label,
            "IC_mean": round(ic.mean(), 4),
            "IC_std": round(ic.std(), 4),
            "IC_IR": round(compute_ic_ir(ic), 4),
            "Hit_Rate": round(compute_hit_rate(ic), 3),
            "LS_Sharpe": round(compute_long_short_sharpe(expanded), 3),
            "n_months": len(ic),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT / "rung12_v2_summary.csv", index=False)

    if _ELASTIC_NET_DIAG:
        diag_df = pd.DataFrame(_ELASTIC_NET_DIAG)
        diag_df.to_csv(OUTPUT / "elastic_net_diag_v2.csv", index=False)
        print(f"\nElasticNet diagnostics ({len(diag_df)} folds):")
        print(f"  l1_ratio mean={diag_df['l1_ratio'].mean():.2f}")
        print(f"  alpha mean={diag_df['alpha'].mean():.4f}")
        print(f"  n_nonzero mean={diag_df['n_nonzero'].mean():.1f}")

    if _FM_DIAG:
        fm_df = pd.DataFrame(_FM_DIAG)
        fm_df.to_csv(OUTPUT / "fama_macbeth_diag_v2.csv", index=False)
        print(f"\nFama-MacBeth diagnostics ({len(fm_df)} folds):")
        print(f"  avg monthly regressions per fold: {fm_df['n_monthly_regressions'].mean():.1f}")
        print(f"  avg beta L2 norm: {fm_df['beta_mean_L2'].mean():.4f}")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
