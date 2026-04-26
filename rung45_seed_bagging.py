"""
Rung 4/5 K-Seed Bagging — variance reduction via seed ensemble.

HYPOTHESIS: Single-seed MLP has SNR=1.33 (mean IC=0.0108, std=0.0082).
Averaging K=5 independent seeds should reduce prediction variance by sqrt(K),
boosting effective SNR to ~3.0 and improving out-of-sample Sharpe.

STAGE 1 — Rung 4 MLP (hidden=64) seed bagging [PRIMARY]
    For each walk-forward fold: train K=5 MLPs with seeds [0,1,2,42,123],
    average predictions, compute Spearman IC of bagged forecast.
    Saves:
      output/rung4_seed_bagging_summary.csv   — aggregate stats
      output/rung4_seed_bagging_diag.csv      — per-fold diagnostics

STAGE 2 — Verdict vs single-seed baseline
    Compare bagged IC/Sharpe to the 5 individual seeds from rung5_audit.py.
    Report SNR improvement.

STAGE 3 — Skipped (MTL complexity; document as future work).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from itertools import combinations

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

# Seeds from prior single-seed audit (same for direct comparison)
SEEDS = [0, 1, 2, 42, 123]

# Single-seed baseline from rung5_audit.py (for Stage 2 verdict)
PRIOR_SINGLE_SEED = {
    0:   {"IC": 0.0210, "Sharpe": 0.648},
    1:   {"IC": 0.0020, "Sharpe": 0.097},
    2:   {"IC": 0.0130, "Sharpe": 0.414},
    42:  {"IC": 0.0007, "Sharpe": 0.347},
    123: {"IC": 0.0176, "Sharpe": 0.556},
}
PRIOR_MEAN_IC     = np.mean([v["IC"]     for v in PRIOR_SINGLE_SEED.values()])
PRIOR_MEAN_SHARPE = np.mean([v["Sharpe"] for v in PRIOR_SINGLE_SEED.values()])
PRIOR_IC_STD      = np.std( [v["IC"]     for v in PRIOR_SINGLE_SEED.values()])
PRIOR_SNR         = PRIOR_MEAN_IC / max(PRIOR_IC_STD, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# MLP architecture — mirrors rung5_audit.py exactly
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Match main.py rung4 arch: n_in → hidden → hidden//2 → 1, ReLU + dropout 0.10."""
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_tr, y_tr, X_val, y_val, X_te, hidden, seed,
                   epochs=100, patience=20, lr=1e-3, batch=256):
    """Train one MLP, return test predictions. Mirrors rung5_audit.py exactly."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X_tr.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    X_tr_t  = torch.FloatTensor(X_tr)
    y_tr_t  = torch.FloatTensor(y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_te_t  = torch.FloatTensor(X_te)
    y_std = max(1e-6, float(np.std(y_tr)))
    huber_beta = max(0.01, 0.5 * y_std)

    best_val = float("inf")
    patience_left = patience
    best_state = None
    n_tr = len(X_tr_t)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(X_tr_t[idx])
            loss = F.smooth_l1_loss(pred, y_tr_t[idx], beta=huber_beta)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(X_val_t)
            vloss = F.smooth_l1_loss(vpred, y_val_t, beta=huber_beta).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X_te_t).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward harness with K-seed bagging
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_seed_bagging(df, seeds, hidden=64):
    """
    Walk-forward with K-seed bagging.

    For each test fold:
      1. Train K MLPs (one per seed).
      2. Bagged prediction = mean across K seed predictions.
      3. Record per-seed IC + bagged IC + diagnostic stats.

    Returns:
      monthly_df  — one row per fold with bagged IC + y_pred/y_true lists
      diag_rows   — list of per-fold diagnostic dicts
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    start  = DEFAULT_MIN_TRAIN_MONTHS + DEFAULT_PURGE_MONTHS

    monthly_results = []
    diag_rows       = []
    n_folds         = len(months) - start

    t_total = time.time()

    for fold_idx, (i, test_month) in enumerate(
        enumerate(months[start:], start=start), start=1
    ):
        t_fold = time.time()
        train_end = months[i - DEFAULT_PURGE_MONTHS - 1]
        df_tr = df[df[DATE_COL] <= train_end]
        df_te = df[df[DATE_COL] == test_month]

        if len(df_te) < 10:
            continue

        # Chronological validation split (10% tail of train months)
        tr_months   = sorted(df_tr[DATE_COL].unique())
        n_val_m     = max(1, int(len(tr_months) * 0.10))
        val_months  = set(tr_months[-n_val_m:])
        df_tr_main  = df_tr[~df_tr[DATE_COL].isin(val_months)]
        df_val      = df_tr[ df_tr[DATE_COL].isin(val_months)]

        if len(df_val) < 10 or len(df_tr_main) < 100:
            continue

        X_tr  = np.nan_to_num(df_tr_main[FEATURES].values, nan=0.0)
        y_tr  = df_tr_main[TARGET_COL].values
        X_val = np.nan_to_num(df_val[FEATURES].values,     nan=0.0)
        y_val = df_val[TARGET_COL].values
        X_te  = np.nan_to_num(df_te[FEATURES].values,      nan=0.0)
        y_te  = df_te[TARGET_COL].values

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_val_s  = scaler.transform(X_val)
        X_te_s   = scaler.transform(X_te)

        # Train K models with different seeds
        seed_preds = {}
        seed_ics   = {}
        for seed in seeds:
            fold_seed = seed * 1000 + i      # same formula as rung5_audit.py
            y_pred_s  = train_mlp_fold(
                X_tr_s, y_tr, X_val_s, y_val, X_te_s,
                hidden=hidden, seed=fold_seed,
            )
            seed_preds[seed] = y_pred_s
            seed_ics[seed]   = float(spearmanr(y_te, y_pred_s).statistic)

        # Bagged prediction = mean across all seed predictions
        pred_matrix    = np.stack(list(seed_preds.values()), axis=0)  # (K, n_te)
        y_pred_bagged  = pred_matrix.mean(axis=0)
        bagged_ic      = float(spearmanr(y_te, y_pred_bagged).statistic)

        # Pairwise prediction correlation (average) — measures seed diversity
        pred_corrs = []
        for s1, s2 in combinations(seeds, 2):
            c = float(np.corrcoef(seed_preds[s1], seed_preds[s2])[0, 1])
            pred_corrs.append(c)
        pred_corr_mean = float(np.mean(pred_corrs)) if pred_corrs else np.nan

        # Std of per-seed IC this fold (local diversity measure)
        std_across_seeds = float(np.std(list(seed_ics.values())))

        elapsed_fold = time.time() - t_fold
        elapsed_total = time.time() - t_total

        # Progress print
        seed_ic_str = "  ".join(
            f"s{s}={seed_ics[s]:+.4f}" for s in seeds
        )
        print(
            f"  fold {fold_idx:3d}/{n_folds}  [{elapsed_fold:.1f}s]  "
            f"bagged_IC={bagged_ic:+.4f}  std={std_across_seeds:.4f}  "
            f"corr={pred_corr_mean:.3f}  |  {seed_ic_str}  |  "
            f"total={elapsed_total/60:.1f}min",
            flush=True,
        )

        monthly_results.append({
            DATE_COL:       test_month,
            "IC":           bagged_ic,
            "pred_std":     float(np.std(y_pred_bagged)),
            "pred_mean":    float(np.mean(y_pred_bagged)),
            "y_pred_list":  y_pred_bagged.tolist(),
            "y_true_list":  y_te.tolist(),
            "tickers":      df_te[STOCK_COL].values.tolist(),
        })

        diag_row = {
            "fold":               fold_idx,
            "test_month":         test_month,
            "bagged_ic":          bagged_ic,
            "std_across_seeds":   std_across_seeds,
            "pred_corr_mean":     pred_corr_mean,
        }
        for s in seeds:
            diag_row[f"seed_{s}_ic"] = seed_ics[s]
        diag_rows.append(diag_row)

    monthly_df = pd.DataFrame(monthly_results)
    return monthly_df, diag_rows


def aggregate_bagged(monthly_df):
    """Compute summary stats from bagged monthly IC + LS Sharpe."""
    ic_series = monthly_df["IC"].dropna()

    rows = []
    for _, r in monthly_df.iterrows():
        for t, yp, yt in zip(r["tickers"], r["y_pred_list"], r["y_true_list"]):
            rows.append({
                DATE_COL: r[DATE_COL],
                STOCK_COL: t,
                "y_pred": yp,
                "y_true": yt,
            })
    expanded = pd.DataFrame(rows)
    sharpe   = compute_long_short_sharpe(expanded)

    return {
        "IC_mean":    float(ic_series.mean()),
        "IC_std":     float(ic_series.std()),
        "IC_IR":      float(compute_ic_ir(ic_series)),
        "Hit_Rate":   float(compute_hit_rate(ic_series)),
        "LS_Sharpe":  float(sharpe),
        "n_months":   int(len(ic_series)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("RUNG 4/5 K-SEED BAGGING — Variance Reduction via Seed Ensemble")
    print("=" * 80, flush=True)
    print(f"Seeds: {SEEDS}  (K={len(SEEDS)})")
    print(f"Prior single-seed baseline:  mean IC={PRIOR_MEAN_IC:+.4f}, "
          f"std={PRIOR_IC_STD:.4f}, SNR={PRIOR_SNR:.2f}, "
          f"mean Sharpe={PRIOR_MEAN_SHARPE:.3f}", flush=True)
    print(f"Expected SNR boost: sqrt({len(SEEDS)}) × {PRIOR_SNR:.2f} = "
          f"{np.sqrt(len(SEEDS)) * PRIOR_SNR:.2f}", flush=True)

    print("\nLoading V3 panel...", flush=True)
    df = pd.read_parquet(DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  {len(df):,} rows, {df[DATE_COL].nunique()} months, "
          f"{len(FEATURES)} features", flush=True)

    # ─────────────────────────────────────────────────────────────
    # STAGE 1 — Rung 4 MLP seed bagging
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 1: Rung 4 MLP (hidden=64) — K=5 seed bagging")
    print("=" * 80, flush=True)

    t0 = time.time()
    monthly_df, diag_rows = walk_forward_seed_bagging(df, seeds=SEEDS, hidden=64)
    elapsed = time.time() - t0
    print(f"\n  Walk-forward complete: {elapsed/60:.1f} min", flush=True)

    if len(monthly_df) == 0:
        print("  ERROR: No folds returned results. Aborting.", flush=True)
        return

    summary = aggregate_bagged(monthly_df)
    ic_series = monthly_df["IC"].dropna()

    print("\n  === STAGE 1 Results ===")
    print(f"  n_months  : {summary['n_months']}")
    print(f"  Bagged IC mean : {summary['IC_mean']:+.4f}")
    print(f"  Bagged IC std  : {summary['IC_std']:.4f}")
    print(f"  Bagged IC IR   : {summary['IC_IR']:.3f}")
    print(f"  Hit Rate       : {summary['Hit_Rate']:.1%}")
    print(f"  LS Sharpe      : {summary['LS_Sharpe']:.3f}", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Save outputs
    # ─────────────────────────────────────────────────────────────
    summary_row = {
        "label":           "rung4_mlp_h64_k5seed_bagged",
        "seeds":           str(SEEDS),
        "hidden":          64,
        **summary,
    }
    pd.DataFrame([summary_row]).to_csv(
        OUTPUT / "rung4_seed_bagging_summary.csv", index=False
    )
    print("  Saved: output/rung4_seed_bagging_summary.csv", flush=True)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUTPUT / "rung4_seed_bagging_diag.csv", index=False)
    print("  Saved: output/rung4_seed_bagging_diag.csv", flush=True)

    # ─────────────────────────────────────────────────────────────
    # STAGE 2 — Verdict vs single-seed baseline
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 2: Verdict vs single-seed baseline")
    print("=" * 80, flush=True)

    # Bagged SNR from per-fold std (diag)
    diag_df = pd.DataFrame(diag_rows)
    if "std_across_seeds" in diag_df.columns:
        mean_within_fold_std = diag_df["std_across_seeds"].mean()
        # Effective SNR: bagged IC / mean within-fold seed spread
        within_fold_snr = (
            abs(summary["IC_mean"]) / max(mean_within_fold_std, 1e-8)
        )
    else:
        mean_within_fold_std = np.nan
        within_fold_snr = np.nan

    # Bagged IC / std of bagged monthly IC (across time)
    bagged_ic_snr = abs(summary["IC_mean"]) / max(summary["IC_std"], 1e-8)

    sharpe_improvement = summary["LS_Sharpe"] - PRIOR_MEAN_SHARPE
    ic_delta           = summary["IC_mean"] - PRIOR_MEAN_IC
    ridge_sharpe       = 0.925  # from rung5_audit.py Ridge α=1

    print(f"  Prior single-seed:  IC={PRIOR_MEAN_IC:+.4f} ± {PRIOR_IC_STD:.4f}, "
          f"SNR={PRIOR_SNR:.2f}, Sharpe={PRIOR_MEAN_SHARPE:.3f}")
    print(f"  Bagged (K=5):       IC={summary['IC_mean']:+.4f} ± {summary['IC_std']:.4f}, "
          f"SNR(across-time)={bagged_ic_snr:.2f}, Sharpe={summary['LS_Sharpe']:.3f}")
    print(f"  IC delta vs prior:  {ic_delta:+.4f}  (expected ≈ 0 — bagging shouldn't shift mean)")
    print(f"  Sharpe improvement: {sharpe_improvement:+.3f}")
    print(f"  SNR (across-time):  {bagged_ic_snr:.2f}  (prior single-seed: {PRIOR_SNR:.2f})")
    if not np.isnan(within_fold_snr):
        print(f"  SNR (within-fold):  {within_fold_snr:.2f}  (bagged IC / mean seed spread per fold)")
    print(f"  Ridge Sharpe baseline: {ridge_sharpe:.3f}")

    # Verdict
    print("\n  === VERDICT ===")
    if summary["LS_Sharpe"] >= ridge_sharpe:
        print(f"  OUTSTANDING: Bagged Sharpe {summary['LS_Sharpe']:.3f} >= Ridge {ridge_sharpe:.3f} — MLP beats Ridge!")
    elif summary["LS_Sharpe"] >= 0.60:
        print(f"  SOLID: Bagged Sharpe {summary['LS_Sharpe']:.3f} >= 0.60 — meaningful improvement from bagging.")
    elif summary["LS_Sharpe"] >= PRIOR_MEAN_SHARPE:
        print(f"  MODEST: Bagged Sharpe {summary['LS_Sharpe']:.3f} > prior mean {PRIOR_MEAN_SHARPE:.3f} — marginal gain.")
    else:
        print(f"  NEUTRAL/NEGATIVE: Bagged Sharpe {summary['LS_Sharpe']:.3f} < prior mean {PRIOR_MEAN_SHARPE:.3f} — no improvement.")

    if bagged_ic_snr >= 2.0:
        print(f"  SNR (across-time) = {bagged_ic_snr:.2f} >= 2 — signal distinguishable from time-series noise.")
    elif bagged_ic_snr >= 1.5:
        print(f"  SNR (across-time) = {bagged_ic_snr:.2f} in [1.5,2) — improvement over single-seed SNR={PRIOR_SNR:.2f}.")
    else:
        print(f"  SNR (across-time) = {bagged_ic_snr:.2f} — limited improvement over single-seed SNR={PRIOR_SNR:.2f}.")

    # ─────────────────────────────────────────────────────────────
    # STAGE 3 — MTL bagging (skipped)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 3: MTL bagging — SKIPPED (future work)")
    print("=" * 80)
    print("  Rationale: Stage 1 took sufficient time; MTL 3-head machinery adds")
    print("  complexity. To implement: replicate MTL class from main.py, add same")
    print("  K-seed loop, average predictions across seeds before IC computation.")

    # ─────────────────────────────────────────────────────────────
    # Final summary table
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    rows_final = []
    for seed, vals in PRIOR_SINGLE_SEED.items():
        rows_final.append({
            "model": f"MLP_h64_seed{seed}",
            "IC_mean": vals["IC"],
            "LS_Sharpe": vals["Sharpe"],
            "type": "single-seed (prior audit)",
        })
    rows_final.append({
        "model": f"MLP_h64_K{len(SEEDS)}_bagged",
        "IC_mean": summary["IC_mean"],
        "LS_Sharpe": summary["LS_Sharpe"],
        "type": "seed-bagged (this run)",
    })
    rows_final.append({
        "model": "Ridge_alpha1",
        "IC_mean": np.nan,
        "LS_Sharpe": ridge_sharpe,
        "type": "linear baseline",
    })
    final_df = pd.DataFrame(rows_final)
    print(final_df.to_string(index=False))
    final_df.to_csv(OUTPUT / "rung4_seed_bagging_final_comparison.csv", index=False)
    print("\n  Saved: output/rung4_seed_bagging_final_comparison.csv")
    print("=" * 80)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
