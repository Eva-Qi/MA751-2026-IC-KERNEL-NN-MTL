"""
Rung 5 Regime: Regime-Gated Mixture-of-Experts MTL MLP

Architecture:
  SharedEncoder(64→32) + RegimeGate(→softmax over K experts)
  → per-task expert heads mixed by gate weights

Key differences from main.py (plain MTL):
  - Gate network conditioned on HMM regime posteriors
  - K expert heads per task (instead of 1 shared head)
  - HMM fitted inside walk-forward loop (expanding window, no look-ahead)

Ported from RG's origin/reg branch with fixes:
  - HMM look-ahead eliminated (expanding window fit per fold)
  - V2 features (14 stock-level, no macro in X)
  - Regime posteriors NOT StandardScaled (preserve simplex)
  - Missing regime filled with 1/N uniform prior (not 0.0)
  - Imports from config.py (no hardcoded column defs)
  - UncertaintyMTLLoss imported from main.py (no copy-paste)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

from config import (
    ALL_FEATURE_COLS_V2, REGIME_HMM_COLS, N_REGIMES,
    TARGET_COL, RET3M_COL, VOL_COL, SECTOR_COL, DATE_COL, STOCK_COL,
    DEFAULT_MIN_TRAIN_MONTHS, DEFAULT_PURGE_MONTHS,
)
from main import (
    UncertaintyMTLLoss,
    compute_monthly_ic,
    compute_long_short_sharpe,
    compute_ret3m_auxiliary_ic,
    compute_vol_auxiliary_corr,
    summarise as base_summarise,
)
from regime import (
    build_market_monthly_features,
    fit_and_predict_regime,
    merge_regime_into_panel,
)

FACTOR_COLS = ALL_FEATURE_COLS_V2  # 14 stock-level features
DEFAULT_N_EXPERTS = 3

ABLATION_TASKS = {
    "5a": {"ret"},
    "5b": {"ret", "ret3m"},
    "5c": {"ret", "vol"},
    "5d": {"ret", "ret3m", "vol"},
}


# ── Model ───────────────────────────────────────────────────────────────


class RegimeGatedMTLMoE(nn.Module):
    """
    Shared stock encoder + regime-driven gate + expert heads per task.
    """

    def __init__(
        self,
        n_factors: int,
        n_regime_features: int,
        n_experts: int = 3,
        hidden1: int = 64,
        hidden2: int = 32,
        gate_hidden: int = 16,
        dropout: float = 0.10,
        active_tasks: set | None = None,
    ):
        super().__init__()

        if active_tasks is None:
            active_tasks = {"ret", "ret3m", "vol"}
        self.active_tasks = set(active_tasks)
        self.n_experts = n_experts

        self.shared = nn.Sequential(
            nn.Linear(n_factors, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Linear(n_regime_features, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, n_experts),
        )

        if "ret" in self.active_tasks:
            self.ret_experts = nn.ModuleList(
                [nn.Linear(hidden2, 1) for _ in range(n_experts)]
            )
        if "ret3m" in self.active_tasks:
            self.ret3m_experts = nn.ModuleList(
                [nn.Linear(hidden2, 1) for _ in range(n_experts)]
            )
        if "vol" in self.active_tasks:
            self.vol_experts = nn.ModuleList(
                [nn.Linear(hidden2, 1) for _ in range(n_experts)]
            )

    def _mix_experts(self, z, gate_weights, experts):
        expert_outs = [expert(z).squeeze(-1) for expert in experts]
        expert_stack = torch.stack(expert_outs, dim=1)  # [B, K]
        return (gate_weights * expert_stack).sum(dim=1)

    def forward(self, x, r):
        z = self.shared(x)
        gate_weights = torch.softmax(self.gate(r), dim=1)

        out = {"gate_weights": gate_weights}

        if "ret" in self.active_tasks:
            out["ret"] = self._mix_experts(z, gate_weights, self.ret_experts)
        if "ret3m" in self.active_tasks:
            out["ret3m"] = self._mix_experts(z, gate_weights, self.ret3m_experts)
        if "vol" in self.active_tasks:
            out["vol"] = self._mix_experts(z, gate_weights, self.vol_experts)

        return out


# ── Data helpers ────────────────────────────────────────────────────────


def make_tensors_with_regime(
    df: pd.DataFrame,
    feature_cols: list[str],
    regime_cols: list[str],
    x_scaler: StandardScaler | None = None,
    vol_mean: float | None = None,
    vol_std: float | None = None,
):
    """
    Build tensors for stock features X, regime posteriors R, and targets.

    NOTE: Regime posteriors are NOT scaled — they are already on [0,1] simplex.
    Scaling would break the probability interpretation and confuse the gate.
    """
    X_raw = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0.0)

    if x_scaler is None:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X_raw)
    else:
        X = x_scaler.transform(X_raw)

    # Regime posteriors: pass through as-is (already [0,1], sum to 1)
    R = np.nan_to_num(df[regime_cols].values.astype(np.float32), nan=1.0 / len(regime_cols))

    y_ret = df[TARGET_COL].values.astype(np.float32)
    y_ret3m_raw = df[RET3M_COL].values.astype(np.float64)
    y_vol = df[VOL_COL].values.astype(np.float32)

    if vol_mean is None or vol_std is None:
        vol_mean = float(np.mean(y_vol))
        vol_std = float(np.std(y_vol))
        vol_std = max(vol_std, 1e-8)

    y_vol_std = ((y_vol - vol_mean) / vol_std).astype(np.float32)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(R, dtype=torch.float32),
        torch.tensor(y_ret, dtype=torch.float32),
        torch.tensor(y_ret3m_raw, dtype=torch.float32),
        torch.tensor(y_vol_std, dtype=torch.float32),
        x_scaler,
        vol_mean,
        vol_std,
    )


# ── Training ────────────────────────────────────────────────────────────


def train_one_fold(
    X_tr, R_tr, y_ret_tr, y_ret3m_tr, y_vol_tr,
    active_tasks,
    n_experts=DEFAULT_N_EXPERTS,
    epochs=150, batch_size=512, lr=1e-3, weight_decay=1e-5,
    dropout=0.10, patience=20, val_frac=0.10,
    device="cpu",
):
    n_val = max(1, int(len(X_tr) * val_frac))

    X_val, X_tr = X_tr[-n_val:], X_tr[:-n_val]
    R_val, R_tr = R_tr[-n_val:], R_tr[:-n_val]
    y_ret_val, y_ret_tr = y_ret_tr[-n_val:], y_ret_tr[:-n_val]
    y_3m_val, y_3m_tr = y_ret3m_tr[-n_val:], y_ret3m_tr[:-n_val]
    y_vol_val, y_vol_tr = y_vol_tr[-n_val:], y_vol_tr[:-n_val]

    criterion = UncertaintyMTLLoss.from_train_data(
        y_ret_tr, y_3m_tr, y_vol_tr, active_tasks
    ).to(device)

    train_dl = DataLoader(
        TensorDataset(X_tr, R_tr, y_ret_tr, y_3m_tr, y_vol_tr),
        batch_size=batch_size, shuffle=True,
    )

    model = RegimeGatedMTLMoE(
        n_factors=X_tr.shape[1],
        n_regime_features=R_tr.shape[1],
        n_experts=n_experts,
        dropout=dropout,
        active_tasks=active_tasks,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-5
    )

    best_val = float("inf")
    best_model_state = None
    best_loss_state = None
    no_imp = 0

    for _ in range(epochs):
        model.train()
        criterion.train()
        for Xb, Rb, yr, y3m, yv in train_dl:
            Xb, Rb, yr, y3m, yv = (t.to(device) for t in (Xb, Rb, yr, y3m, yv))
            optimizer.zero_grad()
            preds = model(Xb, Rb)
            loss = criterion(preds, yr, y3m, yv)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        criterion.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device), R_val.to(device))
            val_loss = criterion(
                val_preds, y_ret_val.to(device),
                y_3m_val.to(device), y_vol_val.to(device),
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_loss_state = {k: v.cpu().clone() for k, v in criterion.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_loss_state is not None:
        criterion.load_state_dict(best_loss_state)

    return (
        model.to("cpu"),
        criterion.learned_task_weights(),
        criterion.learned_log_vars(),
    )


# ── Walk-forward with per-fold HMM ─────────────────────────────────────


def walk_forward_evaluate(
    df: pd.DataFrame,
    market_features: pd.DataFrame,
    active_tasks: set,
    feature_cols: list[str] = FACTOR_COLS,
    regime_cols: list[str] = REGIME_HMM_COLS,
    n_experts: int = DEFAULT_N_EXPERTS,
    min_train_months: int = DEFAULT_MIN_TRAIN_MONTHS,
    purge_months: int = DEFAULT_PURGE_MONTHS,
    epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.10,
    patience: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward evaluation with per-fold HMM regime fitting.

    For each fold:
      1. Fit HMM on market data up to train_end (expanding window)
      2. Predict regime posteriors for all months
      3. Merge regime into stock panel
      4. Train RegimeGatedMoE on enriched panel
      5. Predict on test month

    This eliminates look-ahead bias in regime labels.
    """
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []

    for i, test_month in enumerate(
        months[min_train_months + purge_months:],
        start=min_train_months + purge_months,
    ):
        train_end = months[i - purge_months - 1]

        # ── Step 1: Fit HMM on train data only ──
        try:
            regime_df = fit_and_predict_regime(
                market_features, train_end=train_end
            )
        except ValueError as e:
            if verbose:
                print(f"  fold {i}: HMM skip — {e}")
            continue

        # ── Step 2: Merge regime into panel for this fold ──
        df_tr_raw = df[df[DATE_COL] <= train_end].copy()
        df_te_raw = df[df[DATE_COL] == test_month].copy()

        if df_te_raw.empty:
            continue

        df_tr = merge_regime_into_panel(df_tr_raw, regime_df)
        df_te = merge_regime_into_panel(df_te_raw, regime_df)

        # Check regime columns exist
        available_regime = [c for c in regime_cols if c in df_tr.columns]
        if not available_regime:
            if verbose:
                print(f"  fold {i}: no regime columns available")
            continue

        # ── Step 3: Build tensors ──
        (
            X_tr, R_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            x_scaler, vol_mean, vol_std,
        ) = make_tensors_with_regime(
            df_tr, feature_cols=feature_cols, regime_cols=available_regime,
        )

        (
            X_te, R_te, y_ret_te, y_3m_te, y_vol_te_std,
            _, _, _,
        ) = make_tensors_with_regime(
            df_te, feature_cols=feature_cols, regime_cols=available_regime,
            x_scaler=x_scaler, vol_mean=vol_mean, vol_std=vol_std,
        )

        # ── Step 4: Train ──
        model, learned_weights, learned_log_vars = train_one_fold(
            X_tr, R_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            active_tasks=active_tasks,
            n_experts=n_experts,
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, dropout=dropout,
            patience=patience, device=device,
        )

        # ── Step 5: Predict ──
        model.eval()
        with torch.no_grad():
            pred_dict = model(X_te, R_te)

        y_pred_ret = pred_dict["ret"].numpy()

        out = pd.DataFrame({
            DATE_COL: test_month,
            STOCK_COL: df_te_raw[STOCK_COL].values,
            SECTOR_COL: df_te_raw[SECTOR_COL].values,
            "y_true": y_ret_te.numpy(),
            "y_pred": y_pred_ret,
            "fold": i,
            "realized_vol_true": df_te_raw[VOL_COL].values.astype(float),
            "fwd_ret_3m_true": df_te_raw[RET3M_COL].values.astype(float),
        })

        # Store regime info
        for c in available_regime:
            out[c] = df_te[c].values

        if "ret3m" in pred_dict:
            out["ret3m_pred"] = pred_dict["ret3m"].numpy()
        if "vol" in pred_dict:
            out["vol_pred"] = pred_dict["vol"].numpy() * vol_std + vol_mean

        gate_w = pred_dict["gate_weights"].numpy()
        for k in range(gate_w.shape[1]):
            out[f"gate_w_{k}"] = gate_w[:, k]

        for task_name in ["ret", "ret3m", "vol"]:
            if task_name in learned_weights:
                out[f"uw_weight_{task_name}"] = learned_weights[task_name]
                out[f"uw_logvar_{task_name}"] = learned_log_vars[task_name]

        results.append(out)

        if verbose:
            ic = spearmanr(out["y_true"], out["y_pred"]).statistic
            weight_msg = ", ".join(f"{k}={v:.3f}" for k, v in learned_weights.items())
            gate_msg = ", ".join(f"g{k}={out[f'gate_w_{k}'].mean():.3f}" for k in range(gate_w.shape[1]))
            print(
                f"fold {i:3d} | {str(test_month)[:7]} | "
                f"n_train={len(df_tr):6,d} | IC={ic:+.4f} | "
                f"UW: {weight_msg} | Gate: {gate_msg}"
            )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ── Metrics (extend base with gate summary) ─────────────────────────────


def compute_gate_summary(results: pd.DataFrame) -> dict:
    gate_cols = [c for c in results.columns if c.startswith("gate_w_")]
    out = {}
    for c in gate_cols:
        out[f"{c}_mean"] = round(float(results[c].mean()), 4)
        out[f"{c}_std"] = round(float(results[c].std(ddof=1)), 4) if results[c].nunique() > 1 else 0.0
    return out


def summarise(results: pd.DataFrame, label: str = "") -> dict:
    row = base_summarise(results, label=label)
    row.update(compute_gate_summary(results))
    return row


# ── Run all ablations ───────────────────────────────────────────────────


def run_all_ablations(
    df: pd.DataFrame,
    market_features: pd.DataFrame,
    n_experts: int = DEFAULT_N_EXPERTS,
    device: str = "cpu",
    verbose: bool = True,
    save_results: bool = True,
    output_dir: str = "output",
) -> pd.DataFrame:
    if save_results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []

    for variant, tasks in ABLATION_TASKS.items():
        print(f"\n{'=' * 70}")
        print(f"Regime MoE Ablation {variant}: tasks = {tasks} | experts = {n_experts}")
        print(f"{'=' * 70}")

        results = walk_forward_evaluate(
            df=df,
            market_features=market_features,
            active_tasks=tasks,
            n_experts=n_experts,
            device=device,
            verbose=verbose,
        )

        if results.empty:
            print(f"  {variant}: no results (all folds skipped)")
            continue

        summary_row = summarise(results, label=variant)
        rows.append(summary_row)

        if save_results:
            results_path = Path(output_dir) / f"results_{variant}_regime_moe.parquet"
            results.to_parquet(results_path, index=False)

        print(f"\nSummary ({variant}):")
        for k, v in summary_row.items():
            print(f"  {k}: {v}")

    summary_df = pd.DataFrame(rows)

    if save_results:
        summary_path = Path(output_dir) / "ablation_summary_regime_moe.csv"
        summary_df.to_csv(summary_path, index=False)

    return summary_df


# ── Main ────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rung 5 Regime-Gated MoE MTL MLP (no look-ahead)"
    )
    parser.add_argument("--data", type=str, default="data/master_panel_v2.parquet")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--n_experts", type=int, default=DEFAULT_N_EXPERTS)
    args = parser.parse_args()

    print("Loading panel...")
    df = pd.read_parquet(args.data)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Sanity check: need V2 features
    needed = [DATE_COL, STOCK_COL, SECTOR_COL, TARGET_COL, RET3M_COL, VOL_COL] + FACTOR_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Are you using master_panel_v2?")

    # Drop rows missing core targets
    df = df.dropna(subset=[TARGET_COL, VOL_COL])

    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    print(f"Tickers: {df[STOCK_COL].nunique():,}")
    print(f"Months: {df[DATE_COL].nunique():,}")

    # Build market features for HMM
    print("\nBuilding market features for HMM...")
    market_features = build_market_monthly_features()
    print(f"Market features: {len(market_features)} months")

    # Sanitize factor columns
    for c in FACTOR_COLS:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    summary_df = run_all_ablations(
        df=df,
        market_features=market_features,
        n_experts=args.n_experts,
        device=args.device,
        verbose=not args.quiet,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )

    print(f"\n{'=' * 90}")
    print("REGIME-GATED MoE ABLATION SUMMARY (no look-ahead)")
    print(f"{'=' * 90}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
