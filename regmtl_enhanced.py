"""
Rung 5 Enhanced: Regime-Gated MoE with all proposed improvements.

Changes vs regmtl.py (baseline MoE):
  1. Feature pruning: 7 features (LASSO core + regime-conditional signals)
     instead of 14 — halves parameters, expected lower turnover
  2. Enriched gate: regime posteriors + macro (VIX, T10Y2Y, spread) = 6 dims
     instead of 3 — gate knows market conditions, not just HMM state
  3. Scaled architecture: 32→16 encoder (matches fewer features)
  4. Interaction features: IVOL×Beta, IVOL×GP computed on-the-fly

Run side-by-side with regmtl.py to compare.
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
    ENHANCED_MOE_FEATURE_COLS, REGIME_HMM_COLS, N_REGIMES,
    GATE_MACRO_COLS, MACRO_COLS,
    TARGET_COL, RET3M_COL, VOL_COL, FWD_VOL_COL, SECTOR_COL, DATE_COL, STOCK_COL,
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

# Feature set: 7 core + 2 interactions = 9 stock features
STOCK_FEATURES = ENHANCED_MOE_FEATURE_COLS  # 7 features

INTERACTION_FEATURES = [
    ("IVOL_zscore", "Beta_zscore", "IVOL_x_Beta"),                      # risk clustering
    ("IVOL_zscore", "GrossProfitability_zscore", "IVOL_x_GP"),           # quality vol
    ("AmihudIlliquidity_zscore", "Turnover_zscore", "Amihud_x_Turnover"),# liquidity dual
    ("AmihudIlliquidity_zscore", "GrossProfitability_zscore", "Amihud_x_GP"),  # illiquid quality
]

DEFAULT_N_EXPERTS = 3

ABLATION_TASKS = {
    "5a": {"ret"},
    "5b": {"ret", "ret3m"},
    "5c": {"ret", "vol"},
    "5d": {"ret", "ret3m", "vol"},
}


# ── Model: Scaled-down encoder + enriched gate ─────────────────────────


class EnhancedRegimeMoE(nn.Module):
    """
    Smaller encoder (32→16 for fewer features) + enriched gate
    (regime posteriors + macro indicators).
    """

    def __init__(
        self,
        n_factors: int,
        n_gate_features: int,
        n_experts: int = 3,
        hidden1: int = 32,      # smaller than baseline 64
        hidden2: int = 16,      # smaller than baseline 32
        gate_hidden: int = 12,
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

        # Enriched gate: regime posteriors + macro features
        self.gate = nn.Sequential(
            nn.Linear(n_gate_features, gate_hidden),
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
        expert_stack = torch.stack(expert_outs, dim=1)
        return (gate_weights * expert_stack).sum(dim=1)

    def forward(self, x, g):
        z = self.shared(x)
        gate_weights = torch.softmax(self.gate(g), dim=1)

        out = {"gate_weights": gate_weights}

        if "ret" in self.active_tasks:
            out["ret"] = self._mix_experts(z, gate_weights, self.ret_experts)
        if "ret3m" in self.active_tasks:
            out["ret3m"] = self._mix_experts(z, gate_weights, self.ret3m_experts)
        if "vol" in self.active_tasks:
            out["vol"] = self._mix_experts(z, gate_weights, self.vol_experts)

        return out


# ── Data helpers ────────────────────────────────────────────────────────


def add_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add interaction features (products of z-scored columns). Returns df + new col names."""
    df = df.copy()
    ixn_cols = []
    for col_a, col_b, name in INTERACTION_FEATURES:
        if col_a in df.columns and col_b in df.columns:
            df[name] = df[col_a] * df[col_b]
            ixn_cols.append(name)
    return df, ixn_cols


def make_tensors_enhanced(
    df: pd.DataFrame,
    feature_cols: list[str],
    regime_cols: list[str],
    gate_macro_cols: list[str],
    x_scaler: StandardScaler | None = None,
    g_scaler: StandardScaler | None = None,
    vol_mean: float | None = None,
    vol_std: float | None = None,
):
    """
    Build tensors:
      X — stock features (7 base + 2 interactions)
      G — gate conditioning (3 regime posteriors + 3 macro indicators, scaled)
      y_ret, y_ret3m, y_vol_std — targets
    """
    # Stock features
    X_raw = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0.0)
    if x_scaler is None:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X_raw)
    else:
        X = x_scaler.transform(X_raw)

    # Gate features: regime posteriors (unscaled) + macro (scaled)
    regime_vals = np.nan_to_num(
        df[regime_cols].values.astype(np.float32),
        nan=1.0 / len(regime_cols)
    )
    available_macro = [c for c in gate_macro_cols if c in df.columns]
    if available_macro:
        macro_vals = np.nan_to_num(df[available_macro].values.astype(np.float32), nan=0.0)
        if g_scaler is None:
            g_scaler = StandardScaler()
            macro_scaled = g_scaler.fit_transform(macro_vals)
        else:
            macro_scaled = g_scaler.transform(macro_vals)
        G = np.hstack([regime_vals, macro_scaled])
    else:
        G = regime_vals
        if g_scaler is None:
            g_scaler = StandardScaler()  # dummy, won't be used

    # Targets
    y_ret = df[TARGET_COL].values.astype(np.float32)
    y_ret3m_raw = df[RET3M_COL].values.astype(np.float64)

    y_vol_ref = df[VOL_COL].values.astype(np.float32)
    y_vol_raw = df[FWD_VOL_COL].values.astype(np.float64)

    if vol_mean is None or vol_std is None:
        vol_mean = float(np.nanmean(y_vol_ref))
        vol_std = float(np.nanstd(y_vol_ref))
        vol_std = max(vol_std, 1e-8)

    y_vol_std = ((y_vol_raw - vol_mean) / vol_std).astype(np.float32)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(G, dtype=torch.float32),
        torch.tensor(y_ret, dtype=torch.float32),
        torch.tensor(y_ret3m_raw, dtype=torch.float32),
        torch.tensor(y_vol_std, dtype=torch.float32),
        x_scaler,
        g_scaler,
        vol_mean,
        vol_std,
    )


# ── Training ────────────────────────────────────────────────────────────


def train_one_fold(
    X_tr, G_tr, y_ret_tr, y_ret3m_tr, y_vol_tr,
    active_tasks,
    n_experts=DEFAULT_N_EXPERTS,
    epochs=150, batch_size=512, lr=1e-3, weight_decay=1e-5,
    dropout=0.10, patience=20, val_frac=0.10,
    device="cpu",
    fold_idx=0,  # audit 2026-04-24: per-fold seed
):
    # audit 2026-04-24: reproducibility
    torch.manual_seed(fold_idx)
    np.random.seed(fold_idx)

    n_val = max(1, int(len(X_tr) * val_frac))

    X_val, X_tr = X_tr[-n_val:], X_tr[:-n_val]
    G_val, G_tr = G_tr[-n_val:], G_tr[:-n_val]
    y_ret_val, y_ret_tr = y_ret_tr[-n_val:], y_ret_tr[:-n_val]
    y_3m_val, y_3m_tr = y_ret3m_tr[-n_val:], y_ret3m_tr[:-n_val]
    y_vol_val, y_vol_tr = y_vol_tr[-n_val:], y_vol_tr[:-n_val]

    criterion = UncertaintyMTLLoss.from_train_data(
        y_ret_tr, y_3m_tr, y_vol_tr, active_tasks
    ).to(device)

    train_dl = DataLoader(
        TensorDataset(X_tr, G_tr, y_ret_tr, y_3m_tr, y_vol_tr),
        batch_size=batch_size, shuffle=True,
    )

    model = EnhancedRegimeMoE(
        n_factors=X_tr.shape[1],
        n_gate_features=G_tr.shape[1],
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
        for Xb, Gb, yr, y3m, yv in train_dl:
            Xb, Gb, yr, y3m, yv = (t.to(device) for t in (Xb, Gb, yr, y3m, yv))
            optimizer.zero_grad()
            preds = model(Xb, Gb)
            loss = criterion(preds, yr, y3m, yv)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        criterion.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device), G_val.to(device))
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


# ── Walk-forward ────────────────────────────────────────────────────────


def walk_forward_evaluate(
    df: pd.DataFrame,
    market_features: pd.DataFrame,
    active_tasks: set,
    feature_cols: list[str],
    regime_cols: list[str] = REGIME_HMM_COLS,
    gate_macro_cols: list[str] = GATE_MACRO_COLS,
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
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)
    months = sorted(df[DATE_COL].unique())
    results = []

    for i, test_month in enumerate(
        months[min_train_months + purge_months:],
        start=min_train_months + purge_months,
    ):
        train_end = months[i - purge_months - 1]

        # Fit HMM on train data only
        try:
            regime_df = fit_and_predict_regime(
                market_features, train_end=train_end
            )
        except ValueError as e:
            if verbose:
                print(f"  fold {i}: HMM skip — {e}")
            continue

        df_tr_raw = df[df[DATE_COL] <= train_end].copy()
        df_te_raw = df[df[DATE_COL] == test_month].copy()
        if df_te_raw.empty:
            continue

        df_tr = merge_regime_into_panel(df_tr_raw, regime_df)
        df_te = merge_regime_into_panel(df_te_raw, regime_df)

        available_regime = [c for c in regime_cols if c in df_tr.columns]
        if not available_regime:
            continue

        # Build tensors with enriched gate
        (
            X_tr, G_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            x_scaler, g_scaler, vol_mean, vol_std,
        ) = make_tensors_enhanced(
            df_tr, feature_cols=feature_cols,
            regime_cols=available_regime, gate_macro_cols=gate_macro_cols,
        )

        (
            X_te, G_te, y_ret_te, y_3m_te, y_vol_te_std,
            _, _, _, _,
        ) = make_tensors_enhanced(
            df_te, feature_cols=feature_cols,
            regime_cols=available_regime, gate_macro_cols=gate_macro_cols,
            x_scaler=x_scaler, g_scaler=g_scaler,
            vol_mean=vol_mean, vol_std=vol_std,
        )

        model, learned_weights, learned_log_vars = train_one_fold(
            X_tr, G_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            active_tasks=active_tasks,
            n_experts=n_experts,
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, dropout=dropout,
            patience=patience, device=device,
            fold_idx=i,  # audit 2026-04-24
        )

        model.eval()
        with torch.no_grad():
            pred_dict = model(X_te, G_te)

        y_pred_ret = pred_dict["ret"].numpy()

        out = pd.DataFrame({
            DATE_COL: test_month,
            STOCK_COL: df_te_raw[STOCK_COL].values,
            SECTOR_COL: df_te_raw[SECTOR_COL].values,
            "y_true": y_ret_te.numpy(),
            "y_pred": y_pred_ret,
            "fold": i,
            "realized_vol_true": df_te_raw[FWD_VOL_COL].values.astype(float),
            "current_realized_vol": df_te_raw[VOL_COL].values.astype(float),
            "fwd_ret_3m_true": df_te_raw[RET3M_COL].values.astype(float),
        })

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
            gate_msg = ", ".join(f"g{k}={out[f'gate_w_{k}'].mean():.3f}"
                                for k in range(gate_w.shape[1]))
            print(
                f"fold {i:3d} | {str(test_month)[:7]} | "
                f"n_train={len(df_tr):6,d} | IC={ic:+.4f} | "
                f"UW: {weight_msg} | Gate: {gate_msg}"
            )

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ── Metrics ─────────────────────────────────────────────────────────────


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
    feature_cols: list[str],
    n_experts: int = DEFAULT_N_EXPERTS,
    device: str = "cpu",
    verbose: bool = True,
    save_results: bool = True,
    output_dir: str = "output",
    suffix: str = "enhanced_moe",
) -> pd.DataFrame:
    if save_results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []

    for variant, tasks in ABLATION_TASKS.items():
        print(f"\n{'=' * 70}")
        print(f"Enhanced MoE {variant}: tasks={tasks} | "
              f"features={len(feature_cols)} | experts={n_experts}")
        print(f"{'=' * 70}")

        results = walk_forward_evaluate(
            df=df,
            market_features=market_features,
            active_tasks=tasks,
            feature_cols=feature_cols,
            n_experts=n_experts,
            device=device,
            verbose=verbose,
        )

        if results.empty:
            print(f"  {variant}: no results")
            continue

        summary_row = summarise(results, label=variant)
        rows.append(summary_row)

        if save_results:
            path = Path(output_dir) / f"results_{variant}_{suffix}.parquet"
            results.to_parquet(path, index=False)

        print(f"\nSummary ({variant}):")
        for k, v in summary_row.items():
            print(f"  {k}: {v}")

    summary_df = pd.DataFrame(rows)

    if save_results:
        summary_df.to_csv(
            Path(output_dir) / f"ablation_summary_{suffix}.csv", index=False
        )

    return summary_df


# ── Main ────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Regime MoE: pruned features + enriched gate + interactions"
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

    # Add interaction features
    df, ixn_cols = add_interaction_features(df)
    feature_cols = STOCK_FEATURES + ixn_cols
    print(f"Stock features ({len(feature_cols)}): {feature_cols}")

    # Sanity check
    needed = ([DATE_COL, STOCK_COL, SECTOR_COL, TARGET_COL, RET3M_COL,
               VOL_COL, FWD_VOL_COL] + feature_cols + GATE_MACRO_COLS)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[TARGET_COL, VOL_COL])

    # Sanitize
    for c in feature_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in GATE_MACRO_COLS:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    print(f"Tickers: {df[STOCK_COL].nunique():,}")
    print(f"Months: {df[DATE_COL].nunique():,}")

    print("\nBuilding market features for HMM...")
    market_features = build_market_monthly_features()
    print(f"Market features: {len(market_features)} months")

    print(f"\n--- Architecture ---")
    print(f"  Stock encoder: {len(feature_cols)} → 32 → 16")
    print(f"  Gate input: {N_REGIMES} regime + {len(GATE_MACRO_COLS)} macro = "
          f"{N_REGIMES + len(GATE_MACRO_COLS)} dims")
    print(f"  Experts: {args.n_experts} per task")

    summary_df = run_all_ablations(
        df=df,
        market_features=market_features,
        feature_cols=feature_cols,
        n_experts=args.n_experts,
        device=args.device,
        verbose=not args.quiet,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )

    print(f"\n{'=' * 90}")
    print("ENHANCED REGIME MoE ABLATION SUMMARY")
    print(f"{'=' * 90}")
    print(summary_df.to_string(index=False))

    # Parameter count comparison
    baseline_params = (14*64 + 64 + 64*32 + 32) + (3*16 + 16 + 16*3 + 3) + 3*(32*1+1)*3
    enhanced_params = (9*32 + 32 + 32*16 + 16) + (6*12 + 12 + 12*3 + 3) + 3*(16*1+1)*3
    print(f"\nParameter count: baseline={baseline_params}, enhanced={enhanced_params} "
          f"({enhanced_params/baseline_params:.0%} of baseline)")


if __name__ == "__main__":
    main()
