"""
Rung 5: MTL MLP with Uncertainty Weighting
Architecture  : Dense(64)+ReLU+Dropout -> Dense(32)+ReLU+Dropout (shared)
               -> task-specific heads (1 linear layer each)

Tasks
-----
  ret   - next-month cross-sectional return      [relative MSE]
  ret3m - 3-month forward return                 [relative MSE, NaN-masked]
  vol   - realized volatility                    [relative MSE on fold-standardized vol]

Changes from v4
---------------
* Replaces fixed equal lambda weighting with learned homoscedastic uncertainty weighting:
      L = sum_k [ 0.5 * exp(-s_k) * L_k + 0.5 * s_k ]
  where s_k = log(sigma_k^2) is trainable.

Motivation
----------
Equal weighting across active tasks can be too rigid. If one auxiliary task is noisy
or only weakly aligned with the primary return signal, forcing equal contribution may
hurt the shared representation. Learned uncertainty weighting allows the model to
downweight noisier tasks automatically while preserving useful auxiliary supervision.

Notes
-----
* fwd_ret_3m retains NaN masking exactly as before.
* The loss object has trainable parameters, so its parameters are optimized jointly
  with the model.
* Learned task weights are stored per fold for later diagnostics.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # audit 2026-04-24: for smooth_l1_loss (Huber)
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# 1. COLUMN NAMES — imported from shared config
from config import (
    ALL_FEATURE_COLS_V1, ALL_FEATURE_COLS_V2, ALL_FEATURE_COLS_V3_WITH_MISS,
    TARGET_COL, RET3M_COL, VOL_COL, FWD_VOL_COL, SECTOR_COL, DATE_COL, STOCK_COL,
)
from metrics import compute_long_short_sharpe  # Category B consolidation

# Default to V2; overridden by --v1 / --v3 flags
FACTOR_COLS = ALL_FEATURE_COLS_V2

# 2. DATA HELPERS

def make_tensors(
    df: pd.DataFrame,
    x_scaler: StandardScaler | None = None,
    vol_mean: float | None = None,
    vol_std: float | None = None,
):
    """
    Returns:
      X, y_ret, y_ret3m, y_vol_std,
      fitted_x_scaler, fitted_vol_mean, fitted_vol_std

    y_ret3m may contain NaNs (last 3 months of each training window).
    The loss function masks them; callers should not dropna before passing in.
    """
    X_raw = np.nan_to_num(df[FACTOR_COLS].values.astype(np.float32), nan=0.0)

    if x_scaler is None:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X_raw)
    else:
        X = x_scaler.transform(X_raw)

    y_ret = df[TARGET_COL].values.astype(np.float32)
    y_ret3m_raw = df[RET3M_COL].values.astype(np.float64)  # preserve NaN

    # vol target: NEXT month's realized vol (FWD_VOL_COL); may contain NaN for
    # the last observation of each ticker — masked in UncertaintyMTLLoss.
    # Standardize using current-period realized_vol statistics (VOL_COL) so that
    # vol_mean/vol_std are stable and not contaminated by NaN shift.
    y_vol_ref = df[VOL_COL].values.astype(np.float32)          # for fitting scaler
    y_vol_raw = df[FWD_VOL_COL].values.astype(np.float64)      # actual target (NaN ok)

    if vol_mean is None or vol_std is None:
        vol_mean = float(np.nanmean(y_vol_ref))
        vol_std = float(np.nanstd(y_vol_ref))
        vol_std = max(vol_std, 1e-8)

    # Standardize fwd vol; NaN rows stay NaN (masked by loss)
    y_vol_std = ((y_vol_raw - vol_mean) / vol_std).astype(np.float32)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_ret, dtype=torch.float32),
        torch.tensor(y_ret3m_raw, dtype=torch.float32),
        torch.tensor(y_vol_std, dtype=torch.float32),
        x_scaler,
        vol_mean,
        vol_std,
    )

# 3. MODEL

class MTLNet(nn.Module):
    def __init__(
        self,
        n_factors: int = 15,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.10,
        active_tasks: set | None = None,
    ):
        super().__init__()

        if active_tasks is None:
            active_tasks = {"ret", "ret3m", "vol"}
        self.active_tasks = set(active_tasks)

        self.shared = nn.Sequential(
            nn.Linear(n_factors, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_ret = nn.Linear(hidden2, 1)
        self.head_ret3m = nn.Linear(hidden2, 1)
        self.head_vol = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(x)
        out = {}

        if "ret" in self.active_tasks:
            out["ret"] = self.head_ret(z).squeeze(-1)
        if "ret3m" in self.active_tasks:
            out["ret3m"] = self.head_ret3m(z).squeeze(-1)
        if "vol" in self.active_tasks:
            out["vol"] = self.head_vol(z).squeeze(-1)

        return out


# 4. UNCERTAINTY-WEIGHTED LOSS

class UncertaintyMTLLoss(nn.Module):
    """
    Multi-task loss with learned homoscedastic uncertainty.

    For each active task k:
        0.5 * exp(-s_k) * L_k + 0.5 * s_k
    where s_k = log(sigma_k^2) is a trainable scalar.

    Base losses are relative MSEs normalized by train-fold variance.

    fwd_ret_3m rows with NaN are excluded from the ret3m loss.
    """

    def __init__(
        self,
        active_tasks: set,
        var_ret: float = 1.0,
        var_ret3m: float = 1.0,
        var_vol: float = 1.0,
        # audit 2026-04-24: Huber β per task, auto-scaled to ~0.5 × std(y)
        # Fixes previous β=0.01 which made Huber degenerate to MAE for return std≈0.08
        huber_beta_ret: float = 0.04,
        huber_beta_ret3m: float = 0.08,
        huber_beta_vol: float = 0.5,  # vol is z-scored internally, unit-variance
    ):
        super().__init__()

        self.active_tasks = set(active_tasks)
        self.var_ret = max(var_ret, 1e-8)
        self.var_ret3m = max(var_ret3m, 1e-8)
        self.var_vol = max(var_vol, 1e-8)
        self.huber_beta_ret = huber_beta_ret
        self.huber_beta_ret3m = huber_beta_ret3m
        self.huber_beta_vol = huber_beta_vol

        self.log_vars = nn.ParameterDict()
        if "ret" in self.active_tasks:
            self.log_vars["ret"] = nn.Parameter(torch.zeros(1))
        if "ret3m" in self.active_tasks:
            self.log_vars["ret3m"] = nn.Parameter(torch.zeros(1))
        if "vol" in self.active_tasks:
            self.log_vars["vol"] = nn.Parameter(torch.zeros(1))

    @staticmethod
    def from_train_data(
        y_ret: torch.Tensor,
        y_ret3m: torch.Tensor,
        y_vol_std: torch.Tensor,
        active_tasks: set,
    ) -> "UncertaintyMTLLoss":
        valid_3m = y_ret3m[~torch.isnan(y_ret3m)]
        var_ret3m = float(valid_3m.var()) if len(valid_3m) > 1 else 1.0

        valid_vol = y_vol_std[~torch.isnan(y_vol_std)]
        var_vol = float(valid_vol.var()) if len(valid_vol) > 1 else 1.0

        # audit 2026-04-24: Huber β auto-scaled to 0.5 × std(y)
        # Covers ~1 std deviation; keeps bulk of observations in quadratic region
        std_ret = float(y_ret.std())
        std_3m = float(valid_3m.std()) if len(valid_3m) > 1 else 1.0
        std_vol = float(valid_vol.std()) if len(valid_vol) > 1 else 1.0

        return UncertaintyMTLLoss(
            active_tasks=active_tasks,
            var_ret=float(y_ret.var()),
            var_ret3m=var_ret3m,
            var_vol=var_vol,
            huber_beta_ret=max(0.01, 0.5 * std_ret),
            huber_beta_ret3m=max(0.01, 0.5 * std_3m),
            huber_beta_vol=max(0.1, 0.5 * std_vol),
        )

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        y_ret: torch.Tensor,
        y_ret3m: torch.Tensor,
        y_vol_std: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.zeros(1, device=y_ret.device).squeeze()

        # audit 2026-04-24: Huber β is per-task, auto-scaled via from_train_data
        # (fixed previous mis-scaling at β=0.01 → retained signal magnitude learning)
        if "ret" in preds:
            base_ret = F.smooth_l1_loss(preds["ret"], y_ret, beta=self.huber_beta_ret) / self.var_ret
            s = self.log_vars["ret"].squeeze()
            total = total + 0.5 * torch.exp(-s) * base_ret + 0.5 * s

        if "ret3m" in preds:
            valid = ~torch.isnan(y_ret3m)
            if valid.sum() > 0:
                base_ret3m = F.smooth_l1_loss(preds["ret3m"][valid], y_ret3m[valid], beta=self.huber_beta_ret3m) / self.var_ret3m
                s = self.log_vars["ret3m"].squeeze()
                total = total + 0.5 * torch.exp(-s) * base_ret3m + 0.5 * s

        if "vol" in preds:
            valid_vol = ~torch.isnan(y_vol_std)
            if valid_vol.sum() > 0:
                base_vol = F.smooth_l1_loss(preds["vol"][valid_vol], y_vol_std[valid_vol], beta=self.huber_beta_vol) / self.var_vol
                s = self.log_vars["vol"].squeeze()
                total = total + 0.5 * torch.exp(-s) * base_vol + 0.5 * s

        return total

    def learned_task_weights(self) -> dict[str, float]:
        """
        Returns exp(-s_k), which acts like an inverse variance / precision weight.
        """
        out = {}
        for k, v in self.log_vars.items():
            out[k] = float(torch.exp(-v.detach().cpu()).item())
        return out

    def learned_log_vars(self) -> dict[str, float]:
        out = {}
        for k, v in self.log_vars.items():
            out[k] = float(v.detach().cpu().item())
        return out

# 5. TRAIN ONE FOLD
# Category D audit note: NOT consolidated with regmtl.py / regmtl_enhanced.py /
# rung5_planned.py / rung5_combined.py — each uses a different model class:
#   main.py          → MTLNet(active_tasks),        inputs: (X, y_ret, y_ret3m, y_vol)
#   regmtl.py        → RegimeGatedMTLMoE(n_experts), inputs: (X, R, y_ret, y_ret3m, y_vol)
#   regmtl_enhanced.py → EnhancedRegimeMoE,          inputs: (X, G, y_ret, y_ret3m, y_vol)
#   rung5_planned.py → MTLNet(active_tasks),         inputs: (X, y_ret, y_rank, y_sector_rel)
#   rung5_combined.py → MTLNet(active_tasks),        inputs: (X, train_targets: dict)
# Consolidation would require a model-factory pattern — unsafe 1 day from submission.

def train_one_fold(
    X_tr: torch.Tensor,
    y_ret_tr: torch.Tensor,
    y_ret3m_tr: torch.Tensor,
    y_vol_tr: torch.Tensor,
    active_tasks: set,
    epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.10,
    patience: int = 20,
    val_frac: float = 0.10,
    device: str = "cpu",
    fold_idx: int = 0,  # audit 2026-04-24: per-fold seed for reproducibility
) -> tuple[MTLNet, dict[str, float], dict[str, float]]:
    # audit 2026-04-24: set seeds per fold for reproducibility (across-runs)
    # while preserving cross-fold variation for diagnostics
    torch.manual_seed(fold_idx)
    np.random.seed(fold_idx)

    n_val = max(1, int(len(X_tr) * val_frac))

    X_val, X_tr = X_tr[-n_val:], X_tr[:-n_val]
    y_ret_val, y_ret_tr = y_ret_tr[-n_val:], y_ret_tr[:-n_val]
    y_3m_val, y_3m_tr = y_ret3m_tr[-n_val:], y_ret3m_tr[:-n_val]
    y_vol_val, y_vol_tr = y_vol_tr[-n_val:], y_vol_tr[:-n_val]

    criterion = UncertaintyMTLLoss.from_train_data(
        y_ret_tr, y_3m_tr, y_vol_tr, active_tasks
    ).to(device)

    train_dl = DataLoader(
        TensorDataset(X_tr, y_ret_tr, y_3m_tr, y_vol_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    model = MTLNet(
        n_factors=X_tr.shape[1],
        dropout=dropout,
        active_tasks=active_tasks,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=weight_decay,
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

        for Xb, yr, y3m, yv in train_dl:
            Xb, yr, y3m, yv = (t.to(device) for t in (Xb, yr, y3m, yv))

            optimizer.zero_grad()
            loss = criterion(model(Xb), yr, y3m, yv)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        criterion.eval()

        with torch.no_grad():
            val_loss = criterion(
                model(X_val.to(device)),
                y_ret_val.to(device),
                y_3m_val.to(device),
                y_vol_val.to(device),
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

    learned_weights = criterion.learned_task_weights()
    learned_log_vars = criterion.learned_log_vars()

    return model.to("cpu"), learned_weights, learned_log_vars


# 6. WALK-FORWARD

ABLATION_TASKS = {
    "rung4": {"ret"},  # audit-fix 2026-04-22: was "5a"; architecturally pure single-task MLP = Rung 4 of ladder
    "5b": {"ret", "ret3m"},
    "5c": {"ret", "vol"},
    "5d": {"ret", "ret3m", "vol"},
}


# Category E audit note: walk_forward_evaluate NOT consolidated across files.
# Each version is coupled to a different train_one_fold / model class:
#   main.py          → calls local train_one_fold (MTLNet, 3 positional targets)
#   regmtl.py        → calls local train_one_fold (RegimeGatedMoE, HMM per fold)
#   regmtl_enhanced.py → calls local train_one_fold (EnhancedRegimeMoE, gate features)
#   rung5_planned.py → calls local train_one_fold (MTLNet, rank/sector_rel heads)
#   rung5_combined.py → calls local train_one_fold (MTLNet, dict-based targets)
# A harness abstraction would require model-factory + callable injection — unsafe pre-submission.
def walk_forward_evaluate(
    df: pd.DataFrame,
    active_tasks: set,
    min_train_months: int = 60,
    purge_months: int = 1,
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
        df_tr = df[df[DATE_COL] <= train_end].copy()
        df_te = df[df[DATE_COL] == test_month].copy()

        if df_te.empty:
            continue

        (
            X_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            x_scaler, vol_mean, vol_std,
        ) = make_tensors(df_tr)

        (
            X_te, y_ret_te, y_3m_te, y_vol_te_std,
            _, _, _,
        ) = make_tensors(
            df_te,
            x_scaler=x_scaler,
            vol_mean=vol_mean,
            vol_std=vol_std,
        )

        model, learned_weights, learned_log_vars = train_one_fold(
            X_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            active_tasks=active_tasks,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            patience=patience,
            device=device,
            fold_idx=i,  # audit 2026-04-24: seed per fold for reproducibility
        )

        model.eval()
        with torch.no_grad():
            pred_dict = model(X_te)

        y_pred_ret = pred_dict["ret"].numpy()

        out = pd.DataFrame({
            DATE_COL: test_month,
            STOCK_COL: df_te[STOCK_COL].values,
            "sector": df_te[SECTOR_COL].values,
            "y_true": y_ret_te.numpy(),
            "y_pred": y_pred_ret,
            "fold": i,
            "realized_vol_true": df_te[FWD_VOL_COL].values.astype(float),  # next-month vol (true target)
            "current_realized_vol": df_te[VOL_COL].values.astype(float),    # current-month vol (reference)
            "fwd_ret_3m_true": df_te[RET3M_COL].values.astype(float),
        })

        if "ret3m" in pred_dict:
            out["ret3m_pred"] = pred_dict["ret3m"].numpy()

        if "vol" in pred_dict:
            vol_pred_std = pred_dict["vol"].numpy()
            out["vol_pred"] = vol_pred_std * vol_std + vol_mean

        # store learned uncertainty weights on every row for easy aggregation later
        for task_name in ["ret", "ret3m", "vol"]:
            if task_name in learned_weights:
                out[f"uw_weight_{task_name}"] = learned_weights[task_name]
                out[f"uw_logvar_{task_name}"] = learned_log_vars[task_name]

        results.append(out)

        if verbose:
            ic = spearmanr(out["y_true"], out["y_pred"]).statistic
            weight_msg = ", ".join(
                [f"{k}={v:.3f}" for k, v in learned_weights.items()]
            )
            print(
                f"fold {i:3d} | {str(test_month)[:7]} | "
                f"n_train={len(df_tr):6,d} | IC={ic:+.4f} | weights: {weight_msg}"
            )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# 7. METRICS / SUMMARY HELPERS

def compute_monthly_ic(results: pd.DataFrame) -> pd.Series:
    monthly = results.groupby(DATE_COL).apply(
        lambda g: spearmanr(g["y_true"], g["y_pred"]).statistic if len(g) > 5 else np.nan
    )
    return monthly.dropna()


# compute_long_short_sharpe removed — imported from metrics.py (Category B consolidation)


def compute_ret3m_auxiliary_ic(results: pd.DataFrame) -> dict:
    """IC of the 3m auxiliary head — diagnostic only."""
    if "ret3m_pred" not in results.columns:
        return {"ret3m_ic_mean": np.nan, "ret3m_ic_std": np.nan}

    sub = results.dropna(subset=["fwd_ret_3m_true", "ret3m_pred"])
    if len(sub) == 0:
        return {"ret3m_ic_mean": np.nan, "ret3m_ic_std": np.nan}

    monthly = sub.groupby(DATE_COL).apply(
        lambda g: spearmanr(g["fwd_ret_3m_true"], g["ret3m_pred"]).statistic if len(g) > 5 else np.nan
    ).dropna()

    return {
        "ret3m_ic_mean": round(float(monthly.mean()), 4) if len(monthly) else np.nan,
        "ret3m_ic_std": round(float(monthly.std()), 4) if len(monthly) else np.nan,
    }


def compute_vol_auxiliary_corr(results: pd.DataFrame) -> float:
    """Pearson corr of vol head predictions vs true realized vol — diagnostic."""
    if "vol_pred" not in results.columns:
        return np.nan

    sub = results.dropna(subset=["realized_vol_true", "vol_pred"])
    if len(sub) == 0:
        return np.nan

    return round(float(sub["realized_vol_true"].corr(sub["vol_pred"])), 4)


def summarise(results: pd.DataFrame, label: str = "") -> dict:
    ic = compute_monthly_ic(results)

    if len(ic) >= 2 and ic.std() > 1e-8:
        t_val = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)) + 1e-8)
    else:
        t_val = np.nan

    aux3m = compute_ret3m_auxiliary_ic(results)
    vol_corr = compute_vol_auxiliary_corr(results)
    pred_std_ratio = float(results["y_pred"].std() / (results["y_true"].std() + 1e-8))

    row = {
        "label": label,
        "IC_mean": round(float(ic.mean()), 4) if len(ic) else np.nan,
        "IC_std": round(float(ic.std(ddof=1)), 4) if len(ic) > 1 else np.nan,
        "IC_t": round(float(t_val), 3) if pd.notna(t_val) else np.nan,
        "IC_pos_frac": round(float((ic > 0).mean()), 3) if len(ic) else np.nan,
        "LS_Sharpe": round(compute_long_short_sharpe(results), 3),
        "ret3m_ic_mean": aux3m["ret3m_ic_mean"],
        "ret3m_ic_std": aux3m["ret3m_ic_std"],
        "vol_corr": vol_corr,
        "pred_std_ratio": round(pred_std_ratio, 4),
        "n_months": int(len(ic)),
    }

    # learned uncertainty weights averaged over folds
    for task_name in ["ret", "ret3m", "vol"]:
        weight_col = f"uw_weight_{task_name}"
        logvar_col = f"uw_logvar_{task_name}"
        if weight_col in results.columns:
            row[f"uw_{task_name}_mean"] = round(float(results[weight_col].mean()), 4)
            row[f"uw_{task_name}_std"] = round(float(results[weight_col].std(ddof=1)), 4) if results[weight_col].nunique() > 1 else 0.0
        else:
            row[f"uw_{task_name}_mean"] = np.nan
            row[f"uw_{task_name}_std"] = np.nan

        if logvar_col in results.columns:
            row[f"logvar_{task_name}_mean"] = round(float(results[logvar_col].mean()), 4)
        else:
            row[f"logvar_{task_name}_mean"] = np.nan

    return row

# 8. RUN ALL ABLATIONS

def run_all_ablations(
    df: pd.DataFrame,
    device: str = "cpu",
    verbose: bool = True,
    save_results: bool = True,
    output_dir: str = "output",
) -> pd.DataFrame:
    if save_results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []

    for variant, tasks in ABLATION_TASKS.items():
        print(f"\n{'=' * 60}")
        print(f"Ablation {variant}: tasks = {tasks}")
        print(f"{'=' * 60}")

        results = walk_forward_evaluate(
            df=df,
            active_tasks=tasks,
            device=device,
            verbose=verbose,
        )

        summary_row = summarise(results, label=variant)
        rows.append(summary_row)

        if save_results:
            results_path = Path(output_dir) / f"results_{variant}_uw.parquet"
            results.to_parquet(results_path, index=False)

        print("\nSummary:")
        for k, v in summary_row.items():
            print(f"  {k}: {v}")

    summary_df = pd.DataFrame(rows)

    if save_results:
        summary_path = Path(output_dir) / "ablation_summary_uncertainty.csv"
        summary_df.to_csv(summary_path, index=False)

    return summary_df

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rung 5 MTL MLP with uncertainty weighting")
    parser.add_argument("--data", type=str, default="data/master_panel_v2.parquet", help="Path to master panel parquet. Default is V2 (CRSP-based panel). V1 (master_panel.parquet) has known dei-namespace gaps for V/BRK-B/STZ — use only for legacy comparison.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--quiet", action="store_true", help="Suppress fold-level logging")
    parser.add_argument("--no_save", action="store_true", help="Do not save result files")
    parser.add_argument("--v1", action="store_true", help="Use V1 features (15) instead of V2 (25)")
    parser.add_argument("--v3", action="store_true", help="Use V3 features + missingness flags (22 stock + 5 = 27)")
    args = parser.parse_args()

    global FACTOR_COLS
    if args.v1:
        FACTOR_COLS = ALL_FEATURE_COLS_V1
    elif args.v3:
        FACTOR_COLS = ALL_FEATURE_COLS_V3_WITH_MISS

    print("Loading panel...")
    df = pd.read_parquet(args.data)

    # minimal sanity check
    needed = [DATE_COL, STOCK_COL, SECTOR_COL, TARGET_COL, RET3M_COL, VOL_COL, FWD_VOL_COL] + FACTOR_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"Unique tickers: {df[STOCK_COL].nunique():,}")
    print(f"Unique months: {df[DATE_COL].nunique():,}")

    summary_df = run_all_ablations(
        df=df,
        device=args.device,
        verbose=not args.quiet,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("UNCERTAINTY-WEIGHTED ABLATION SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()