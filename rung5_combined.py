"""
Rung 5 (Combined): MTL MLP — Best Auxiliary Target Combinations
================================================================

Architecture  : Dense(64)+ReLU+Dropout -> Dense(32)+ReLU+Dropout (shared)
               -> task-specific heads (1 linear layer each)

Motivation
----------
Two prior MTL experiments found complementary strengths:

  RG's version (main.py):     ret + ret3m + vol   -> IC=0.0086, Sharpe=0.579
  Planned version:            ret + rank + sector_rel -> IC=0.0174, Sharpe=0.313

Rank as aux target produced the highest IC; vol produced the best Sharpe.
This script tests all interesting combinations of the 4 available auxiliary
targets to find the best overall MTL configuration.

Ablation configs
-----------------
  combo_a : ret + rank                        (best single aux from planned)
  combo_b : ret + rank + vol                  (best IC aux + best Sharpe aux)
  combo_c : ret + rank + ret3m                (best IC aux + RG's other aux)
  combo_d : ret + rank + vol + ret3m          (3 auxiliary targets)
  combo_e : ret + rank + vol + ret3m + sector_rel  (full 5-task MTL)

All hyperparameters match main.py exactly:
  - Adam lr=1e-3, weight_decay=1e-5
  - ReduceLROnPlateau patience=8, factor=0.5
  - Early stopping patience=20
  - Uncertainty weighting (Kendall et al. 2018)
  - Walk-forward: 60-month min train, 1-month purge
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

# ──────────────────────────────────────────────────────────────────────
# 1. COLUMN NAMES
# ──────────────────────────────────────────────────────────────────────

FACTOR_ZSCORE_COLS = [
    "EarningsYield_zscore",
    "GrossProfitability_zscore",
    "AssetGrowth_zscore",
    "Accruals_zscore",
    "Momentum12_1_zscore",
    "NetDebtEBITDA_zscore",
]

MACRO_COLS = [
    "T10Y2Y", "VIXCLS", "UMCSENT", "CFNAI", "UNRATE",
    "BAMLH0A0HYM2", "CPI_YOY", "VIX_TERM_STRUCTURE", "LEADING_COMPOSITE",
]

FACTOR_COLS = FACTOR_ZSCORE_COLS + MACRO_COLS
TARGET_COL = "fwd_ret_1m"
RET3M_COL = "fwd_ret_3m"
VOL_COL = "realized_vol"
SECTOR_COL = "sector"
DATE_COL = "date"
STOCK_COL = "ticker"

# Computed auxiliary target columns
RANK_COL = "rank_target"
SECTOR_REL_COL = "sector_rel_target"

# All possible task names (primary + 4 aux)
ALL_TASK_NAMES = ["ret", "rank", "ret3m", "vol", "sector_rel"]

# ──────────────────────────────────────────────────────────────────────
# 2. AUXILIARY TARGET CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────

def compute_auxiliary_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two computed auxiliary target columns to the panel:

    rank_target      : Within each month, rank stocks by fwd_ret_1m and
                       scale to [0, 1].  Rank 0 = worst, 1 = best.
    sector_rel_target: fwd_ret_1m minus the sector mean for that month.

    fwd_ret_3m and realized_vol are already in the data.
    """
    df = df.copy()

    # Cross-sectional rank scaled to [0, 1]
    df[RANK_COL] = df.groupby(DATE_COL)[TARGET_COL].rank(pct=True)

    # Sector-relative return
    sector_month_mean = df.groupby([DATE_COL, SECTOR_COL])[TARGET_COL].transform("mean")
    df[SECTOR_REL_COL] = df[TARGET_COL] - sector_month_mean

    return df


# ──────────────────────────────────────────────────────────────────────
# 3. DATA HELPERS
# ──────────────────────────────────────────────────────────────────────

def make_tensors(
    df: pd.DataFrame,
    x_scaler: StandardScaler | None = None,
    vol_mean: float | None = None,
    vol_std: float | None = None,
):
    """
    Returns:
      X, y_ret, y_ret3m, y_vol_std, y_rank, y_sector_rel,
      fitted_x_scaler, fitted_vol_mean, fitted_vol_std

    y_ret3m may contain NaNs (last 3 months of each training window).
    The loss function masks them.
    """
    X_raw = np.nan_to_num(df[FACTOR_COLS].values.astype(np.float32), nan=0.0)

    if x_scaler is None:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X_raw)
    else:
        X = x_scaler.transform(X_raw)

    y_ret = df[TARGET_COL].values.astype(np.float32)
    y_ret3m_raw = df[RET3M_COL].values.astype(np.float64)  # preserve NaN
    y_vol = df[VOL_COL].values.astype(np.float32)
    y_rank = df[RANK_COL].values.astype(np.float32)
    y_sector_rel = df[SECTOR_REL_COL].values.astype(np.float32)

    # Fold-standardise vol (same as main.py)
    if vol_mean is None or vol_std is None:
        vol_mean = float(np.nanmean(y_vol))
        vol_std = float(np.nanstd(y_vol))
        vol_std = max(vol_std, 1e-8)

    y_vol_std = ((y_vol - vol_mean) / vol_std).astype(np.float32)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_ret, dtype=torch.float32),
        torch.tensor(y_ret3m_raw, dtype=torch.float32),
        torch.tensor(y_vol_std, dtype=torch.float32),
        torch.tensor(y_rank, dtype=torch.float32),
        torch.tensor(y_sector_rel, dtype=torch.float32),
        x_scaler,
        vol_mean,
        vol_std,
    )


# ──────────────────────────────────────────────────────────────────────
# 4. MODEL
# ──────────────────────────────────────────────────────────────────────

class MTLNet(nn.Module):
    """
    Same shared trunk as main.py.  Up to 5 task heads:
      ret        — 1-month forward return (primary)
      ret3m      — 3-month forward return
      vol        — realized volatility (fold-standardised)
      rank       — cross-sectional rank [0,1]
      sector_rel — sector-relative return
    """
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
            active_tasks = {"ret"}
        self.active_tasks = set(active_tasks)

        self.shared = nn.Sequential(
            nn.Linear(n_factors, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Create heads only for active tasks
        self.heads = nn.ModuleDict()
        for task in self.active_tasks:
            self.heads[task] = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(x)
        out = {}
        for task in self.active_tasks:
            out[task] = self.heads[task](z).squeeze(-1)
        return out


# ──────────────────────────────────────────────────────────────────────
# 5. UNCERTAINTY-WEIGHTED LOSS
# ──────────────────────────────────────────────────────────────────────

class UncertaintyMTLLoss(nn.Module):
    """
    Multi-task loss with learned homoscedastic uncertainty.

    For each active task k:
        0.5 * exp(-s_k) * L_k + 0.5 * s_k
    where s_k = log(sigma_k^2) is a trainable scalar.

    Base losses are relative MSEs normalized by train-fold variance.

    fwd_ret_3m rows with NaN are excluded from the ret3m loss.
    realized_vol rows with NaN are excluded from the vol loss.
    """

    def __init__(
        self,
        active_tasks: set,
        variances: dict[str, float],
    ):
        super().__init__()

        self.active_tasks = set(active_tasks)
        self.variances = {k: max(v, 1e-8) for k, v in variances.items()}

        self.log_vars = nn.ParameterDict()
        for task in self.active_tasks:
            self.log_vars[task] = nn.Parameter(torch.zeros(1))

    @staticmethod
    def from_train_data(
        targets: dict[str, torch.Tensor],
        active_tasks: set,
    ) -> "UncertaintyMTLLoss":
        variances = {}
        for task in active_tasks:
            t = targets[task]
            valid = t[~torch.isnan(t)]
            variances[task] = float(valid.var()) if len(valid) > 1 else 1.0

        return UncertaintyMTLLoss(
            active_tasks=active_tasks,
            variances=variances,
        )

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = torch.zeros(1, device=next(iter(preds.values())).device).squeeze()

        for task in self.active_tasks:
            if task not in preds:
                continue

            pred = preds[task]
            target = targets[task]

            # Mask NaN targets (fwd_ret_3m has NaN in last 3 months, vol may have NaN)
            valid = ~torch.isnan(target)
            if valid.sum() == 0:
                continue

            base_loss = ((pred[valid] - target[valid]) ** 2).mean() / self.variances[task]
            s = self.log_vars[task].squeeze()
            total = total + 0.5 * torch.exp(-s) * base_loss + 0.5 * s

        return total

    def learned_task_weights(self) -> dict[str, float]:
        """Returns exp(-s_k), the inverse variance / precision weight."""
        out = {}
        for k, v in self.log_vars.items():
            out[k] = float(torch.exp(-v.detach().cpu()).item())
        return out

    def learned_log_vars(self) -> dict[str, float]:
        out = {}
        for k, v in self.log_vars.items():
            out[k] = float(v.detach().cpu().item())
        return out


# ──────────────────────────────────────────────────────────────────────
# 6. TRAIN ONE FOLD
# ──────────────────────────────────────────────────────────────────────

def train_one_fold(
    X_tr: torch.Tensor,
    train_targets: dict[str, torch.Tensor],
    active_tasks: set,
    epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.10,
    patience: int = 20,
    val_frac: float = 0.10,
    device: str = "cpu",
) -> tuple[MTLNet, dict[str, float], dict[str, float]]:
    n = len(X_tr)
    n_val = max(1, int(n * val_frac))

    # Split: last n_val rows for validation (temporal split)
    X_val = X_tr[-n_val:]
    X_tr_split = X_tr[:-n_val]

    val_targets = {}
    tr_targets = {}
    for task, tensor in train_targets.items():
        val_targets[task] = tensor[-n_val:]
        tr_targets[task] = tensor[:-n_val]

    criterion = UncertaintyMTLLoss.from_train_data(
        tr_targets, active_tasks
    ).to(device)

    # Build DataLoader: pack all targets into a single TensorDataset
    # Order: X, then targets in ALL_TASK_NAMES order (for consistent unpacking)
    task_order = [t for t in ALL_TASK_NAMES if t in active_tasks]
    train_tensors = [X_tr_split] + [tr_targets[t] for t in task_order]
    train_dl = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=batch_size,
        shuffle=True,
    )

    model = MTLNet(
        n_factors=X_tr_split.shape[1],
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

        for batch in train_dl:
            batch = [t.to(device) for t in batch]
            Xb = batch[0]
            batch_targets = {task_order[i]: batch[i + 1] for i in range(len(task_order))}

            optimizer.zero_grad()
            loss = criterion(model(Xb), batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        criterion.eval()

        with torch.no_grad():
            val_targets_device = {k: v.to(device) for k, v in val_targets.items()}
            val_loss = criterion(
                model(X_val.to(device)),
                val_targets_device,
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


# ──────────────────────────────────────────────────────────────────────
# 7. WALK-FORWARD
# ──────────────────────────────────────────────────────────────────────

ABLATION_TASKS = {
    "combo_a": {"ret", "rank"},
    "combo_b": {"ret", "rank", "vol"},
    "combo_c": {"ret", "rank", "ret3m"},
    "combo_d": {"ret", "rank", "vol", "ret3m"},
    "combo_e": {"ret", "rank", "vol", "ret3m", "sector_rel"},
}


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
            X_tr, y_ret_tr, y_ret3m_tr, y_vol_tr, y_rank_tr, y_sr_tr,
            x_scaler, vol_mean, vol_std,
        ) = make_tensors(df_tr)

        (
            X_te, y_ret_te, y_ret3m_te, y_vol_te, y_rank_te, y_sr_te,
            _, _, _,
        ) = make_tensors(df_te, x_scaler=x_scaler, vol_mean=vol_mean, vol_std=vol_std)

        # Build target dicts for active tasks only
        all_train_targets = {
            "ret": y_ret_tr,
            "ret3m": y_ret3m_tr,
            "vol": y_vol_tr,
            "rank": y_rank_tr,
            "sector_rel": y_sr_tr,
        }
        train_targets = {k: v for k, v in all_train_targets.items() if k in active_tasks}

        model, learned_weights, learned_log_vars = train_one_fold(
            X_tr, train_targets,
            active_tasks=active_tasks,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            patience=patience,
            device=device,
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
            "realized_vol_true": df_te[VOL_COL].values.astype(float),
            "fwd_ret_3m_true": df_te[RET3M_COL].values.astype(float),
            "rank_true": y_rank_te.numpy(),
            "sector_rel_true": y_sr_te.numpy(),
        })

        # Store auxiliary predictions
        for task in ["ret3m", "vol", "rank", "sector_rel"]:
            if task in pred_dict:
                if task == "vol":
                    # Un-standardise vol predictions
                    out["vol_pred"] = pred_dict["vol"].numpy() * vol_std + vol_mean
                else:
                    out[f"{task}_pred"] = pred_dict[task].numpy()

        # Store learned uncertainty weights on every row for easy aggregation
        for task_name in active_tasks:
            if task_name in learned_weights:
                out[f"uw_weight_{task_name}"] = learned_weights[task_name]
                out[f"uw_logvar_{task_name}"] = learned_log_vars[task_name]

        results.append(out)

        if verbose:
            ic = spearmanr(out["y_true"], out["y_pred"]).statistic
            weight_msg = ", ".join(
                [f"{k}={v:.3f}" for k, v in sorted(learned_weights.items())]
            )
            print(
                f"fold {i:3d} | {str(test_month)[:7]} | "
                f"n_train={len(df_tr):6,d} | IC={ic:+.4f} | weights: {weight_msg}"
            )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# 8. METRICS / SUMMARY HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_monthly_ic(results: pd.DataFrame) -> pd.Series:
    monthly = results.groupby(DATE_COL).apply(
        lambda g: spearmanr(g["y_true"], g["y_pred"]).statistic if len(g) > 5 else np.nan
    )
    return monthly.dropna()


def compute_long_short_sharpe(
    results: pd.DataFrame,
    top_q: float = 0.2,
    bottom_q: float = 0.2,
) -> float:
    """
    Simple equal-weight long-short portfolio formed within each month:
      long  = top q by predicted return
      short = bottom q by predicted return
    """
    monthly_rets = []

    for _, g in results.groupby(DATE_COL):
        g = g.dropna(subset=["y_pred", "y_true"]).copy()
        if len(g) < 10:
            continue

        n_long = max(1, int(len(g) * top_q))
        n_short = max(1, int(len(g) * bottom_q))

        g = g.sort_values("y_pred", ascending=False)
        long_ret = g.head(n_long)["y_true"].mean()
        short_ret = g.tail(n_short)["y_true"].mean()

        monthly_rets.append(long_ret - short_ret)

    if len(monthly_rets) < 2:
        return np.nan

    monthly_rets = np.asarray(monthly_rets, dtype=float)
    mean_ret = monthly_rets.mean()
    std_ret = monthly_rets.std(ddof=1)

    if std_ret < 1e-8:
        return np.nan

    return float(np.sqrt(12.0) * mean_ret / std_ret)


def compute_auxiliary_ic(results: pd.DataFrame, task: str) -> dict:
    """Spearman IC of an auxiliary head — diagnostic."""
    pred_col = f"{task}_pred"
    true_col_map = {
        "ret3m": "fwd_ret_3m_true",
        "vol": "realized_vol_true",
        "rank": "rank_true",
        "sector_rel": "sector_rel_true",
    }
    true_col = true_col_map.get(task)

    if pred_col not in results.columns or true_col not in results.columns:
        return {f"{task}_ic_mean": np.nan, f"{task}_ic_std": np.nan}

    # For vol, use Pearson correlation (continuous positive values)
    if task == "vol":
        sub = results.dropna(subset=[true_col, "vol_pred"])
        if len(sub) == 0:
            return {"vol_corr": np.nan}
        return {"vol_corr": round(float(sub[true_col].corr(sub["vol_pred"])), 4)}

    sub = results.dropna(subset=[true_col, pred_col])
    if len(sub) == 0:
        return {f"{task}_ic_mean": np.nan, f"{task}_ic_std": np.nan}

    monthly = sub.groupby(DATE_COL).apply(
        lambda g: spearmanr(g[true_col], g[pred_col]).statistic if len(g) > 5 else np.nan
    ).dropna()

    return {
        f"{task}_ic_mean": round(float(monthly.mean()), 4) if len(monthly) else np.nan,
        f"{task}_ic_std": round(float(monthly.std()), 4) if len(monthly) else np.nan,
    }


def summarise(results: pd.DataFrame, label: str = "", active_tasks: set = None) -> dict:
    ic = compute_monthly_ic(results)

    if len(ic) >= 2 and ic.std() > 1e-8:
        t_val = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)) + 1e-8)
    else:
        t_val = np.nan

    pred_std_ratio = float(results["y_pred"].std() / (results["y_true"].std() + 1e-8))

    row = {
        "label": label,
        "IC_mean": round(float(ic.mean()), 4) if len(ic) else np.nan,
        "IC_std": round(float(ic.std(ddof=1)), 4) if len(ic) > 1 else np.nan,
        "IC_t": round(float(t_val), 3) if pd.notna(t_val) else np.nan,
        "IC_pos_frac": round(float((ic > 0).mean()), 3) if len(ic) else np.nan,
        "LS_Sharpe": round(compute_long_short_sharpe(results), 3),
        "pred_std_ratio": round(pred_std_ratio, 4),
        "n_months": int(len(ic)),
    }

    # Auxiliary head diagnostics
    if active_tasks is None:
        active_tasks = set()
    for task in ["ret3m", "rank", "sector_rel", "vol"]:
        if task in active_tasks:
            aux = compute_auxiliary_ic(results, task)
            row.update(aux)

    # Learned uncertainty weights averaged over folds
    for task_name in ALL_TASK_NAMES:
        weight_col = f"uw_weight_{task_name}"
        logvar_col = f"uw_logvar_{task_name}"
        if weight_col in results.columns:
            row[f"uw_{task_name}_mean"] = round(float(results[weight_col].mean()), 4)
            row[f"uw_{task_name}_std"] = round(
                float(results[weight_col].std(ddof=1)), 4
            ) if results[weight_col].nunique() > 1 else 0.0
        if logvar_col in results.columns:
            row[f"logvar_{task_name}_mean"] = round(float(results[logvar_col].mean()), 4)

    return row


# ──────────────────────────────────────────────────────────────────────
# 9. RUN ALL ABLATIONS
# ──────────────────────────────────────────────────────────────────────

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
        tasks_sorted = sorted(tasks)
        print(f"\n{'=' * 70}")
        print(f"Ablation {variant}: tasks = {{{', '.join(tasks_sorted)}}}")
        print(f"{'=' * 70}")

        results = walk_forward_evaluate(
            df=df,
            active_tasks=tasks,
            device=device,
            verbose=verbose,
        )

        summary_row = summarise(results, label=variant, active_tasks=tasks)
        rows.append(summary_row)

        if save_results:
            results_path = Path(output_dir) / f"results_{variant}.parquet"
            results.to_parquet(results_path, index=False)

        print("\nSummary:")
        for k, v in summary_row.items():
            print(f"  {k}: {v}")

    summary_df = pd.DataFrame(rows)

    if save_results:
        summary_path = Path(output_dir) / "ablation_summary_combined.csv"
        summary_df.to_csv(summary_path, index=False)

    return summary_df


# ──────────────────────────────────────────────────────────────────────
# 10. COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────

def print_full_comparison(combined_df: pd.DataFrame, output_dir: str = "output"):
    """
    Print comparison table including RG's results, planned results, and
    combined results. Highlight the best configuration.
    """
    rg_path = Path(output_dir) / "ablation_summary_uncertainty.csv"
    plan_path = Path(output_dir) / "ablation_summary_planned.csv"

    print("\n" + "=" * 100)
    print("FULL COMPARISON: RG's + Planned + Combined Auxiliary Targets")
    print("=" * 100)

    # Header
    print(f"\n{'Config':<14} {'Aux Tasks':<35} {'IC_mean':>8} {'IC_t':>8} {'Sharpe':>8} {'IC_pos%':>8} {'pred_std':>9}")
    print("-" * 100)

    # RG's results
    if rg_path.exists():
        rg_df = pd.read_csv(rg_path)
        print("-- RG's implementation --")
        task_desc_rg = {
            "5a": "ret only",
            "5b": "ret + ret3m",
            "5c": "ret + vol",
            "5d": "ret + ret3m + vol",
        }
        for _, row in rg_df.iterrows():
            desc = task_desc_rg.get(row["label"], "")
            print(
                f"  {row['label']:<12} {desc:<35} {row['IC_mean']:>8.4f} "
                f"{row['IC_t']:>8.3f} {row['LS_Sharpe']:>8.3f} "
                f"{row['IC_pos_frac']:>8.3f} {row['pred_std_ratio']:>9.4f}"
            )
    else:
        print("  [RG's ablation_summary_uncertainty.csv not found]")

    # Planned results
    if plan_path.exists():
        plan_df = pd.read_csv(plan_path)
        print("\n-- Planned implementation --")
        task_desc_plan = {
            "5a_plan": "ret only (baseline)",
            "5b_plan": "ret + rank",
            "5c_plan": "ret + sector_rel",
            "5d_plan": "ret + rank + sector_rel",
        }
        for _, row in plan_df.iterrows():
            desc = task_desc_plan.get(row["label"], "")
            print(
                f"  {row['label']:<12} {desc:<35} {row['IC_mean']:>8.4f} "
                f"{row['IC_t']:>8.3f} {row['LS_Sharpe']:>8.3f} "
                f"{row['IC_pos_frac']:>8.3f} {row['pred_std_ratio']:>9.4f}"
            )
    else:
        print("  [Planned ablation_summary_planned.csv not found]")

    # Combined results
    print("\n-- Combined (this run) --")
    task_desc_combined = {
        "combo_a": "ret + rank",
        "combo_b": "ret + rank + vol",
        "combo_c": "ret + rank + ret3m",
        "combo_d": "ret + rank + vol + ret3m",
        "combo_e": "ret + rank + vol + ret3m + sector_rel",
    }
    for _, row in combined_df.iterrows():
        desc = task_desc_combined.get(row["label"], "")
        print(
            f"  {row['label']:<12} {desc:<35} {row['IC_mean']:>8.4f} "
            f"{row['IC_t']:>8.3f} {row['LS_Sharpe']:>8.3f} "
            f"{row['IC_pos_frac']:>8.3f} {row['pred_std_ratio']:>9.4f}"
        )

    # Find overall winner
    print("\n" + "-" * 100)

    # Collect all results for ranking
    all_rows = []
    if rg_path.exists():
        for _, row in pd.read_csv(rg_path).iterrows():
            all_rows.append(row)
    if plan_path.exists():
        for _, row in pd.read_csv(plan_path).iterrows():
            all_rows.append(row)
    for _, row in combined_df.iterrows():
        all_rows.append(row)

    if all_rows:
        all_df = pd.DataFrame(all_rows)

        # Best by IC
        best_ic_idx = all_df["IC_mean"].idxmax()
        best_ic = all_df.loc[best_ic_idx]
        print(f"\nBest IC_mean   : {best_ic['label']}  (IC={best_ic['IC_mean']:.4f})")

        # Best by Sharpe
        best_sharpe_idx = all_df["LS_Sharpe"].idxmax()
        best_sharpe = all_df.loc[best_sharpe_idx]
        print(f"Best LS_Sharpe : {best_sharpe['label']}  (Sharpe={best_sharpe['LS_Sharpe']:.3f})")

        # Best by IC_t
        best_t_idx = all_df["IC_t"].idxmax()
        best_t = all_df.loc[best_t_idx]
        print(f"Best IC_t      : {best_t['label']}  (t={best_t['IC_t']:.3f})")

    # Learned weight analysis for combined runs
    print("\n" + "=" * 100)
    print("LEARNED UNCERTAINTY WEIGHTS (combined configs)")
    print("=" * 100)
    for _, row in combined_df.iterrows():
        label = row["label"]
        tasks = ABLATION_TASKS[label]
        weights = []
        for task in sorted(tasks):
            w_key = f"uw_{task}_mean"
            if w_key in row and pd.notna(row[w_key]):
                weights.append(f"{task}={row[w_key]:.4f}")
        print(f"  {label:<12} : {', '.join(weights)}")

    print()


# ──────────────────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rung 5 Combined MTL — Best auxiliary target combinations"
    )
    parser.add_argument(
        "--data", type=str, default="data/master_panel.parquet",
        help="Path to master panel parquet",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory to save outputs",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--quiet", action="store_true", help="Suppress fold-level logging")
    parser.add_argument("--no_save", action="store_true", help="Do not save result files")
    args = parser.parse_args()

    print("Loading panel...")
    df = pd.read_parquet(args.data)

    # Minimal sanity check
    needed = [DATE_COL, STOCK_COL, SECTOR_COL, TARGET_COL, RET3M_COL, VOL_COL] + FACTOR_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"Unique tickers: {df[STOCK_COL].nunique():,}")
    print(f"Unique months: {df[DATE_COL].nunique():,}")

    # Compute auxiliary targets
    print("\nComputing auxiliary targets...")
    df = compute_auxiliary_targets(df)

    # Quick diagnostics
    print(f"  rank_target      — mean={df[RANK_COL].mean():.4f}, "
          f"std={df[RANK_COL].std():.4f}, "
          f"range=[{df[RANK_COL].min():.4f}, {df[RANK_COL].max():.4f}]")
    print(f"  sector_rel_target — mean={df[SECTOR_REL_COL].mean():.6f}, "
          f"std={df[SECTOR_REL_COL].std():.4f}, "
          f"range=[{df[SECTOR_REL_COL].min():.4f}, {df[SECTOR_REL_COL].max():.4f}]")
    print(f"  fwd_ret_3m       — NaN frac={df[RET3M_COL].isna().mean():.4f}")
    print(f"  realized_vol     — NaN frac={df[VOL_COL].isna().mean():.4f}")

    # Run ablations
    summary_df = run_all_ablations(
        df=df,
        device=args.device,
        verbose=not args.quiet,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 100)
    print("COMBINED AUXILIARY TARGETS — ABLATION SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    # Full comparison with prior results
    print_full_comparison(summary_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
