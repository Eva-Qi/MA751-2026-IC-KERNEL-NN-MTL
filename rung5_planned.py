"""
Rung 5 (Planned): MTL MLP with PLANNED Auxiliary Targets
=========================================================

Architecture  : Dense(64)+ReLU+Dropout -> Dense(32)+ReLU+Dropout (shared)
               -> task-specific heads (1 linear layer each)

Tasks
-----
  ret        - next-month cross-sectional return          [relative MSE]
  rank       - cross-sectional rank (scaled to [0,1])     [relative MSE]
  sector_rel - sector-relative return (ret - sector mean) [relative MSE]

This file mirrors main.py exactly in architecture and hyperparameters.
The ONLY difference is the auxiliary targets:

  main.py (RG's)          rung5_planned.py (planned)
  ─────────────────       ──────────────────────────
  Aux 1: fwd_ret_3m       Aux 1: cross-sectional rank
  Aux 2: realized_vol     Aux 2: sector-relative return

Motivation
----------
* Cross-sectional rank: Spearman IC is the primary evaluation metric, so
  directly predicting rank gives a gradient signal aligned with evaluation.
  Rank targets are also robust to return outliers.

* Sector-relative return: return minus sector mean separates systematic
  (beta) from idiosyncratic (alpha). Forces shared layers to learn
  stock-specific signals rather than sector rotation.

Uncertainty weighting (Kendall et al. 2018) is identical:
    L = sum_k [ 0.5 * exp(-s_k) * L_k + 0.5 * s_k ]
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from metrics import compute_long_short_sharpe  # Category B consolidation

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# 1. COLUMN NAMES (same as main.py)
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
SECTOR_COL = "sector"
DATE_COL = "date"
STOCK_COL = "ticker"

# Planned auxiliary target columns (computed, not in raw data)
RANK_COL = "rank_target"
SECTOR_REL_COL = "sector_rel_target"

# ──────────────────────────────────────────────────────────────────────
# 2. AUXILIARY TARGET CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────

def compute_planned_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two auxiliary target columns to the panel:

    rank_target      : Within each month, rank stocks by fwd_ret_1m and
                       scale to [0, 1].  Rank 0 = worst, 1 = best.
    sector_rel_target: fwd_ret_1m minus the sector mean for that month.
    """
    df = df.copy()

    # ── Cross-sectional rank scaled to [0, 1] ──
    # pct=True gives percentile rank in [0, 1] directly
    df[RANK_COL] = df.groupby(DATE_COL)[TARGET_COL].rank(pct=True)

    # ── Sector-relative return ──
    sector_month_mean = df.groupby([DATE_COL, SECTOR_COL])[TARGET_COL].transform("mean")
    df[SECTOR_REL_COL] = df[TARGET_COL] - sector_month_mean

    return df


# ──────────────────────────────────────────────────────────────────────
# 3. DATA HELPERS
# ──────────────────────────────────────────────────────────────────────

def make_tensors(
    df: pd.DataFrame,
    x_scaler: StandardScaler | None = None,
):
    """
    Returns:
      X, y_ret, y_rank, y_sector_rel, fitted_x_scaler

    Unlike main.py, neither auxiliary target has NaN issues — both are
    derived directly from fwd_ret_1m which is always present.
    """
    X_raw = np.nan_to_num(df[FACTOR_COLS].values.astype(np.float32), nan=0.0)

    if x_scaler is None:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X_raw)
    else:
        X = x_scaler.transform(X_raw)

    y_ret = df[TARGET_COL].values.astype(np.float32)
    y_rank = df[RANK_COL].values.astype(np.float32)
    y_sector_rel = df[SECTOR_REL_COL].values.astype(np.float32)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_ret, dtype=torch.float32),
        torch.tensor(y_rank, dtype=torch.float32),
        torch.tensor(y_sector_rel, dtype=torch.float32),
        x_scaler,
    )


# ──────────────────────────────────────────────────────────────────────
# 4. MODEL
# ──────────────────────────────────────────────────────────────────────

class MTLNet(nn.Module):
    """
    Same shared trunk as main.py.  Task heads changed to:
      ret        — 1-month forward return
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
            active_tasks = {"ret", "rank", "sector_rel"}
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
        self.head_rank = nn.Linear(hidden2, 1)
        self.head_sector_rel = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(x)
        out = {}

        if "ret" in self.active_tasks:
            out["ret"] = self.head_ret(z).squeeze(-1)
        if "rank" in self.active_tasks:
            out["rank"] = self.head_rank(z).squeeze(-1)
        if "sector_rel" in self.active_tasks:
            out["sector_rel"] = self.head_sector_rel(z).squeeze(-1)

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
    """

    def __init__(
        self,
        active_tasks: set,
        var_ret: float = 1.0,
        var_rank: float = 1.0,
        var_sector_rel: float = 1.0,
    ):
        super().__init__()

        self.active_tasks = set(active_tasks)
        self.var_ret = max(var_ret, 1e-8)
        self.var_rank = max(var_rank, 1e-8)
        self.var_sector_rel = max(var_sector_rel, 1e-8)

        self.log_vars = nn.ParameterDict()
        if "ret" in self.active_tasks:
            self.log_vars["ret"] = nn.Parameter(torch.zeros(1))
        if "rank" in self.active_tasks:
            self.log_vars["rank"] = nn.Parameter(torch.zeros(1))
        if "sector_rel" in self.active_tasks:
            self.log_vars["sector_rel"] = nn.Parameter(torch.zeros(1))

    @staticmethod
    def from_train_data(
        y_ret: torch.Tensor,
        y_rank: torch.Tensor,
        y_sector_rel: torch.Tensor,
        active_tasks: set,
    ) -> "UncertaintyMTLLoss":
        return UncertaintyMTLLoss(
            active_tasks=active_tasks,
            var_ret=float(y_ret.var()),
            var_rank=float(y_rank.var()),
            var_sector_rel=float(y_sector_rel.var()),
        )

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        y_ret: torch.Tensor,
        y_rank: torch.Tensor,
        y_sector_rel: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.zeros(1, device=y_ret.device).squeeze()

        if "ret" in preds:
            base_ret = ((preds["ret"] - y_ret) ** 2).mean() / self.var_ret
            s = self.log_vars["ret"].squeeze()
            total = total + 0.5 * torch.exp(-s) * base_ret + 0.5 * s

        if "rank" in preds:
            base_rank = ((preds["rank"] - y_rank) ** 2).mean() / self.var_rank
            s = self.log_vars["rank"].squeeze()
            total = total + 0.5 * torch.exp(-s) * base_rank + 0.5 * s

        if "sector_rel" in preds:
            base_sr = ((preds["sector_rel"] - y_sector_rel) ** 2).mean() / self.var_sector_rel
            s = self.log_vars["sector_rel"].squeeze()
            total = total + 0.5 * torch.exp(-s) * base_sr + 0.5 * s

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
    y_ret_tr: torch.Tensor,
    y_rank_tr: torch.Tensor,
    y_sector_rel_tr: torch.Tensor,
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
    n_val = max(1, int(len(X_tr) * val_frac))

    X_val, X_tr = X_tr[-n_val:], X_tr[:-n_val]
    y_ret_val, y_ret_tr = y_ret_tr[-n_val:], y_ret_tr[:-n_val]
    y_rank_val, y_rank_tr = y_rank_tr[-n_val:], y_rank_tr[:-n_val]
    y_sr_val, y_sr_tr = y_sector_rel_tr[-n_val:], y_sector_rel_tr[:-n_val]

    criterion = UncertaintyMTLLoss.from_train_data(
        y_ret_tr, y_rank_tr, y_sr_tr, active_tasks
    ).to(device)

    train_dl = DataLoader(
        TensorDataset(X_tr, y_ret_tr, y_rank_tr, y_sr_tr),
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

        for Xb, yr, yrk, ysr in train_dl:
            Xb, yr, yrk, ysr = (t.to(device) for t in (Xb, yr, yrk, ysr))

            optimizer.zero_grad()
            loss = criterion(model(Xb), yr, yrk, ysr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        criterion.eval()

        with torch.no_grad():
            val_loss = criterion(
                model(X_val.to(device)),
                y_ret_val.to(device),
                y_rank_val.to(device),
                y_sr_val.to(device),
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
    "5a_plan": {"ret"},                            # single-task baseline
    "5b_plan": {"ret", "rank"},                    # + cross-sectional rank
    "5c_plan": {"ret", "sector_rel"},              # + sector-relative return
    "5d_plan": {"ret", "rank", "sector_rel"},      # full planned MTL
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
            X_tr, y_ret_tr, y_rank_tr, y_sr_tr, x_scaler,
        ) = make_tensors(df_tr)

        (
            X_te, y_ret_te, y_rank_te, y_sr_te, _,
        ) = make_tensors(df_te, x_scaler=x_scaler)

        model, learned_weights, learned_log_vars = train_one_fold(
            X_tr, y_ret_tr, y_rank_tr, y_sr_tr,
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
            "rank_true": y_rank_te.numpy(),
            "sector_rel_true": y_sr_te.numpy(),
        })

        if "rank" in pred_dict:
            out["rank_pred"] = pred_dict["rank"].numpy()

        if "sector_rel" in pred_dict:
            out["sector_rel_pred"] = pred_dict["sector_rel"].numpy()

        # store learned uncertainty weights on every row for easy aggregation
        for task_name in ["ret", "rank", "sector_rel"]:
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


# ──────────────────────────────────────────────────────────────────────
# 8. METRICS / SUMMARY HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_monthly_ic(results: pd.DataFrame) -> pd.Series:
    monthly = results.groupby(DATE_COL).apply(
        lambda g: spearmanr(g["y_true"], g["y_pred"]).statistic if len(g) > 5 else np.nan
    )
    return monthly.dropna()


# compute_long_short_sharpe removed — imported from metrics.py (Category B consolidation)


def compute_rank_auxiliary_ic(results: pd.DataFrame) -> dict:
    """Spearman IC of the rank auxiliary head — diagnostic."""
    if "rank_pred" not in results.columns:
        return {"rank_ic_mean": np.nan, "rank_ic_std": np.nan}

    sub = results.dropna(subset=["rank_true", "rank_pred"])
    if len(sub) == 0:
        return {"rank_ic_mean": np.nan, "rank_ic_std": np.nan}

    monthly = sub.groupby(DATE_COL).apply(
        lambda g: spearmanr(g["rank_true"], g["rank_pred"]).statistic if len(g) > 5 else np.nan
    ).dropna()

    return {
        "rank_ic_mean": round(float(monthly.mean()), 4) if len(monthly) else np.nan,
        "rank_ic_std": round(float(monthly.std()), 4) if len(monthly) else np.nan,
    }


def compute_sector_rel_auxiliary_ic(results: pd.DataFrame) -> dict:
    """Spearman IC of the sector-relative auxiliary head — diagnostic."""
    if "sector_rel_pred" not in results.columns:
        return {"sector_rel_ic_mean": np.nan, "sector_rel_ic_std": np.nan}

    sub = results.dropna(subset=["sector_rel_true", "sector_rel_pred"])
    if len(sub) == 0:
        return {"sector_rel_ic_mean": np.nan, "sector_rel_ic_std": np.nan}

    monthly = sub.groupby(DATE_COL).apply(
        lambda g: spearmanr(g["sector_rel_true"], g["sector_rel_pred"]).statistic if len(g) > 5 else np.nan
    ).dropna()

    return {
        "sector_rel_ic_mean": round(float(monthly.mean()), 4) if len(monthly) else np.nan,
        "sector_rel_ic_std": round(float(monthly.std()), 4) if len(monthly) else np.nan,
    }


def summarise(results: pd.DataFrame, label: str = "") -> dict:
    ic = compute_monthly_ic(results)

    if len(ic) >= 2 and ic.std() > 1e-8:
        t_val = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)) + 1e-8)
    else:
        t_val = np.nan

    aux_rank = compute_rank_auxiliary_ic(results)
    aux_sr = compute_sector_rel_auxiliary_ic(results)
    pred_std_ratio = float(results["y_pred"].std() / (results["y_true"].std() + 1e-8))

    row = {
        "label": label,
        "IC_mean": round(float(ic.mean()), 4) if len(ic) else np.nan,
        "IC_std": round(float(ic.std(ddof=1)), 4) if len(ic) > 1 else np.nan,
        "IC_t": round(float(t_val), 3) if pd.notna(t_val) else np.nan,
        "IC_pos_frac": round(float((ic > 0).mean()), 3) if len(ic) else np.nan,
        "LS_Sharpe": round(compute_long_short_sharpe(results), 3),
        "rank_ic_mean": aux_rank["rank_ic_mean"],
        "rank_ic_std": aux_rank["rank_ic_std"],
        "sector_rel_ic_mean": aux_sr["sector_rel_ic_mean"],
        "sector_rel_ic_std": aux_sr["sector_rel_ic_std"],
        "pred_std_ratio": round(pred_std_ratio, 4),
        "n_months": int(len(ic)),
    }

    # learned uncertainty weights averaged over folds
    for task_name in ["ret", "rank", "sector_rel"]:
        weight_col = f"uw_weight_{task_name}"
        logvar_col = f"uw_logvar_{task_name}"
        if weight_col in results.columns:
            row[f"uw_{task_name}_mean"] = round(float(results[weight_col].mean()), 4)
            row[f"uw_{task_name}_std"] = round(
                float(results[weight_col].std(ddof=1)), 4
            ) if results[weight_col].nunique() > 1 else 0.0
        else:
            row[f"uw_{task_name}_mean"] = np.nan
            row[f"uw_{task_name}_std"] = np.nan

        if logvar_col in results.columns:
            row[f"logvar_{task_name}_mean"] = round(float(results[logvar_col].mean()), 4)
        else:
            row[f"logvar_{task_name}_mean"] = np.nan

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
            results_path = Path(output_dir) / f"results_{variant}.parquet"
            results.to_parquet(results_path, index=False)

        print("\nSummary:")
        for k, v in summary_row.items():
            print(f"  {k}: {v}")

    summary_df = pd.DataFrame(rows)

    if save_results:
        summary_path = Path(output_dir) / "ablation_summary_planned.csv"
        summary_df.to_csv(summary_path, index=False)

    return summary_df


# ──────────────────────────────────────────────────────────────────────
# 10. SIDE-BY-SIDE COMPARISON
# ──────────────────────────────────────────────────────────────────────

def print_comparison(planned_df: pd.DataFrame, output_dir: str = "output"):
    """
    Print side-by-side comparison table: planned vs RG's results.
    """
    rg_path = Path(output_dir) / "ablation_summary_uncertainty.csv"
    if not rg_path.exists():
        print("\n[WARN] RG's ablation_summary_uncertainty.csv not found — skipping comparison.")
        return

    rg_df = pd.read_csv(rg_path)

    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE COMPARISON: Planned vs RG's Auxiliary Targets")
    print("=" * 100)

    # Header
    print(f"\n{'Config':<12} {'IC_mean':>8} {'IC_t':>8} {'Sharpe':>8} {'IC_pos%':>8} {'pred_std':>9}   Notes")
    print("-" * 80)

    # RG's results
    print("── RG's implementation (fwd_ret_3m + realized_vol) ──")
    for _, row in rg_df.iterrows():
        task_desc = {
            "5a": "ret only",
            "5b": "ret + ret3m",
            "5c": "ret + vol",
            "5d": "ret + ret3m + vol",
        }.get(row["label"], "")
        print(
            f"  {row['label']:<10} {row['IC_mean']:>8.4f} {row['IC_t']:>8.3f} "
            f"{row['LS_Sharpe']:>8.3f} {row['IC_pos_frac']:>8.3f} {row['pred_std_ratio']:>9.4f}   {task_desc}"
        )

    print()
    print("── Planned implementation (rank + sector-relative) ──")
    for _, row in planned_df.iterrows():
        task_desc = {
            "5a_plan": "ret only (baseline)",
            "5b_plan": "ret + rank",
            "5c_plan": "ret + sector_rel",
            "5d_plan": "ret + rank + sector_rel",
        }.get(row["label"], "")
        print(
            f"  {row['label']:<10} {row['IC_mean']:>8.4f} {row['IC_t']:>8.3f} "
            f"{row['LS_Sharpe']:>8.3f} {row['IC_pos_frac']:>8.3f} {row['pred_std_ratio']:>9.4f}   {task_desc}"
        )

    # Quick delta summary
    print()
    print("── Delta (planned - RG) for matched configs ──")
    match_pairs = [
        ("5a", "5a_plan", "single-task baseline"),
        ("5b", "5b_plan", "aux 1: ret3m vs rank"),
        ("5c", "5c_plan", "aux 2: vol vs sector_rel"),
        ("5d", "5d_plan", "full MTL"),
    ]

    rg_map = {r["label"]: r for _, r in rg_df.iterrows()}
    plan_map = {r["label"]: r for _, r in planned_df.iterrows()}

    print(f"  {'Match':<30} {'dIC_mean':>9} {'dIC_t':>8} {'dSharpe':>9}")
    print("  " + "-" * 60)
    for rg_lbl, plan_lbl, desc in match_pairs:
        if rg_lbl in rg_map and plan_lbl in plan_map:
            rg_r = rg_map[rg_lbl]
            pl_r = plan_map[plan_lbl]
            d_ic = pl_r["IC_mean"] - rg_r["IC_mean"]
            d_t = pl_r["IC_t"] - rg_r["IC_t"]
            d_sh = pl_r["LS_Sharpe"] - rg_r["LS_Sharpe"]
            print(f"  {desc:<30} {d_ic:>+9.4f} {d_t:>+8.3f} {d_sh:>+9.3f}")

    print()


# ──────────────────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rung 5 MTL MLP with PLANNED auxiliary targets (rank + sector-relative)"
    )
    parser.add_argument(
        "--data", type=str, default="data/master_panel_v2.parquet",
        help="Path to master panel parquet. Default is V2 (CRSP-based panel). V1 (master_panel.parquet) has known dei-namespace gaps for V/BRK-B/STZ — use only for legacy comparison.",
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
    needed = [DATE_COL, STOCK_COL, SECTOR_COL, TARGET_COL] + FACTOR_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"Unique tickers: {df[STOCK_COL].nunique():,}")
    print(f"Unique months: {df[DATE_COL].nunique():,}")

    # ── Compute planned auxiliary targets ──
    print("\nComputing planned auxiliary targets...")
    df = compute_planned_targets(df)

    # Quick diagnostics
    print(f"  rank_target   — mean={df[RANK_COL].mean():.4f}, "
          f"std={df[RANK_COL].std():.4f}, "
          f"range=[{df[RANK_COL].min():.4f}, {df[RANK_COL].max():.4f}]")
    print(f"  sector_rel    — mean={df[SECTOR_REL_COL].mean():.6f}, "
          f"std={df[SECTOR_REL_COL].std():.4f}, "
          f"range=[{df[SECTOR_REL_COL].min():.4f}, {df[SECTOR_REL_COL].max():.4f}]")

    # Correlation between planned targets and fwd_ret_1m
    from scipy.stats import spearmanr as _sp
    rho_rank, _ = _sp(df[TARGET_COL], df[RANK_COL])
    rho_sr, _ = _sp(df[TARGET_COL], df[SECTOR_REL_COL])
    print(f"  corr(ret, rank)       = {rho_rank:.4f}")
    print(f"  corr(ret, sector_rel) = {rho_sr:.4f}")

    # ── Run ablations ──
    summary_df = run_all_ablations(
        df=df,
        device=args.device,
        verbose=not args.quiet,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 100)
    print("PLANNED AUXILIARY TARGETS — ABLATION SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    # ── Side-by-side comparison ──
    print_comparison(summary_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
