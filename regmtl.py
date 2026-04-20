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

# Precomputed regime columns expected in the enriched panel
REGIME_COLS = [
    "regime_p0",
    "regime_p1",
    "regime_p2",
]

FACTOR_COLS = FACTOR_ZSCORE_COLS + MACRO_COLS

TARGET_COL = "fwd_ret_1m"
RET3M_COL = "fwd_ret_3m"
VOL_COL = "realized_vol"
SECTOR_COL = "sector"
DATE_COL = "date"
STOCK_COL = "ticker"

DEFAULT_N_EXPERTS = 3

ABLATION_TASKS = {
    "5a": {"ret"},
    "5b": {"ret", "ret3m"},
    "5c": {"ret", "vol"},
    "5d": {"ret", "ret3m", "vol"},
}


def infer_available_regime_cols(df: pd.DataFrame, requested_cols: list[str]) -> list[str]:
    cols = [c for c in requested_cols if c in df.columns]
    if len(cols) == 0:
        raise ValueError(
            f"No regime columns found. Requested: {requested_cols}. "
            f"Available regime-like columns: {[c for c in df.columns if 'regime' in c.lower()]}"
        )
    return cols


def prepare_panel_for_training(
    df: pd.DataFrame,
    factor_cols: list[str],
    regime_cols: list[str],
) -> pd.DataFrame:
    """
    Minimal training prep.

    - enforce datetime
    - sort rows
    - drop rows missing core targets
    - fill early missing lagged regime values
    - sanitize numeric features
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, STOCK_COL]).reset_index(drop=True)

    # Need primary return and vol targets to train/evaluate.
    df = df.dropna(subset=[TARGET_COL, VOL_COL])

    # Fill regime fields so earliest lagged months do not get dropped.
    for c in regime_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            df[c] = df[c].fillna("MISSING")

    # Sanitize feature columns
    for c in factor_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def expand_regime_features(df: pd.DataFrame, regime_cols: list[str]) -> pd.DataFrame:
    """
    Converts regime inputs into numeric gate features.
    Numeric columns pass through, categorical columns are one-hot encoded.
    """
    blocks = []

    for c in regime_cols:
        s = df[c]

        if pd.api.types.is_numeric_dtype(s):
            arr = s.astype(float).fillna(0.0).to_frame()
            arr.columns = [c]
            blocks.append(arr)
        else:
            dummies = pd.get_dummies(s.fillna("MISSING"), prefix=c, dtype=float)
            blocks.append(dummies)

    out = pd.concat(blocks, axis=1)
    return out


def make_tensors(
    df: pd.DataFrame,
    feature_cols: list[str],
    regime_cols: list[str],
    x_scaler: StandardScaler | None = None,
    r_scaler: StandardScaler | None = None,
    vol_mean: float | None = None,
    vol_std: float | None = None,
):
    """
    Returns:
      X, R, y_ret, y_ret3m, y_vol_std,
      fitted_x_scaler, fitted_r_scaler, fitted_vol_mean, fitted_vol_std,
      regime_feature_names
    """
    X_raw = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0.0)

    if x_scaler is None:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X_raw)
    else:
        X = x_scaler.transform(X_raw)

    regime_df = expand_regime_features(df, regime_cols)
    R_raw = np.nan_to_num(regime_df.values.astype(np.float32), nan=0.0)

    if r_scaler is None:
        r_scaler = StandardScaler()
        R = r_scaler.fit_transform(R_raw)
    else:
        R = r_scaler.transform(R_raw)

    y_ret = df[TARGET_COL].values.astype(np.float32)
    y_ret3m_raw = df[RET3M_COL].values.astype(np.float64)  # preserve NaN
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
        r_scaler,
        vol_mean,
        vol_std,
        regime_df.columns.tolist(),
    )


# 3. MODEL: REGIME-GATED MIXTURE OF EXPERTS
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

    def _mix_experts(
        self,
        z: torch.Tensor,
        gate_weights: torch.Tensor,
        experts: nn.ModuleList,
    ) -> torch.Tensor:
        expert_outs = [expert(z).squeeze(-1) for expert in experts]  # list of [B]
        expert_stack = torch.stack(expert_outs, dim=1)  # [B, K]
        return (gate_weights * expert_stack).sum(dim=1)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.shared(x)
        gate_logits = self.gate(r)
        gate_weights = torch.softmax(gate_logits, dim=1)

        out = {"gate_weights": gate_weights}

        if "ret" in self.active_tasks:
            out["ret"] = self._mix_experts(z, gate_weights, self.ret_experts)

        if "ret3m" in self.active_tasks:
            out["ret3m"] = self._mix_experts(z, gate_weights, self.ret3m_experts)

        if "vol" in self.active_tasks:
            out["vol"] = self._mix_experts(z, gate_weights, self.vol_experts)

        return out


# 4. UNCERTAINTY-WEIGHTED MULTI-TASK LOSS

class UncertaintyMTLLoss(nn.Module):
    """
    For each active task k:
        0.5 * exp(-s_k) * L_k + 0.5 * s_k
    where s_k = log(sigma_k^2) is trainable.

    Base losses are relative MSEs normalized by train-fold variance.
    fwd_ret_3m rows with NaN are excluded from ret3m loss.
    """

    def __init__(
        self,
        active_tasks: set,
        var_ret: float = 1.0,
        var_ret3m: float = 1.0,
        var_vol: float = 1.0,
    ):
        super().__init__()

        self.active_tasks = set(active_tasks)
        self.var_ret = max(var_ret, 1e-8)
        self.var_ret3m = max(var_ret3m, 1e-8)
        self.var_vol = max(var_vol, 1e-8)

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

        return UncertaintyMTLLoss(
            active_tasks=active_tasks,
            var_ret=float(y_ret.var()),
            var_ret3m=var_ret3m,
            var_vol=float(y_vol_std.var()),
        )

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        y_ret: torch.Tensor,
        y_ret3m: torch.Tensor,
        y_vol_std: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.zeros(1, device=y_ret.device).squeeze()

        if "ret" in preds:
            base_ret = ((preds["ret"] - y_ret) ** 2).mean() / self.var_ret
            s = self.log_vars["ret"].squeeze()
            total = total + 0.5 * torch.exp(-s) * base_ret + 0.5 * s

        if "ret3m" in preds:
            valid = ~torch.isnan(y_ret3m)
            if valid.sum() > 0:
                base_ret3m = ((preds["ret3m"][valid] - y_ret3m[valid]) ** 2).mean() / self.var_ret3m
                s = self.log_vars["ret3m"].squeeze()
                total = total + 0.5 * torch.exp(-s) * base_ret3m + 0.5 * s

        if "vol" in preds:
            base_vol = ((preds["vol"] - y_vol_std) ** 2).mean() / self.var_vol
            s = self.log_vars["vol"].squeeze()
            total = total + 0.5 * torch.exp(-s) * base_vol + 0.5 * s

        return total

    def learned_task_weights(self) -> dict[str, float]:
        return {k: float(torch.exp(-v.detach().cpu()).item()) for k, v in self.log_vars.items()}

    def learned_log_vars(self) -> dict[str, float]:
        return {k: float(v.detach().cpu().item()) for k, v in self.log_vars.items()}


# 5. TRAIN ONE FOLD

def train_one_fold(
    X_tr: torch.Tensor,
    R_tr: torch.Tensor,
    y_ret_tr: torch.Tensor,
    y_ret3m_tr: torch.Tensor,
    y_vol_tr: torch.Tensor,
    active_tasks: set,
    n_experts: int = DEFAULT_N_EXPERTS,
    epochs: int = 150,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.10,
    patience: int = 20,
    val_frac: float = 0.10,
    device: str = "cpu",
) -> tuple[RegimeGatedMTLMoE, dict[str, float], dict[str, float]]:
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
        batch_size=batch_size,
        shuffle=True,
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
                val_preds,
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

# 6. WALK-FORWARD EVALUATION
def walk_forward_evaluate(
    df: pd.DataFrame,
    active_tasks: set,
    feature_cols: list[str],
    regime_cols: list[str],
    n_experts: int = DEFAULT_N_EXPERTS,
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
            X_tr, R_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            x_scaler, r_scaler, vol_mean, vol_std, regime_feature_names,
        ) = make_tensors(
            df_tr,
            feature_cols=feature_cols,
            regime_cols=regime_cols,
        )

        (
            X_te, R_te, y_ret_te, y_3m_te, y_vol_te_std,
            _, _, _, _, _,
        ) = make_tensors(
            df_te,
            feature_cols=feature_cols,
            regime_cols=regime_cols,
            x_scaler=x_scaler,
            r_scaler=r_scaler,
            vol_mean=vol_mean,
            vol_std=vol_std,
        )

        model, learned_weights, learned_log_vars = train_one_fold(
            X_tr, R_tr, y_ret_tr, y_3m_tr, y_vol_tr,
            active_tasks=active_tasks,
            n_experts=n_experts,
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
            pred_dict = model(X_te, R_te)

        y_pred_ret = pred_dict["ret"].numpy()

        out = pd.DataFrame({
            DATE_COL: test_month,
            STOCK_COL: df_te[STOCK_COL].values,
            SECTOR_COL: df_te[SECTOR_COL].values,
            "y_true": y_ret_te.numpy(),
            "y_pred": y_pred_ret,
            "fold": i,
            "realized_vol_true": df_te[VOL_COL].values.astype(float),
            "fwd_ret_3m_true": df_te[RET3M_COL].values.astype(float),
        })

        for c in regime_cols:
            out[c] = df_te[c].values

        if "ret3m" in pred_dict:
            out["ret3m_pred"] = pred_dict["ret3m"].numpy()

        if "vol" in pred_dict:
            vol_pred_std = pred_dict["vol"].numpy()
            out["vol_pred"] = vol_pred_std * vol_std + vol_mean

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
            weight_msg = ", ".join([f"{k}={v:.3f}" for k, v in learned_weights.items()])
            gate_msg = ", ".join([f"g{k}={out[f'gate_w_{k}'].mean():.3f}" for k in range(gate_w.shape[1])])
            print(
                f"fold {i:3d} | {str(test_month)[:7]} | "
                f"n_train={len(df_tr):6,d} | IC={ic:+.4f} | "
                f"UW: {weight_msg} | Gate avg: {gate_msg}"
            )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# 7. METRICS / SUMMARIES

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


def compute_ret3m_auxiliary_ic(results: pd.DataFrame) -> dict:
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
    if "vol_pred" not in results.columns:
        return np.nan

    sub = results.dropna(subset=["realized_vol_true", "vol_pred"])
    if len(sub) == 0:
        return np.nan

    return round(float(sub["realized_vol_true"].corr(sub["vol_pred"])), 4)


def compute_gate_summary(results: pd.DataFrame) -> dict:
    gate_cols = [c for c in results.columns if c.startswith("gate_w_")]
    out = {}
    for c in gate_cols:
        out[f"{c}_mean"] = round(float(results[c].mean()), 4)
        out[f"{c}_std"] = round(float(results[c].std(ddof=1)), 4) if results[c].nunique() > 1 else 0.0
    return out


def summarise(results: pd.DataFrame, label: str = "") -> dict:
    ic = compute_monthly_ic(results)

    if len(ic) >= 2 and ic.std() > 1e-8:
        t_val = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)) + 1e-8)
    else:
        t_val = np.nan

    aux3m = compute_ret3m_auxiliary_ic(results)
    vol_corr = compute_vol_auxiliary_corr(results)
    pred_std_ratio = float(results["y_pred"].std() / (results["y_true"].std() + 1e-8))
    gate_summary = compute_gate_summary(results)

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

    row.update(gate_summary)
    return row


# =========================================================
# 8. RUN ALL ABLATIONS
# =========================================================

def run_all_ablations(
    df: pd.DataFrame,
    feature_cols: list[str],
    regime_cols: list[str],
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
        print(f"Ablation {variant}: tasks = {tasks} | experts = {n_experts}")
        print(f"{'=' * 70}")

        results = walk_forward_evaluate(
            df=df,
            active_tasks=tasks,
            feature_cols=feature_cols,
            regime_cols=regime_cols,
            n_experts=n_experts,
            device=device,
            verbose=verbose,
        )

        summary_row = summarise(results, label=variant)
        rows.append(summary_row)

        if save_results:
            results_path = Path(output_dir) / f"results_{variant}_regime_moe.parquet"
            results.to_parquet(results_path, index=False)

        print("\nSummary:")
        for k, v in summary_row.items():
            print(f"  {k}: {v}")

    summary_df = pd.DataFrame(rows)

    if save_results:
        summary_path = Path(output_dir) / "ablation_summary_regime_moe.csv"
        summary_df.to_csv(summary_path, index=False)

    return summary_df


# =========================================================
# 9. MAIN
# =========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rung 5 Regime-Gated Mixture-of-Experts MTL MLP"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/master_panel_with_regimes.parquet",
        help="Path to master panel parquet WITH precomputed regime columns"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress fold-level logging"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save result files"
    )
    parser.add_argument(
        "--n_experts",
        type=int,
        default=DEFAULT_N_EXPERTS,
        help="Number of experts in the mixture"
    )
    parser.add_argument(
        "--regime_cols",
        type=str,
        default=",".join(REGIME_COLS),
        help="Comma-separated regime columns to use"
    )
    args = parser.parse_args()

    print("Loading panel...")
    df = pd.read_parquet(args.data)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    regime_cols = [c.strip() for c in args.regime_cols.split(",") if c.strip()]

    # Strict: only use precomputed regime columns.
    missing_regimes = [c for c in regime_cols if c not in df.columns]
    if missing_regimes:
        raise ValueError(
            f"Missing precomputed regime columns: {missing_regimes}\n"
            f"Loaded file: {args.data}\n"
            "Expected a regime-enriched panel such as "
            "'data/master_panel_with_regimes.parquet'.\n"
            "Run the regime-generation script first, then rerun this training script."
        )

    regime_cols = infer_available_regime_cols(df, regime_cols)

    needed = [DATE_COL, STOCK_COL, SECTOR_COL, TARGET_COL, RET3M_COL, VOL_COL] + FACTOR_COLS + regime_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = prepare_panel_for_training(
        df=df,
        factor_cols=FACTOR_COLS,
        regime_cols=regime_cols,
    )

    print(f"Rows: {len(df):,}")
    print(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"Unique tickers: {df[STOCK_COL].nunique():,}")
    print(f"Unique months: {df[DATE_COL].nunique():,}")
    print(f"Using regime columns: {regime_cols}")
    print(f"Using experts: {args.n_experts}")

    summary_df = run_all_ablations(
        df=df,
        feature_cols=FACTOR_COLS,
        regime_cols=regime_cols,
        n_experts=args.n_experts,
        device=args.device,
        verbose=not args.quiet,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 90)
    print("REGIME-GATED MIXTURE-OF-EXPERTS ABLATION SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()