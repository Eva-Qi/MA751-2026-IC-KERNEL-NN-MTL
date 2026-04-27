"""
Shared MLP architecture and single-task training function for audit rungs.

Consolidated from 3 identical/equivalent definitions:
  - rung4_hp_search.py  (canonical: parameterized dropout)
  - rung5_audit.py      (hardcoded Dropout(0.10))
  - rung45_seed_bagging.py (hardcoded Dropout(0.10))

BEHAVIOR NOTE: When dropout=0.10 (the default), all 3 original versions produce
IDENTICAL results because nn.Dropout(0.10) == nn.Dropout(dropout=0.10).

Category C + D (Group 2) consolidation — MA751 code-council audit 2026-04-24.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """n_in → hidden → hidden//2 → 1, ReLU + configurable dropout.

    Architecture matches main.py rung4: two hidden layers with ReLU activation
    and dropout after each. When dropout=0.10, output is bit-identical to the
    hardcoded rung5_audit.py / rung45_seed_bagging.py versions.
    """
    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(
    X_tr, y_tr, X_val, y_val, X_te,
    hidden: int,
    seed: int,
    lr: float = 1e-3,
    dropout: float = 0.10,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 256,
):
    """Train one MLP fold; return test predictions as numpy array.

    Early-stopping on validation Huber loss (smooth_l1_loss).
    Scaler must be applied BEFORE calling this function.

    Signature is the superset of rung4_hp_search.py (which has dropout param)
    and rung5_audit.py / rung45_seed_bagging.py (which hardcode dropout=0.10).
    Callers that do NOT pass dropout get the same behavior as the original
    hardcoded versions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(X_tr.shape[1], hidden=hidden, dropout=dropout)
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

    for _ in range(epochs):
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
