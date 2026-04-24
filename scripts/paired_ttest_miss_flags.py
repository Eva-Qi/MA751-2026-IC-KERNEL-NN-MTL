"""
Paired t-test: do missingness indicators change Rung 1-2 walk-forward IC?

Runs each of OLS/IC-Ensemble/LASSO/Ridge twice on the same folds:
  (A) without flags: ALL_FEATURE_COLS_V2 (14 features)
  (B) with flags   : ALL_FEATURE_COLS_V2_WITH_MISS (19 features)

For each model, computes monthly IC series for both and runs a paired t-test
(scipy.stats.ttest_rel) on the per-month delta. 58 test months.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURE_COLS_V2, ALL_FEATURE_COLS_V2_WITH_MISS,
    TARGET_COL, DATE_COL,
)
from run_rung12_v2 import (
    run_walk_forward, ols_model, ic_ensemble_model, lasso_model, ridge_model,
)

DATA = ROOT / "data" / "master_panel_v2.parquet"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# In run_rung12_v2.py we already flipped ALL_FEATURE_COLS_V2 to WITH_MISS.
# Use the raw base list from config explicitly to avoid confusion.
BASE_FEATURES = [c for c in ALL_FEATURE_COLS_V2_WITH_MISS
                 if c not in ("has_sue", "has_short_interest", "has_inst_ownership",
                              "has_analyst_consensus", "has_positive_earnings")]
WITH_MISS_FEATURES = list(ALL_FEATURE_COLS_V2_WITH_MISS)

print(f"Base features  : {len(BASE_FEATURES)}")
print(f"WithMiss feats : {len(WITH_MISS_FEATURES)}")

models = [
    ("1a_OLS",         ols_model),
    ("1b_IC_Ensemble", ic_ensemble_model),
    ("2a_LASSO",       lasso_model),
    ("2b_Ridge",       ridge_model),
]


def run_config(df, feature_set, tag):
    """Run all 4 models with a specific feature set, return dict of monthly IC series."""
    print(f"\n--- Running with tag={tag} ({len(feature_set)} features) ---")
    out = {}
    for label, fn in models:
        monthly = run_walk_forward(df, fn, f"{label}_{tag}", features=feature_set)
        ic_series = (
            monthly.set_index(DATE_COL)["IC"]
            .dropna()
            .sort_index()
        )
        out[label] = ic_series
    return out


print("Loading panel...")
df = pd.read_parquet(DATA)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Run both configs
ics_nomiss   = run_config(df, BASE_FEATURES, "nomiss")
ics_withmiss = run_config(df, WITH_MISS_FEATURES, "withmiss")

# Paired t-test per model
print("\n" + "=" * 80)
print("PAIRED T-TEST: monthly IC (WithMiss - NoMiss)")
print("=" * 80)

rows = []
for label, _ in models:
    a = ics_nomiss[label]
    b = ics_withmiss[label]
    # Align on common index (should already be aligned; defensive)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    delta = b - a
    tstat, pval = ttest_rel(b, a, nan_policy="omit")
    rows.append({
        "Model": label,
        "n_months": int(len(delta)),
        "IC_nomiss": a.mean(),
        "IC_withmiss": b.mean(),
        "delta_mean": delta.mean(),
        "delta_std": delta.std(),
        "t_stat": tstat,
        "p_value": pval,
        "sig_at_0.05": bool(pval < 0.05),
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) else str(x)))

result.to_csv(OUTPUT / "paired_ttest_miss_flags.csv", index=False)
print(f"\nSaved -> {OUTPUT / 'paired_ttest_miss_flags.csv'}")

# Also write monthly delta series for diagnostic
delta_df = pd.DataFrame({
    label: (ics_withmiss[label] - ics_nomiss[label])
    for label, _ in models
})
delta_df.to_csv(OUTPUT / "monthly_ic_delta_miss_flags.csv")
print(f"Saved -> {OUTPUT / 'monthly_ic_delta_miss_flags.csv'}")
