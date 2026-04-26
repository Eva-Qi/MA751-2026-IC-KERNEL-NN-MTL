"""
Generate all paper figures from the audit-cycle output CSVs.

Run:
    python make_figures.py

Outputs PDFs in paper/figures/ for direct inclusion in main.tex.

Figures produced:
    Fig 3: GAM n_splines sweep (combined coarse + fine-tune)
    Fig 4: XGBoost sensitivity grid + GKX-tuned callout
    Fig 5: MLP seed sensitivity bar chart
    Fig 6: CPCV Sharpe distribution boxplot
    Fig 7: Pareto plot with CPCV uncertainty error bars
    Fig 8: Cumulative LS portfolio return curves (top 5 models)

Figures 1 (data flow) and 2 (ladder schematic) are not data-driven —
draw separately in TikZ or draw.io.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Common style
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Reference benchmarks (from audit)
RIDGE_FLOOR_SHARPE = 0.925
RIDGE_FLOOR_IC = 0.0164
ZERO_PRED_SHARPE = 0.262
ZERO_PRED_IC = 0.0002


def _save(fig, basename):
    """Save figure as both vector PDF (for LaTeX) and 300-DPI PNG (for preview)."""
    pdf_path = OUT / f"{basename}.pdf"
    png_path = OUT / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"[ok] {basename}.pdf  +  {basename}.png")


# ---------------------------------------------------------------------
def fig3_gam_n_splines():
    """GAM n_splines sweep — dual-axis IC + Sharpe."""
    coarse = pd.read_csv(ROOT / "output" / "rung3a_n_splines_sweep.csv")
    fine = pd.read_csv(ROOT / "output" / "rung3a_n_splines_finetune.csv")
    df = pd.concat([coarse, fine], ignore_index=True)
    df = df.dropna(subset=["IC_mean"]).sort_values("n_splines")
    # de-duplicate (n=5 may exist in both, prefer coarse since that's the original)
    df = df.drop_duplicates("n_splines", keep="first").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(df["n_splines"], df["IC_mean"], "o-", color="#1f77b4",
             markersize=7, linewidth=2, label="IC mean")
    ax1.set_xlabel("$n_{splines}$ (cubic spline basis count per feature)")
    ax1.set_ylabel("IC mean (Spearman)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.axhline(0, ls=":", color="gray", alpha=0.4, linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(df["n_splines"], df["LS_Sharpe"], "s-", color="#ff7f0e",
             markersize=7, linewidth=2, label="LS Sharpe")
    ax2.set_ylabel("LS Sharpe (annualized)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.axhline(RIDGE_FLOOR_SHARPE, ls="--", color="#ff7f0e", alpha=0.4,
                linewidth=1.2, label="Ridge floor (0.925)")
    ax2.spines["right"].set_visible(True)

    # Annotate peaks
    peak_ic = df.loc[df["IC_mean"].idxmax()]
    peak_sharpe = df.loc[df["LS_Sharpe"].idxmax()]
    ax1.annotate(f"IC peak\nn={int(peak_ic['n_splines'])}",
                 xy=(peak_ic["n_splines"], peak_ic["IC_mean"]),
                 xytext=(peak_ic["n_splines"] + 1.5, peak_ic["IC_mean"] + 0.001),
                 fontsize=8, color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.6))
    ax2.annotate(f"Sharpe peak\nn={int(peak_sharpe['n_splines'])}",
                 xy=(peak_sharpe["n_splines"], peak_sharpe["LS_Sharpe"]),
                 xytext=(peak_sharpe["n_splines"] + 1.5, peak_sharpe["LS_Sharpe"] - 0.08),
                 fontsize=8, color="#ff7f0e",
                 arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.6))

    # Mark default
    ax1.axvline(10, ls=":", color="gray", alpha=0.5)
    ax1.text(10, df["IC_mean"].min() - 0.0005, "pygam\ndefault",
             ha="center", va="top", fontsize=8, color="gray")

    ax1.set_title("Rung 3a: GAM $n_{splines}$ sweep — peak at $n\\in\\{4,5\\}$, "
                  "default $n{=}10$ over-fits")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    _save(plt.gcf(), "fig_gam_n_splines")
    plt.close()


# ---------------------------------------------------------------------
def fig4_xgb_sensitivity():
    """XGBoost sensitivity grid — 3x3 heatmap + GKX callout."""
    df = pd.read_csv(ROOT / "output" / "rung3b_sensitivity_grid.csv")
    pivot = df.pivot(index="max_depth", columns="learning_rate", values="LS_Sharpe")

    # GKX-tuned (from rung3b_audit_log.txt or rung3b_gkx_tuned.csv)
    gkx_path = ROOT / "output" / "rung3b_gkx_tuned.csv"
    gkx_sharpe = None
    if gkx_path.exists():
        gkx = pd.read_csv(gkx_path)
        if "LS_Sharpe" in gkx.columns and len(gkx):
            gkx_sharpe = float(gkx["LS_Sharpe"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis",
                cbar_kws={"label": "LS Sharpe"}, linewidths=0.5,
                annot_kws={"size": 11, "color": "white"}, ax=ax)
    ax.set_xlabel("learning rate $\\eta$")
    ax.set_ylabel("max_depth")
    ax.invert_yaxis()  # depth=1 at top

    title = "Rung 3b: XGBoost sensitivity (LS Sharpe)\n"
    title += "9-grid: depth$\\in${1,4,6} $\\times$ $\\eta\\in${0.01,0.05,0.1}, $T{=}300$ trees"
    if gkx_sharpe is not None:
        title += (f"\nGKX-tuned (depth=1, $\\eta{{=}}$0.01, $T{{=}}$1000): "
                  f"\\textbf{{Sharpe = {gkx_sharpe:.3f}}}")
    ax.set_title(title)

    plt.tight_layout()
    _save(plt.gcf(), "fig_xgb_sensitivity")
    plt.close()


# ---------------------------------------------------------------------
def fig5_mlp_seeds():
    """MLP seed sensitivity bar chart."""
    df = pd.read_csv(ROOT / "output" / "rung5_seed_sensitivity.csv")
    df = df.sort_values("seed").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(df["seed"].astype(str), df["LS_Sharpe"],
                   color="#4c72b0", alpha=0.8, edgecolor="black", linewidth=0.6)

    # Reference lines
    ax.axhline(RIDGE_FLOOR_SHARPE, ls="--", color="#ff7f0e",
               linewidth=1.5, label=f"Ridge floor ({RIDGE_FLOOR_SHARPE:.3f})")
    ax.axhline(ZERO_PRED_SHARPE, ls="--", color="gray",
               linewidth=1, alpha=0.6, label=f"Zero predictor ({ZERO_PRED_SHARPE:.3f})")
    mean_sharpe = df["LS_Sharpe"].mean()
    ax.axhline(mean_sharpe, ls=":", color="#4c72b0",
               linewidth=1.5, label=f"5-seed mean ({mean_sharpe:.3f})")

    # Annotate per-bar IC
    for bar, ic in zip(bars, df["IC_mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"IC={ic:+.4f}", ha="center", va="bottom", fontsize=8)

    # SNR annotation
    ic_mean, ic_std = df["IC_mean"].mean(), df["IC_mean"].std()
    snr = abs(ic_mean) / max(ic_std, 1e-12)
    ax.text(0.02, 0.95,
            f"|mean|/std (SNR)\n= {abs(ic_mean):.4f} / {ic_std:.4f}\n= {snr:.2f}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=9)

    ax.set_xlabel("Random seed")
    ax.set_ylabel("LS Sharpe (annualized)")
    ax.set_title("Rung 4 MLP — All 5 seeds < Ridge floor (signal barely above init noise)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(df["LS_Sharpe"].max(), RIDGE_FLOOR_SHARPE) * 1.15)

    plt.tight_layout()
    _save(plt.gcf(), "fig_mlp_seeds")
    plt.close()


# ---------------------------------------------------------------------
def fig6_cpcv_sharpe():
    """CPCV 15-path Sharpe distribution boxplot."""
    df = pd.read_parquet(ROOT / "output" / "cpcv_results.parquet")

    # Sort models by mean Sharpe (highest first)
    order = (df.groupby("model")["Sharpe"].mean()
               .sort_values(ascending=False).index.tolist())

    # Tidy model labels
    label_map = {
        "1a_OLS": "1a OLS",
        "1b_IC_Ensemble": "1b IC-Ens.",
        "1c_FamaMacBeth": "1c FM",
        "1d_Barra": "1d Barra",
        "2a_LASSO": "2a LASSO",
        "2b_Ridge": "2b Ridge",
        "2d_ElasticNet": "2d ElNet",
        "3b_XGB_GKX": "3b XGB-GKX",
    }
    df = df.copy()
    df["model_lbl"] = df["model"].map(label_map).fillna(df["model"])
    order_lbl = [label_map.get(m, m) for m in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="model_lbl", y="Sharpe", order=order_lbl,
                ax=ax, color="#4c72b0", width=0.55, fliersize=0)
    sns.stripplot(data=df, x="model_lbl", y="Sharpe", order=order_lbl,
                  color="black", size=3.5, alpha=0.55, jitter=0.15, ax=ax)

    # Reference lines
    ax.axhline(0, ls="--", color="red", linewidth=1.2, alpha=0.6, label="Sharpe = 0")
    ax.axhline(RIDGE_FLOOR_SHARPE, ls="--", color="#ff7f0e",
               linewidth=1.2, alpha=0.7, label=f"Walk-fwd Ridge ({RIDGE_FLOOR_SHARPE:.2f})")

    # Mean markers
    means = df.groupby("model_lbl")["Sharpe"].mean().reindex(order_lbl)
    ax.scatter(range(len(order_lbl)), means.values, marker="D", color="white",
               edgecolor="black", s=50, zorder=5, label="Mean")

    # Highlight top-6 overlap
    ax.axhspan(means.iloc[5], means.iloc[0], alpha=0.08, color="green",
               label="Top-6 mean range (overlap)")

    ax.set_xlabel("")
    ax.set_ylabel("LS Sharpe (annualized) — 15 CPCV paths per model")
    ax.set_title("CPCV 15-path Sharpe distribution — top-6 5-95% bands all overlap")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    _save(plt.gcf(), "fig_cpcv_sharpe")
    plt.close()


# ---------------------------------------------------------------------
def fig7_pareto_with_bounds():
    """Pareto plot — IC vs Sharpe with CPCV uncertainty bands."""
    df = pd.read_csv(ROOT / "output" / "cpcv_summary.csv")

    label_map = {
        "1a_OLS": "1a OLS", "1b_IC_Ensemble": "1b IC-Ens.",
        "1c_FamaMacBeth": "1c FM", "1d_Barra": "1d Barra",
        "2a_LASSO": "2a LASSO", "2b_Ridge": "2b Ridge",
        "2d_ElasticNet": "2d ElNet", "3b_XGB_GKX": "3b XGB-GKX",
    }
    df = df.copy()
    # cpcv_summary uses 'label' column (not 'model')
    model_col = "label" if "label" in df.columns else "model"
    df["lbl"] = df[model_col].map(label_map).fillna(df[model_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(df["IC_mean"], df["Sharpe_mean"],
                xerr=df.get("IC_std", 0), yerr=df["Sharpe_std"],
                fmt="o", capsize=3, alpha=0.65, color="#4c72b0",
                markersize=8, ecolor="gray", linewidth=1.4,
                label="CPCV $\\pm 1\\sigma$ (15 paths)")

    # Annotate each model
    for _, r in df.iterrows():
        ax.annotate(r["lbl"], (r["IC_mean"], r["Sharpe_mean"]),
                    fontsize=9, textcoords="offset points",
                    xytext=(7, 6), color="black")

    # Reference points
    ax.scatter([ZERO_PRED_IC], [ZERO_PRED_SHARPE], marker="x", s=80,
               color="gray", linewidths=2, label="Zero predictor")
    ax.scatter([RIDGE_FLOOR_IC], [RIDGE_FLOOR_SHARPE], marker="s", s=80,
               color="#ff7f0e", label="Walk-fwd Ridge (single path)")

    ax.set_xlabel("IC mean (CPCV 15-path)")
    ax.set_ylabel("LS Sharpe (CPCV 15-path)")
    ax.set_title("Pareto plot with CPCV uncertainty — error bars show all top models overlap")
    ax.axhline(0, color="gray", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="gray", alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save(plt.gcf(), "fig_pareto_with_bounds")
    plt.close()


# ---------------------------------------------------------------------
def fig8_cumret_optional():
    """OPTIONAL: Cumulative LS portfolio return curves for top 5 models."""
    # Only run if results parquet files exist
    candidates = {
        "1c FM":     "results_1c_FamaMacBeth_v3.parquet",
        "2a LASSO":  "results_2a_LASSO_v3.parquet",
        "2b Ridge":  "results_2b_Ridge_v3.parquet",
        "2d ElNet":  "results_2d_ElasticNet_v3.parquet",
        "1a OLS":    "results_1a_OLS_v3.parquet",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for label, fname in candidates.items():
        path = ROOT / "output" / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        try:
            df = pd.read_parquet(path)
            # Build LS quintile portfolio per month
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            ls_returns = []
            for date, group in df.groupby("date"):
                if len(group) < 10:
                    continue
                q_lo, q_hi = group["y_pred"].quantile([0.2, 0.8])
                long = group[group["y_pred"] >= q_hi]["y_true"].mean()
                short = group[group["y_pred"] <= q_lo]["y_true"].mean()
                ls_returns.append({"date": date, "ls_ret": long - short})
            ls_df = pd.DataFrame(ls_returns).sort_values("date")
            ls_df["cumret"] = (1 + ls_df["ls_ret"]).cumprod()
            ax.plot(ls_df["date"], ls_df["cumret"], label=label, linewidth=1.6)
            plotted += 1
        except Exception as e:
            print(f"  [err] {fname}: {e}")

    if plotted == 0:
        print("[skip] fig_cumret.pdf — no input parquet files")
        plt.close()
        return

    ax.axhline(1, ls=":", color="gray", alpha=0.4)
    ax.set_xlabel("Test month")
    ax.set_ylabel("Cumulative LS quintile portfolio return")
    ax.set_title("Long-Short quintile portfolio cumulative returns (top 5 models, walk-forward)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    _save(plt.gcf(), "fig_cumret")
    plt.close()


# ---------------------------------------------------------------------
def main():
    print(f"Generating figures into {OUT}/")
    fig3_gam_n_splines()
    fig4_xgb_sensitivity()
    fig5_mlp_seeds()
    fig6_cpcv_sharpe()
    fig7_pareto_with_bounds()
    fig8_cumret_optional()
    print("\nAll figures generated.")
    print(f"Output dir: {OUT}/")


if __name__ == "__main__":
    main()
