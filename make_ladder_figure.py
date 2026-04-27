"""
Generate the 5-rung complexity ladder schematic for the paper (Fig 2).

Run:
    python make_ladder_figure.py

Outputs:
    paper/figures/fig_ladder.pdf
    paper/figures/fig_ladder.png

Standalone from `make_figures.py` so the design can iterate without
touching the data-driven plots.

Design:
    5 horizontal boxes (rungs 1 → 5) with model lists + param counts.
    Color gradient: cool (linear) → warm (deep learning).
    Connecting arrows between rungs labeled with "added complexity".
    Header annotation showing complexity dimensions covered.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Rung definitions ─────────────────────────────────────────────────
RUNGS = [
    {
        "id":    "Rung 1",
        "title": "Linear baselines",
        "models": [
            "1a  OLS",
            "1b  IC-Ensemble",
            "1c  Fama-MacBeth",
            "1d  Barra (G-K)",
        ],
        "params": "~27 weights",
        "color": "#3b6abc",     # blue
    },
    {
        "id":    "Rung 2",
        "title": "Regularized linear",
        "models": [
            "2a  LASSO",
            "2b  Ridge",
            "2d  Elastic Net",
            "2e  Adaptive LASSO",
        ],
        "params": "~27 + α",
        "color": "#52a071",     # teal
    },
    {
        "id":    "Rung 3",
        "title": "Smooth non-linear",
        "models": [
            "3a  GAM",
            "    (cubic splines)",
            "",
            "3b  XGBoost",
            "    (GKX-tuned)",
        ],
        "params": "140 (GAM)\n~50k (XGB)",
        "color": "#d9b454",     # gold
    },
    {
        "id":    "Rung 4",
        "title": "Single-task MLP",
        "models": [
            "MLP",
            "  (in→64→32→1)",
            "  ReLU + dropout",
            "  Huber loss",
            "",
        ],
        "params": "3{,}073 weights",
        "color": "#d4762e",     # orange
    },
    {
        "id":    "Rung 5",
        "title": "Multi-task + MoE",
        "models": [
            "5b  MTL ret+3m",
            "5c  MTL ret+vol",
            "5d  MTL all-3",
            "    Regime-MoE",
            "    Enhanced MoE",
        ],
        "params": "3{,}139+",
        "color": "#b54a4a",     # red
    },
]

# ─── Layout config ────────────────────────────────────────────────────
BOX_W   = 2.4
BOX_H   = 3.4
GAP_X   = 0.8
Y_BASE  = 0.5

ARROW_LABELS = [
    "+ regularization",
    "+ non-linearity",
    "+ deep capacity",
    "+ multi-task",
]


def main():
    n = len(RUNGS)
    total_w = n * BOX_W + (n - 1) * GAP_X + 1.0   # margins
    fig_w = total_w
    fig_h = BOX_H + 2.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Header line: complexity dimension ──────────────────────────────
    ax.annotate(
        "Increasing model complexity →",
        xy=(0.5, 0.96), xycoords="axes fraction",
        ha="center", va="top",
        fontsize=12, fontweight="bold",
        color="#333",
    )

    # ── Draw rung boxes ────────────────────────────────────────────────
    box_centers_x = []
    for i, rung in enumerate(RUNGS):
        x0 = 0.5 + i * (BOX_W + GAP_X)
        x_center = x0 + BOX_W / 2
        box_centers_x.append(x_center)

        # Header strip (color band)
        header_h = 0.55
        header = FancyBboxPatch(
            (x0, Y_BASE + BOX_H - header_h),
            BOX_W, header_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.4,
            edgecolor=rung["color"],
            facecolor=rung["color"],
            zorder=2,
        )
        ax.add_patch(header)

        # Body box
        body = FancyBboxPatch(
            (x0, Y_BASE),
            BOX_W, BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.4,
            edgecolor=rung["color"],
            facecolor="white",
            zorder=1,
        )
        ax.add_patch(body)

        # Header text (rung id + title)
        ax.text(
            x_center, Y_BASE + BOX_H - header_h / 2,
            f"{rung['id']}",
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="white", zorder=3,
        )

        # Title under header
        ax.text(
            x_center, Y_BASE + BOX_H - header_h - 0.32,
            rung["title"],
            ha="center", va="top",
            fontsize=9.5, fontstyle="italic",
            color="#222", zorder=3,
        )

        # Models list
        models_text = "\n".join(rung["models"])
        ax.text(
            x_center, Y_BASE + BOX_H - header_h - 0.85,
            models_text,
            ha="center", va="top",
            fontsize=9, family="monospace",
            color="#222", zorder=3,
        )

        # Param count footer
        ax.text(
            x_center, Y_BASE + 0.18,
            rung["params"],
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color=rung["color"], zorder=3,
        )

    # ── Connecting arrows between rungs ────────────────────────────────
    arrow_y = Y_BASE - 0.4
    for i in range(n - 1):
        x_start = box_centers_x[i] + BOX_W / 2 - 0.05
        x_end = box_centers_x[i + 1] - BOX_W / 2 + 0.05
        arrow = FancyArrowPatch(
            (x_start, arrow_y),
            (x_end, arrow_y),
            arrowstyle="-|>",
            mutation_scale=15,
            color="#666",
            linewidth=1.4,
            zorder=4,
        )
        ax.add_patch(arrow)

        # Label
        x_mid = (x_start + x_end) / 2
        ax.text(
            x_mid, arrow_y - 0.18,
            ARROW_LABELS[i],
            ha="center", va="top",
            fontsize=8.5, fontstyle="italic",
            color="#444",
        )

    # ── Final axes setup ───────────────────────────────────────────────
    ax.set_xlim(0, total_w)
    ax.set_ylim(arrow_y - 0.7, Y_BASE + BOX_H + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Save ───────────────────────────────────────────────────────────
    pdf_path = OUT / "fig_ladder.pdf"
    png_path = OUT / "fig_ladder.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[ok] {pdf_path.name} + {png_path.name}")


if __name__ == "__main__":
    main()
