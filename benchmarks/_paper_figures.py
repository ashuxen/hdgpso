"""Generate paper-ready figures as PDF (vector) for the IEEE conference paper.

Outputs to paper/figures/:
  - fig3_bar_ranks.pdf       Horizontal bar chart with bootstrap CI error bars (Fig. 3)
  - fig4_rank_vs_budget.pdf  Line plot of rank vs evaluation budget (Fig. 4)
  - fig5_heatmap_per_task.pdf Per-task rank heatmap (Fig. 5)

Reads:
  ../results_claim_check_v5/summary.csv
  ../results_budget_sweep/summary_budget_sweep.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from hdgpso.stats import build_rank_matrix, bootstrap_rank_ci

OUT = "../paper/figures"
os.makedirs(OUT, exist_ok=True)

# IEEE conference column widths (in inches): single column ~3.5, full text ~7.16
COL_W = 3.45
TEXT_W = 7.16

# Consistent color scheme across all figures
TUNER_COLORS = {
    "HDGPSO":       "#d62728",  # red
    "HDGPSO-MF":    "#9467bd",  # purple
    "Bayes":        "#1f77b4",  # blue
    "Optuna-TPE":   "#17becf",  # cyan
    "DE":           "#2ca02c",  # green
    "PSO":          "#ff7f0e",  # orange
    "RandomSearch": "#7f7f7f",  # gray
    "GridSearch":   "#000000",  # black
}

# ============================================================
# Load data
# ============================================================

v5 = pd.read_csv("../results_claim_check_v5/summary.csv")
sweep = pd.read_csv("../results_budget_sweep/summary_budget_sweep.csv")

# Exclude HDGPSO-MF from analysis: it is presented as future work, not a headline result.
v5 = v5[v5["tuner"] != "HDGPSO-MF"].reset_index(drop=True)
sweep = sweep[sweep["tuner"] != "HDGPSO-MF"].reset_index(drop=True)

# ============================================================
# Fig. 3: Horizontal bar chart with bootstrap CI
# ============================================================

ranks_v5 = build_rank_matrix(v5)
mean_ranks = ranks_v5.mean(axis=0).sort_values()
boot = bootstrap_rank_ci(v5, n_boot=2000, seed=0)
boot = boot.set_index("tuner").loc[mean_ranks.index]

fig, ax = plt.subplots(figsize=(COL_W, 2.8))
y_positions = np.arange(len(mean_ranks))
colors = [TUNER_COLORS[t] for t in mean_ranks.index]

# Asymmetric error bars from the bootstrap CI
err_lo = mean_ranks.values - boot["ci_lo_95"].values
err_hi = boot["ci_hi_95"].values - mean_ranks.values
xerr = np.vstack([err_lo, err_hi])

bars = ax.barh(
    y_positions, mean_ranks.values, xerr=xerr,
    color=colors, edgecolor="black", linewidth=0.6,
    error_kw={"capsize": 2.5, "elinewidth": 1.0, "ecolor": "black"},
    alpha=0.85,
)

# Value labels at the end of each bar
for i, (val, bar) in enumerate(zip(mean_ranks.values, bars)):
    ax.text(val + 0.15 + xerr[1][i], i, f"{val:.2f}",
            va="center", ha="left", fontsize=7)

ax.set_yticks(y_positions)
ax.set_yticklabels(mean_ranks.index, fontsize=8)
ax.set_xlabel("Mean rank across 27 cells (lower = better)", fontsize=8)
ax.tick_params(axis="x", labelsize=7)
ax.set_xlim(0, 9.0)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Invert so best (lowest rank) appears at top
ax.invert_yaxis()

plt.tight_layout(pad=0.4)
plt.savefig(os.path.join(OUT, "fig3_bar_ranks.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.savefig(os.path.join(OUT, "fig3_bar_ranks.png"), bbox_inches="tight", pad_inches=0.05, dpi=220)
plt.close()
print(f"Wrote {OUT}/fig3_bar_ranks.{{pdf,png}}")

# ============================================================
# Fig. 4: Line plot of rank vs budget
# ============================================================

# Build long-format per-budget mean rank table
rows = []
for b in sorted(sweep["budget"].unique()):
    sub = sweep[sweep["budget"] == b]
    rm = build_rank_matrix(sub)
    for t, r in rm.mean(axis=0).items():
        rows.append({"budget": int(b), "tuner": t, "mean_rank": r})
rb_long = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(COL_W, 2.6))

tuner_order = ["HDGPSO", "Bayes", "Optuna-TPE",
               "PSO", "DE", "RandomSearch", "GridSearch"]

for t in tuner_order:
    sub = rb_long[rb_long["tuner"] == t].sort_values("budget")
    lw = 2.2 if t in ("HDGPSO", "HDGPSO-MF") else 1.1
    marker = {"HDGPSO": "o", "HDGPSO-MF": "s", "Bayes": "^", "Optuna-TPE": "D",
              "PSO": "o", "DE": "v", "RandomSearch": "s", "GridSearch": "x"}.get(t, "o")
    linestyle = "--" if t == "GridSearch" else "-"
    ax.plot(
        sub["budget"], sub["mean_rank"],
        color=TUNER_COLORS[t], linewidth=lw, marker=marker, markersize=4.5,
        linestyle=linestyle, label=t,
    )

ax.invert_yaxis()
ax.set_xticks([20, 40, 60, 100])
ax.set_xlabel("Function-evaluation budget", fontsize=8)
ax.set_ylabel("Mean rank (lower = better)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.30),
    ncol=4, fontsize=6.5, frameon=False, columnspacing=0.8, handletextpad=0.4,
)

plt.tight_layout(pad=0.4)
plt.savefig(os.path.join(OUT, "fig4_rank_vs_budget.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.savefig(os.path.join(OUT, "fig4_rank_vs_budget.png"), bbox_inches="tight", pad_inches=0.05, dpi=220)
plt.close()
print(f"Wrote {OUT}/fig4_rank_vs_budget.{{pdf,png}}")

# ============================================================
# Fig. 5: Per-task heatmap
# ============================================================

# Per-(dataset, model) mean rank across seeds
v5_ranks = build_rank_matrix(v5)
v5_ranks_reset = v5_ranks.reset_index()  # makes dataset/model/seed real columns

# Average across seeds within each (dataset, model)
per_task = (
    v5_ranks_reset
    .melt(id_vars=["dataset", "model", "seed"], var_name="tuner", value_name="rank")
    .groupby(["dataset", "model", "tuner"])["rank"]
    .mean()
    .unstack("tuner")
)

# Order columns by overall mean rank (best left)
col_order = ["HDGPSO", "Bayes", "Optuna-TPE",
             "DE", "PSO", "RandomSearch", "GridSearch"]
per_task = per_task[col_order]

# Order rows: classification tasks first, then regression, then PINN
task_order = [
    ("breast_cancer", "GradientBoosting"),
    ("breast_cancer", "RandomForest"),
    ("breast_cancer", "MLP"),
    ("wine", "GradientBoosting"),
    ("wine", "RandomForest"),
    ("wine", "MLP"),
    ("diabetes", "GradientBoosting"),
    ("diabetes", "RandomForest"),
    ("pinn_heat", "PINN-Heat"),
]
per_task = per_task.reindex(task_order)

# Custom green→yellow→red colormap
cmap = LinearSegmentedColormap.from_list(
    "GnYlRd",
    [(0.0, "#1a9850"), (0.25, "#a6d96a"), (0.5, "#ffffbf"),
     (0.75, "#fdae61"), (1.0, "#d73027")],
)

fig, ax = plt.subplots(figsize=(TEXT_W, 3.4))
im = ax.imshow(per_task.values, cmap=cmap, aspect="auto", vmin=1, vmax=7)

# Annotate each cell with its value
for i in range(per_task.shape[0]):
    for j in range(per_task.shape[1]):
        v = per_task.values[i, j]
        # Pick text color for contrast
        text_color = "white" if (v < 1.8 or v > 5.8) else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold")

# Y-axis: task labels
ax.set_yticks(np.arange(len(per_task)))
ax.set_yticklabels(
    [f"{ds} / {mdl}" for (ds, mdl) in per_task.index],
    fontsize=8,
)

# X-axis: tuner labels (slightly rotated to avoid overlap)
ax.set_xticks(np.arange(len(per_task.columns)))
ax.set_xticklabels(per_task.columns, fontsize=8, rotation=20, ha="right",
                   rotation_mode="anchor")

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Mean rank", fontsize=8)
cbar.ax.tick_params(labelsize=7)
cbar.set_ticks([1, 2, 3, 4, 5, 6, 7])

# Grid lines between cells for readability
ax.set_xticks(np.arange(-0.5, len(per_task.columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(per_task), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", length=0)

plt.tight_layout(pad=0.4)
plt.savefig(os.path.join(OUT, "fig5_heatmap_per_task.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.savefig(os.path.join(OUT, "fig5_heatmap_per_task.png"), bbox_inches="tight", pad_inches=0.05, dpi=220)
plt.close()
print(f"Wrote {OUT}/fig5_heatmap_per_task.{{pdf,png}}")

print(f"\nAll three paper figures written to {OUT}/")
