"""Plot helpers that produce paper-ready figures from benchmark CSVs."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Consistent ordering / colors for the paper
TUNER_ORDER = [
    "GridSearch",
    "RandomSearch",
    "Bayes",
    "Optuna-TPE",
    "DE",
    "PSO",
    "HDGPSO",
]
TUNER_COLORS = {
    "GridSearch": "#888888",
    "RandomSearch": "#aaaaaa",
    "Bayes": "#1f77b4",
    "Optuna-TPE": "#17becf",
    "DE": "#2ca02c",
    "PSO": "#ff7f0e",
    "HDGPSO": "#d62728",
}


def _ordered(tuners):
    seen = set()
    return [t for t in TUNER_ORDER if t in tuners and not (t in seen or seen.add(t))]


def convergence_plot(
    history_df: pd.DataFrame,
    dataset: str,
    model: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot mean running-best loss against iteration, averaged over seeds, per tuner."""
    df = history_df.query("dataset == @dataset and model == @model").copy()
    if df.empty:
        raise ValueError(f"No history for {dataset}/{model}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    tuners = _ordered(df["tuner"].unique())
    for t in tuners:
        sub = df[df["tuner"] == t]
        # align trials per seed using cumulative running best
        curves = []
        for seed, g in sub.groupby("seed"):
            best = g["loss"].cummin().values
            curves.append(best)
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.stack(
            [np.concatenate([c, np.full(max_len - len(c), c[-1])]) for c in curves]
        )
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        x = np.arange(1, max_len + 1)
        ax.plot(x, mean, label=t, color=TUNER_COLORS.get(t), linewidth=1.5)
        ax.fill_between(
            x, mean - std, mean + std, color=TUNER_COLORS.get(t), alpha=0.15
        )

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best loss so far")
    ax.set_title(title or f"{dataset} / {model}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def convergence_grid(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """Render one convergence subplot per (dataset, model) cell."""
    cells = (
        history_df[["dataset", "model"]]
        .drop_duplicates()
        .sort_values(["dataset", "model"])
        .values.tolist()
    )
    n = len(cells)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for i, (ds, mdl) in enumerate(cells):
        ax = axes[i // cols][i % cols]
        convergence_plot(history_df, ds, mdl, ax=ax)
    # hide unused
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def best_loss_bar(
    summary_df: pd.DataFrame,
    dataset: str,
    model: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    df = summary_df.query("dataset == @dataset and model == @model").copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    agg = (
        df.groupby("tuner")["best_loss"]
        .agg(["mean", "std"])
        .reindex(_ordered(df["tuner"].unique()))
        .reset_index()
    )
    colors = [TUNER_COLORS.get(t, "#444") for t in agg["tuner"]]
    ax.bar(agg["tuner"], agg["mean"], yerr=agg["std"], color=colors, capsize=4)
    ax.set_ylabel("Best loss (mean ± std)")
    ax.set_title(f"{dataset} / {model}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    return ax


def mean_rank_bar(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean rank per tuner across all (dataset, model, seed) cells. Lower is better."""
    df = summary_df.copy()
    df["rank"] = df.groupby(["dataset", "model", "seed"])["best_loss"].rank(method="min")
    agg = (
        df.groupby("tuner")["rank"]
        .agg(["mean", "std"])
        .reindex(_ordered(df["tuner"].unique()))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [TUNER_COLORS.get(t, "#444") for t in agg["tuner"]]
    ax.bar(agg["tuner"], agg["mean"], yerr=agg["std"], color=colors, capsize=4)
    ax.set_ylabel("Mean rank (1 = best)")
    ax.set_title("Average rank across all (dataset, model) cells")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def wins_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of per-tuner wins across the (dataset, model, seed) cells."""
    df = summary_df.copy()
    df["rank"] = df.groupby(["dataset", "model", "seed"])["best_loss"].rank(method="min")
    wins = df[df["rank"] == 1.0].groupby("tuner").size().reset_index(name="wins")
    cells = df[["dataset", "model", "seed"]].drop_duplicates().shape[0]
    wins["pct"] = wins["wins"] / cells * 100
    return wins.sort_values("wins", ascending=False).reset_index(drop=True)
