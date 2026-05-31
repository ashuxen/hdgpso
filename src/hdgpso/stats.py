"""Publication-grade statistical analysis for multi-tuner benchmarks.

Implements the Demsar (2006) methodology for comparing N classifiers
across K datasets:

  1. Friedman omnibus test (reject "all equivalent")
  2. Nemenyi post-hoc pairwise comparisons
  3. Critical Difference (CD) diagram

Plus:
  - Bootstrap 95% CI on per-tuner mean rank
  - Cliff's delta effect size for HDGPSO vs each baseline
  - Median percentage improvement

Reference:
  Demsar, J. (2006). Statistical Comparisons of Classifiers over
  Multiple Data Sets. Journal of Machine Learning Research 7:1-30.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Nemenyi critical-value table (Demsar 2006, Table 5)
# Two-tailed Studentized range / sqrt(2), alpha = 0.05
# k = number of algorithms compared
# ---------------------------------------------------------------------------

_Q_ALPHA_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
    20: 3.544,
}

_Q_ALPHA_10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
    8: 2.780, 9: 2.855, 10: 2.920,
}


def critical_difference(k: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference: ranks differing by less than CD are NOT
    significantly different at the given alpha."""
    table = _Q_ALPHA_05 if alpha == 0.05 else _Q_ALPHA_10
    if k not in table:
        raise ValueError(f"No Nemenyi q value for k={k}, alpha={alpha}")
    q = table[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n_datasets))


# ---------------------------------------------------------------------------
# Build the per-dataset ranking matrix used by every Demsar tool
# ---------------------------------------------------------------------------


def build_rank_matrix(summary_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, seed); columns are tuners; cells are ranks.

    Lower rank = better (rank 1 is the best tuner on that cell).
    Ties get average ranks.
    """
    pivot = summary_df.pivot_table(
        index=["dataset", "model", "seed"],
        columns="tuner",
        values="best_loss",
        aggfunc="first",
    ).dropna()
    ranks = pivot.rank(axis=1, method="average")
    return ranks


# ---------------------------------------------------------------------------
# 1. Friedman test
# ---------------------------------------------------------------------------


@dataclass
class FriedmanResult:
    statistic: float
    pvalue: float
    n_datasets: int
    n_tuners: int
    reject_null: bool

    def __repr__(self):
        verdict = "REJECT H0" if self.reject_null else "fail to reject H0"
        return (
            f"Friedman: chi2={self.statistic:.3f}, p={self.pvalue:.4g}, "
            f"k={self.n_tuners} tuners over n={self.n_datasets} cells -> {verdict}"
        )


def friedman_test(summary_df: pd.DataFrame, alpha: float = 0.05) -> FriedmanResult:
    """Friedman omnibus test that all tuners perform equivalently.

    H0: all tuners have the same expected rank.
    Rejection (p < alpha) is required before reporting Nemenyi.
    """
    ranks = build_rank_matrix(summary_df)
    columns = [ranks[c].values for c in ranks.columns]
    stat, p = sps.friedmanchisquare(*columns)
    return FriedmanResult(
        statistic=float(stat),
        pvalue=float(p),
        n_datasets=len(ranks),
        n_tuners=ranks.shape[1],
        reject_null=p < alpha,
    )


# ---------------------------------------------------------------------------
# 2. Nemenyi post-hoc
# ---------------------------------------------------------------------------


def nemenyi_matrix(summary_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Nemenyi: cell [i, j] = 1 iff tuner i is significantly better
    than j (lower mean rank by more than CD). Cell = 0 if not distinguishable."""
    ranks = build_rank_matrix(summary_df)
    mean_ranks = ranks.mean(axis=0).sort_values()
    tuners = list(mean_ranks.index)
    k, n = len(tuners), len(ranks)
    cd = critical_difference(k, n, alpha)
    matrix = pd.DataFrame(0, index=tuners, columns=tuners)
    for i, ti in enumerate(tuners):
        for tj in tuners[i + 1 :]:
            diff = mean_ranks[tj] - mean_ranks[ti]
            if diff > cd:
                matrix.loc[ti, tj] = 1  # ti better than tj
                matrix.loc[tj, ti] = -1
    matrix.attrs["CD"] = cd
    matrix.attrs["alpha"] = alpha
    return matrix


def hdgpso_vs_baselines_table(
    summary_df: pd.DataFrame, target: str = "HDGPSO", alpha: float = 0.05
) -> pd.DataFrame:
    """One row per baseline. For each baseline reports:
       - mean rank delta (baseline - HDGPSO)
       - Wilcoxon signed-rank p-value
       - Cliff's delta effect size
       - 'Beats per CD?' (Nemenyi significant)
    """
    ranks = build_rank_matrix(summary_df)
    if target not in ranks.columns:
        raise ValueError(f"target {target!r} not in summary; have {list(ranks.columns)}")
    mean_ranks = ranks.mean(axis=0)
    n = len(ranks)
    k = ranks.shape[1]
    cd = critical_difference(k, n, alpha)

    pivot = summary_df.pivot_table(
        index=["dataset", "model", "seed"],
        columns="tuner",
        values="best_loss",
        aggfunc="first",
    ).dropna()

    rows = []
    target_losses = pivot[target].values
    for other in mean_ranks.index:
        if other == target:
            continue
        other_losses = pivot[other].values
        try:
            stat, wp = sps.wilcoxon(target_losses, other_losses)
        except ValueError:
            wp = np.nan
        diff = other_losses - target_losses
        med_pct = float(np.nanmedian(diff / (np.abs(other_losses) + 1e-12))) * 100
        delta = cliffs_delta(target_losses, other_losses)
        rank_delta = float(mean_ranks[other] - mean_ranks[target])
        rows.append(
            {
                "baseline": other,
                "mean_rank_target": float(mean_ranks[target]),
                "mean_rank_baseline": float(mean_ranks[other]),
                "rank_delta": rank_delta,
                "median_pct_improvement": med_pct,
                "cliffs_delta": delta,
                "wilcoxon_p": float(wp) if wp == wp else float("nan"),
                "nemenyi_significant": rank_delta > cd,
                "CD_at_alpha": cd,
            }
        )
    return pd.DataFrame(rows).sort_values("rank_delta", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Critical Difference diagram (Demsar 2006 Figure 1 style)
# ---------------------------------------------------------------------------


def cd_diagram(
    summary_df: pd.DataFrame,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render a Critical Difference diagram.

    Tuners are placed on a horizontal axis at their mean rank (best = left).
    Horizontal bars connect cliques of tuners that are NOT significantly
    different at the given alpha (rank difference < CD).
    """
    ranks = build_rank_matrix(summary_df)
    mean_ranks = ranks.mean(axis=0).sort_values()
    tuners = list(mean_ranks.index)
    k, n = len(tuners), len(ranks)
    cd = critical_difference(k, n, alpha)

    # Build cliques: each clique is a maximal set of consecutive (sorted by rank)
    # tuners whose total span is <= CD.
    cliques: List[List[str]] = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and mean_ranks[tuners[j + 1]] - mean_ranks[tuners[i]] <= cd:
            j += 1
        if j > i:
            cliques.append(tuners[i : j + 1])
        i += 1

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2.5 + 0.25 * k))
    else:
        fig = ax.figure

    min_r = float(np.floor(mean_ranks.min()))
    max_r = float(np.ceil(mean_ranks.max()))
    span = max(max_r - min_r, 1.0)
    # leave some margin
    min_r -= 0.2 * span
    max_r += 0.2 * span

    # Top axis: rank scale
    ax.set_xlim(min_r, max_r)
    ax.set_ylim(-(k + 4) * 0.4, 1.6)
    ax.axis("off")

    # CD bar at top
    cd_x0 = min_r + 0.1 * (max_r - min_r)
    cd_x1 = cd_x0 + cd
    ax.plot([cd_x0, cd_x1], [1.2, 1.2], "k", lw=2)
    ax.plot([cd_x0, cd_x0], [1.1, 1.3], "k", lw=2)
    ax.plot([cd_x1, cd_x1], [1.1, 1.3], "k", lw=2)
    ax.text((cd_x0 + cd_x1) / 2, 1.4, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=10)

    # Top ruler
    for r in np.arange(np.ceil(min_r), np.floor(max_r) + 1):
        ax.plot([r, r], [0.7, 0.85], "k", lw=1)
        ax.text(r, 0.95, f"{int(r)}", ha="center", va="bottom", fontsize=9)
    ax.plot([min_r, max_r], [0.7, 0.7], "k", lw=1)

    # Place tuners: half on left, half on right
    half = (k + 1) // 2
    left = list(reversed(tuners[:half]))   # best-ranked first; we draw downward
    right = tuners[half:]

    text_y_start = 0.2
    y_step = 0.5
    line_y_base = -0.2

    def draw_branch(tname, y, side):
        x_rank = mean_ranks[tname]
        # vertical line from rank axis (y=0.7) down to row y
        ax.plot([x_rank, x_rank], [0.7, y], "k", lw=1)
        if side == "left":
            ax.plot([x_rank, min_r + 0.05 * (max_r - min_r)], [y, y], "k", lw=1)
            ax.text(
                min_r + 0.04 * (max_r - min_r),
                y,
                f"{tname}  ({mean_ranks[tname]:.2f})",
                ha="right",
                va="center",
                fontsize=10,
            )
        else:
            ax.plot([x_rank, max_r - 0.05 * (max_r - min_r)], [y, y], "k", lw=1)
            ax.text(
                max_r - 0.04 * (max_r - min_r),
                y,
                f"({mean_ranks[tname]:.2f})  {tname}",
                ha="left",
                va="center",
                fontsize=10,
            )

    for idx, t in enumerate(left):
        draw_branch(t, text_y_start - idx * y_step - 0.3, "left")
    for idx, t in enumerate(right):
        draw_branch(t, text_y_start - idx * y_step - 0.3, "right")

    # Clique bars
    for ci, clique in enumerate(cliques):
        x0 = mean_ranks[clique[0]]
        x1 = mean_ranks[clique[-1]]
        y = line_y_base - (k + 1) * 0.05 - ci * 0.10
        ax.plot([x0 - 0.03, x1 + 0.03], [y, y], "k", lw=4, solid_capstyle="butt")

    if title:
        ax.set_title(title, fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Effect size: Cliff's delta
# ---------------------------------------------------------------------------


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta nonparametric effect size in [-1, 1].

    delta(x, y) = P(x < y) - P(x > y). For loss values where lower is
    better, a positive delta means x (target) tends to have lower losses
    than y (baseline). Common thresholds: |d|<0.147 negligible,
    <0.33 small, <0.474 medium, else large.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    # vectorized: count pairs
    diffs = x[:, None] - y[None, :]
    return float((np.sum(diffs < 0) - np.sum(diffs > 0)) / (nx * ny))


# ---------------------------------------------------------------------------
# Bootstrap CI on mean rank
# ---------------------------------------------------------------------------


def bootstrap_rank_ci(
    summary_df: pd.DataFrame,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap 95% CI on per-tuner mean rank.

    Resamples (dataset, model, seed) cells with replacement n_boot times,
    recomputes mean rank per tuner, and returns mean / lo / hi columns.
    """
    ranks = build_rank_matrix(summary_df)
    rng = np.random.default_rng(seed)
    n = len(ranks)
    boot_means = np.empty((n_boot, ranks.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = ranks.iloc[idx].mean(axis=0).values
    lo = np.quantile(boot_means, alpha / 2, axis=0)
    hi = np.quantile(boot_means, 1 - alpha / 2, axis=0)
    mean = ranks.mean(axis=0).values
    out = pd.DataFrame(
        {
            "tuner": ranks.columns,
            "mean_rank": mean,
            f"ci_lo_{int((1-alpha)*100)}": lo,
            f"ci_hi_{int((1-alpha)*100)}": hi,
        }
    ).sort_values("mean_rank").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Convenience: full Demsar report on stdout
# ---------------------------------------------------------------------------


def demsar_report(summary_df: pd.DataFrame, target: str = "HDGPSO", alpha: float = 0.05) -> Dict:
    """Print and return a full Demsar-style analysis."""
    fr = friedman_test(summary_df, alpha)
    print(fr)
    print()
    boot = bootstrap_rank_ci(summary_df)
    print("--- Mean rank with 95% bootstrap CI ---")
    print(boot.to_string(index=False))
    print()
    vs = hdgpso_vs_baselines_table(summary_df, target=target, alpha=alpha)
    print(f"--- {target} vs each baseline ---")
    print(vs.to_string(index=False))
    return {"friedman": fr, "bootstrap": boot, "vs_baselines": vs}
