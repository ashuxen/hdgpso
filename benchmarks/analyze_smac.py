"""Re-rank the main run with SMAC added, using the paper's own stats module.

Reports mean ranks at K=7 and K=8, Friedman, Nemenyi CD, Wilcoxon, Cliff's
delta and bootstrap CIs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# In the repo this sits in benchmarks/ with hdgpso pip-installed, so no path
# setup is needed; src_snapshot only exists in the standalone revision folder.
_SNAP = os.path.join(HERE, "src_snapshot")
if os.path.isdir(_SNAP):
    sys.path.insert(0, _SNAP)

from hdgpso.stats import (          # noqa: E402
    bootstrap_rank_ci,
    build_rank_matrix,
    cliffs_delta,
    critical_difference,
    friedman_test,
)
from scipy.stats import wilcoxon    # noqa: E402

def _find(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(candidates)


ROOT = os.path.dirname(HERE)
PUB = _find(os.path.join(ROOT, "results_claim_check_v5", "summary.csv"),
            os.path.join(HERE, "results", "published_summary_v5.csv"))
SMAC = _find(os.path.join(ROOT, "results_smac", "summary.csv"),
             os.path.join(HERE, "results", "smac_b60.csv"))
OUTDIR = os.path.dirname(SMAC)


def main():
    pub = pd.read_csv(PUB)
    pub = pub[pub.tuner != "HDGPSO-MF"]          # as the paper does
    smac = pd.read_csv(SMAC)

    keep = ["dataset", "model", "tuner", "seed", "best_loss"]
    k7 = pub[keep].copy()
    k8 = pd.concat([k7, smac[keep]], ignore_index=True)

    n_cells = k8.groupby("tuner").size()
    if n_cells.nunique() != 1:
        print("WARNING: unbalanced blocks — SMAC may be incomplete")
        print(n_cells.to_string())
        return

    r7 = build_rank_matrix(k7).mean().sort_values()
    r8 = build_rank_matrix(k8).mean().sort_values()
    N = len(build_rank_matrix(k8))

    print("=" * 74)
    print(f"MAIN RUN, budget=60, N={N} blocks")
    print("=" * 74)
    print(f"\n{'tuner':<14}{'K=7 (paper)':>13}{'K=8 (+SMAC)':>13}{'shift':>9}")
    print("-" * 74)
    for t in r8.index:
        before = r7.get(t, np.nan)
        shift = r8[t] - before if t in r7 else np.nan
        b = f"{before:.3f}" if t in r7 else "--"
        s = f"{shift:+.3f}" if t in r7 else "--"
        mark = "  <-- SMAC" if t == "SMAC" else ""
        print(f"{t:<14}{b:>13}{r8[t]:>13.3f}{s:>9}{mark}")

    # ---- headline survival ------------------------------------------------
    ranks8 = build_rank_matrix(k8)
    p = {t: float((ranks8["SMAC"] < ranks8[t]).mean())
         for t in ranks8.columns if t != "SMAC"}
    gap = 0.2222
    delta = p["HDGPSO"] - p["Optuna-TPE"]

    print(f"\n{'-'*74}\nHEADLINE SURVIVAL\n{'-'*74}")
    print(f"  p_i = fraction of {N} blocks where SMAC outranks tuner i")
    for t in ["HDGPSO", "Optuna-TPE", "Bayes"]:
        print(f"    p_{t:<12} = {p[t]:.3f}")
    print(f"\n  p_HDGPSO - p_Optuna = {delta:+.3f}   (must be < {gap:.3f})")
    lead = r8["HDGPSO"] < r8["Optuna-TPE"] and r8["HDGPSO"] == r8.min()
    print(f"  -> HDGPSO still lowest mean rank: {'YES' if lead else 'NO'}")

    # ---- omnibus + post-hoc ----------------------------------------------
    fr7, fr8 = friedman_test(k7), friedman_test(k8)
    cd7 = critical_difference(7, N)
    cd8 = critical_difference(8, N)
    print(f"\n{'-'*74}\nSTATISTICS\n{'-'*74}")
    print(f"  Friedman K=7 : chi2={fr7.statistic:.2f}  p={fr7.pvalue:.3g}")
    print(f"  Friedman K=8 : chi2={fr8.statistic:.2f}  p={fr8.pvalue:.3g}")
    print(f"  Nemenyi CD   : K=7 {cd7:.3f}  ->  K=8 {cd8:.3f}   "
          f"({100*(cd8-cd7)/cd7:+.1f}%)")

    best = r8.index[0]
    print(f"\n  Nemenyi vs {best} (K=8, CD={cd8:.3f}):")
    for t in r8.index[1:]:
        d = r8[t] - r8[best]
        print(f"    {t:<14} delta={d:.3f}  {'SEPARATED' if d > cd8 else 'tie'}")

    # ---- pairwise vs HDGPSO ----------------------------------------------
    piv = k8.pivot_table(index=["dataset", "model", "seed"],
                         columns="tuner", values="best_loss").dropna()
    print(f"\n  Wilcoxon / Cliff's delta vs HDGPSO (paired best_loss):")
    for t in r8.index:
        if t == "HDGPSO":
            continue
        try:
            _, pv = wilcoxon(piv["HDGPSO"], piv[t])
        except ValueError:
            pv = float("nan")
        # Argument order matters: stats.cliffs_delta is delta(x, y) = P(x<y) - P(x>y)
        # with x the target, so HDGPSO goes first and positive means HDGPSO wins.
        d = cliffs_delta(piv["HDGPSO"].values, piv[t].values)
        print(f"    {t:<14} p={pv:<10.4g} cliff_delta={d:+.3f}")

    boot = bootstrap_rank_ci(k8, n_boot=2000, seed=0)
    print(f"\n  Bootstrap 95% CI on mean rank (K=8):")
    print(boot.to_string(index=False))

    # ---- Table VI, exactly as it appears in the paper ---------------------
    b = boot.set_index("tuner")
    print(f"\n{'='*74}\nTABLE VI  (paper-ready)\n{'='*74}")
    print(f"{'Tuner':<14}{'Rank':>6}{'95% CI':>16}{'Wilcoxon p':>13}"
          f"{'Cliff d':>9}{'Nem.':>6}")
    print("-" * 74)
    for t in r8.index:
        lo, hi = b.loc[t, "ci_lo_95"], b.loc[t, "ci_hi_95"]
        if t == "HDGPSO":
            print(f"{t:<14}{r8[t]:>6.2f}   [{lo:.2f}, {hi:.2f}]{'---':>13}"
                  f"{'---':>9}{'---':>6}")
            continue
        _, pv = wilcoxon(piv["HDGPSO"], piv[t])
        d = cliffs_delta(piv["HDGPSO"].values, piv[t].values)
        gap = r8[t] - r8["HDGPSO"]
        nem = "yes" if gap > cd8 else ("no" if gap > cd8 - 0.15 else "tie")
        print(f"{t:<14}{r8[t]:>6.2f}   [{lo:.2f}, {hi:.2f}]{pv:>13.2g}"
              f"{d:>+9.2f}{nem:>6}")
    print("-" * 74)
    print(f"  Friedman chi2={fr8.statistic:.2f}, p={fr8.pvalue:.2g}; "
          f"Nemenyi CD={cd8:.2f} at alpha=0.05, K=8, N={N}")
    print("  'no' = below the CD threshold but within 0.15 of it.")

    out = os.path.join(OUTDIR, "k8_mean_ranks.csv")
    pd.DataFrame({"tuner": r8.index, "mean_rank_k8": r8.values}).to_csv(out, index=False)
    boot.to_csv(os.path.join(OUTDIR, "k8_bootstrap_ci.csv"), index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
