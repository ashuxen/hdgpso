"""Claim-check v4. HDGPSO with UCB acquisition, lower surrogate threshold, and restart.

The task configuration is identical to v3 (189 cells); only the
algorithm side of HDGPSO is changed:

  - The surrogate filter now uses tree-variance UCB (LCB for
    minimization) with kappa = 1.5 instead of pure exploit.
  - The surrogate minimum history is lowered from 12 points to 6,
    which activates the surrogate filter earlier in the run.
  - A stagnation-triggered restart is added. After 2 iterations
    without an improvement in best loss, the worst 50% of the
    population is randomized.

Results are written to ``results_claim_check_v4/``.
"""
from __future__ import annotations

import os
import sys
import time
from io import StringIO

import pandas as pd

from benchmark import run_benchmark
from stats import (
    bootstrap_rank_ci,
    cd_diagram,
    friedman_test,
    hdgpso_vs_baselines_table,
    build_rank_matrix,
)

OUT_DIR = "results_claim_check_v4"


def run():
    t0 = time.time()
    summary, history = run_benchmark(
        budget=60,
        seeds=[0, 1, 2],
        datasets=["breast_cancer", "wine", "diabetes", "pinn_heat"],
        models=["RandomForest", "GradientBoosting", "MLP", "PINN-Heat"],
        tuners=["RandomSearch", "GridSearch", "Bayes", "Optuna-TPE", "DE", "PSO", "HDGPSO"],
        cv=3,
        population_size=5,
        include_openml=False,
        include_deep=True,
        out_dir=OUT_DIR,
        verbose=True,
    )
    print(f"\nTotal benchmark time: {(time.time()-t0)/60:.1f} min")
    return summary, history


def check_claims(summary):
    report = {}
    log = StringIO()

    def out(s=""):
        print(s)
        log.write(s + "\n")

    out("=" * 70)
    out("PAPER CLAIM CHECK v4 (HDGPSO with UCB + threshold=6 + restart)")
    out("=" * 70)

    ranks = build_rank_matrix(summary)
    mean_ranks = ranks.mean(axis=0).sort_values()
    out("\nMean ranks (lower=better):")
    for t, r in mean_ranks.items():
        marker = "  <-- HDGPSO" if t == "HDGPSO" else ""
        out(f"  {t:14s}  {r:.3f}{marker}")

    fr = friedman_test(summary, alpha=0.05)
    claim1 = fr.reject_null
    out(f"\nClaim 1: Friedman rejects H0: p={fr.pvalue:.4g} -> {'PASS' if claim1 else 'FAIL'}")
    report["claim1"] = claim1

    top = mean_ranks.index[0]
    claim2 = top == "HDGPSO"
    out(f"\nClaim 2: HDGPSO has lowest mean rank -> {'PASS' if claim2 else f'FAIL (winner: {top})'}")
    report["claim2"] = claim2

    vs = hdgpso_vs_baselines_table(summary, target="HDGPSO", alpha=0.05)
    nemenyi_wins = vs[vs["nemenyi_significant"]]
    claim3 = len(nemenyi_wins) >= 1
    out(f"\nClaim 3: HDGPSO Nemenyi-beats >=1 baseline -> {'PASS' if claim3 else 'FAIL'}")
    out(f"  CD={vs['CD_at_alpha'].iloc[0]:.3f}, beats: {list(nemenyi_wins['baseline'])}")
    report["claim3"] = claim3

    under = vs[vs["rank_delta"] > 0]
    bad_p = under[under["wilcoxon_p"] >= 0.05]
    claim4 = (len(under) > 0) and (len(bad_p) == 0)
    out(f"\nClaim 4: Wilcoxon p<0.05 vs each underperformer -> {'PASS' if claim4 else 'FAIL'}")
    for _, row in vs.iterrows():
        if row["rank_delta"] > 0:
            tag = "p OK" if row["wilcoxon_p"] < 0.05 else "p FAIL"
            out(f"  {row['baseline']:14s}  rank_delta={row['rank_delta']:+.3f}  "
                f"p={row['wilcoxon_p']:.4g}  [{tag}]")
        else:
            out(f"  {row['baseline']:14s}  rank_delta={row['rank_delta']:+.3f}  "
                f"[HDGPSO does NOT beat]")
    report["claim4"] = claim4

    weak = under[under["cliffs_delta"].abs() <= 0.147]
    claim5 = (len(under) > 0) and (len(weak) == 0)
    out(f"\nClaim 5: |Cliff's delta| > 0.147 vs each underperformer -> "
        f"{'PASS' if claim5 else 'FAIL'}")
    for _, row in vs.iterrows():
        if row["rank_delta"] > 0:
            sz = abs(row["cliffs_delta"])
            band = ("large" if sz >= 0.474 else
                    "medium" if sz >= 0.33 else
                    "small" if sz >= 0.147 else "negligible")
            out(f"  {row['baseline']:14s}  delta={row['cliffs_delta']:+.3f}  ({band})")
    report["claim5"] = claim5

    out("\n--- Per (dataset, model) mean rank of HDGPSO ---")
    by_task = (
        ranks.reset_index().melt(id_vars=['dataset', 'model', 'seed'],
                                  var_name='tuner', value_name='rank')
        .groupby(['dataset', 'model', 'tuner'])['rank'].mean().reset_index()
    )
    pivot = by_task.pivot_table(index=['dataset', 'model'], columns='tuner', values='rank')
    out(pivot.round(2).to_string())

    out("\n" + "=" * 70)
    passed = sum(1 for v in report.values() if v)
    out(f"OVERALL: {passed}/5 claims pass")
    out("=" * 70)
    return report, log.getvalue()


def main():
    summary, history = run()
    report, log_text = check_claims(summary)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "claims_report.txt"), "w") as f:
        f.write(log_text)

    import matplotlib
    matplotlib.use("Agg")
    cd_diagram(summary, save_path=os.path.join(OUT_DIR, "fig_cd_diagram.png"),
               title=f"CD diagram v4 (n={len(build_rank_matrix(summary))} cells)")
    bootstrap_rank_ci(summary, n_boot=2000, seed=0).to_csv(
        os.path.join(OUT_DIR, "rank_bootstrap_ci.csv"), index=False
    )

    failed = [k for k, v in report.items() if not v]
    if failed:
        print(f"\nFailed claims: {failed}")
        sys.exit(2)
    print("\nAll 5 claims pass!")


if __name__ == "__main__":
    main()
