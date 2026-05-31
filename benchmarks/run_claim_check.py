"""Run the claim-check benchmark and evaluate all 6 paper claims.

Config sized for ~1-1.5 hr on a typical workstation: 4 datasets x 2 models
x 7 tuners x 3 seeds = 168 cells. Statistically meaningful (n=24 per tuner)
while finishing in a single check loop.

Writes:
  results_claim_check/summary.csv
  results_claim_check/history.csv
  results_claim_check/claims_report.txt
"""
from __future__ import annotations

import os
import sys
import time
from io import StringIO

import numpy as np
import pandas as pd

from benchmark import run_benchmark, rank_table
from stats import (
    bootstrap_rank_ci,
    cd_diagram,
    critical_difference,
    friedman_test,
    hdgpso_vs_baselines_table,
    build_rank_matrix,
)

OUT_DIR = "results_claim_check"


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    summary, history = run_benchmark(
        budget=60,
        seeds=[0, 1, 2],
        datasets=["breast_cancer", "wine", "diabetes", "california_housing"],
        models=["RandomForest", "GradientBoosting"],
        tuners=["RandomSearch", "GridSearch", "Bayes", "Optuna-TPE", "DE", "PSO", "HDGPSO"],
        cv=3,
        population_size=5,
        include_openml=False,
        out_dir=OUT_DIR,
        verbose=True,
    )
    print(f"\nTotal benchmark time: {(time.time()-t0)/60:.1f} min")
    return summary, history


def check_claims(summary: pd.DataFrame) -> dict:
    """Evaluate the 6 paper claims and return a structured report."""
    report = {}
    log = StringIO()

    def out(s=""):
        print(s)
        log.write(s + "\n")

    out("=" * 70)
    out("PAPER CLAIM CHECK")
    out("=" * 70)

    # 1. Friedman
    fr = friedman_test(summary, alpha=0.05)
    claim1 = fr.reject_null
    out(f"\nClaim 1: Friedman rejects H0 at alpha=0.05")
    out(f"  chi2={fr.statistic:.3f}, p={fr.pvalue:.4g}, n={fr.n_datasets}, k={fr.n_tuners}")
    out(f"  -> {'PASS' if claim1 else 'FAIL'}")
    report["claim1_friedman"] = (claim1, fr.pvalue)

    # 2. Lowest mean rank
    ranks = build_rank_matrix(summary)
    mean_ranks = ranks.mean(axis=0).sort_values()
    top_tuner = mean_ranks.index[0]
    claim2 = top_tuner == "HDGPSO"
    out(f"\nClaim 2: HDGPSO has the lowest mean rank")
    out(f"  Mean ranks (lower = better):")
    for t, r in mean_ranks.items():
        marker = "  <-- HDGPSO" if t == "HDGPSO" else ""
        out(f"    {t:14s}  {r:.3f}{marker}")
    out(f"  Best: {top_tuner} (rank {mean_ranks.iloc[0]:.3f})")
    out(f"  -> {'PASS' if claim2 else 'FAIL'}")
    report["claim2_top_rank"] = (claim2, top_tuner, float(mean_ranks.get("HDGPSO", float("nan"))))

    # 3. Nemenyi significantly better than at least one named baseline
    vs = hdgpso_vs_baselines_table(summary, target="HDGPSO", alpha=0.05)
    nemenyi_wins = vs[vs["nemenyi_significant"]]
    claim3 = len(nemenyi_wins) >= 1
    out(f"\nClaim 3: HDGPSO is Nemenyi-significantly better than >=1 baseline")
    out(f"  CD at alpha=0.05: {vs['CD_at_alpha'].iloc[0]:.3f}")
    if len(nemenyi_wins) > 0:
        out(f"  Nemenyi-beats: {list(nemenyi_wins['baseline'])}")
    else:
        out(f"  Nemenyi-beats: none")
    out(f"  -> {'PASS' if claim3 else 'FAIL'}")
    report["claim3_nemenyi"] = (claim3, list(nemenyi_wins["baseline"]))

    # 4. Wilcoxon p<0.05 vs each underperforming baseline
    underperformers = vs[vs["rank_delta"] > 0]  # baseline ranks worse than HDGPSO
    bad_p = underperformers[underperformers["wilcoxon_p"] >= 0.05]
    claim4 = (len(underperformers) > 0) and (len(bad_p) == 0)
    out(f"\nClaim 4: Wilcoxon p<0.05 for HDGPSO vs each underperforming baseline")
    out(f"  Underperformers (rank_delta>0): {len(underperformers)} of {len(vs)}")
    for _, row in vs.iterrows():
        flag = ""
        if row["rank_delta"] > 0:
            flag = "  [under, p OK]" if row["wilcoxon_p"] < 0.05 else "  [under, p FAIL]"
        else:
            flag = "  [HDGPSO does not beat this baseline]"
        out(f"    {row['baseline']:14s}  rank_delta={row['rank_delta']:+.3f}  p={row['wilcoxon_p']:.4g}{flag}")
    out(f"  -> {'PASS' if claim4 else 'FAIL'}")
    report["claim4_wilcoxon"] = (claim4, vs.to_dict("records"))

    # 5. Cliff's delta > 0.147 vs underperformers
    under = vs[vs["rank_delta"] > 0]
    weak_effect = under[under["cliffs_delta"].abs() <= 0.147]
    claim5 = (len(under) > 0) and (len(weak_effect) == 0)
    out(f"\nClaim 5: |Cliff's delta| > 0.147 vs each underperforming baseline")
    for _, row in vs.iterrows():
        if row["rank_delta"] > 0:
            sz = abs(row["cliffs_delta"])
            band = (
                "large" if sz >= 0.474
                else "medium" if sz >= 0.33
                else "small" if sz >= 0.147
                else "negligible"
            )
            out(f"    {row['baseline']:14s}  delta={row['cliffs_delta']:+.3f}  ({band})")
    out(f"  -> {'PASS' if claim5 else 'FAIL'}")
    report["claim5_cliffs"] = (claim5, vs.to_dict("records"))

    # 6. Budget sweep -- skipped here (separate script)
    out(f"\nClaim 6: HDGPSO's rank stays low across budgets")
    out(f"  -> SKIPPED (budget sweep is a separate run; see run_budget_sweep.py)")
    report["claim6_budget"] = (None, "skipped")

    out("\n" + "=" * 70)
    passed = sum(int(bool(v[0])) for k, v in report.items() if v[0] is not None)
    total = sum(1 for k, v in report.items() if v[0] is not None)
    out(f"OVERALL: {passed}/{total} claims pass (claim 6 deferred)")
    out("=" * 70)

    return report, log.getvalue()


def main():
    summary, history = run()
    report, log_text = check_claims(summary)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "claims_report.txt"), "w") as f:
        f.write(log_text)

    # Save CD diagram + rank-CI for the paper
    import matplotlib
    matplotlib.use("Agg")
    fig = cd_diagram(summary, save_path=os.path.join(OUT_DIR, "fig_cd_diagram.png"),
                     title=f"CD diagram (claim-check, n={len(build_rank_matrix(summary))} cells)")
    ci = bootstrap_rank_ci(summary, n_boot=2000, seed=0)
    ci.to_csv(os.path.join(OUT_DIR, "rank_bootstrap_ci.csv"), index=False)

    print(f"\nWrote: {OUT_DIR}/claims_report.txt, fig_cd_diagram.png, rank_bootstrap_ci.csv")

    # Exit code reflects pass/fail
    failed = [k for k, v in report.items() if v[0] is False]
    if failed:
        print(f"\nFailed claims: {failed}")
        sys.exit(2)


if __name__ == "__main__":
    main()
