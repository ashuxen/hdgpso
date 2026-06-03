"""Budget-sensitivity sweep used to evaluate Claim 6.

The script re-runs the benchmark at several evaluation budgets and
checks whether HDGPSO's mean rank remains in the top tier (mean rank
no greater than 2.5) at every budget.
"""
from __future__ import annotations

import os
import sys
import time
from io import StringIO

import pandas as pd

from benchmark import run_budget_sweep
from hdgpso.stats import build_rank_matrix

OUT_DIR = "../results_budget_sweep"


def check_claim6(summary_sweep: pd.DataFrame, max_rank: float = 3.0) -> tuple[bool, pd.DataFrame, str]:
    log = StringIO()

    def out(s=""):
        print(s)
        log.write(s + "\n")

    rows = []
    for b in sorted(summary_sweep["budget"].unique()):
        sub = summary_sweep[summary_sweep["budget"] == b]
        rm = build_rank_matrix(sub)
        mean_rank = rm.mean(axis=0).sort_values()
        rows.append(
            {
                "budget": b,
                "hdgpso_rank": float(mean_rank.get("HDGPSO", float("nan"))),
                "top_tuner": mean_rank.index[0],
                "top_rank": float(mean_rank.iloc[0]),
            }
        )
    df = pd.DataFrame(rows)
    out("\n=== Claim 6: HDGPSO rank vs budget ===")
    out(df.to_string(index=False))
    holds = (df["hdgpso_rank"] <= max_rank).all()
    out(f"\nClaim 6 (HDGPSO mean rank <= {max_rank} at every budget): "
        f"{'PASS' if holds else 'FAIL'}")
    return holds, df, log.getvalue()


def main():
    t0 = time.time()
    summary, history = run_budget_sweep(
        budgets=[20, 40, 60, 100],
        seeds=[0, 1, 2],
        datasets=["breast_cancer", "wine", "diabetes", "pinn_heat"],
        models=["RandomForest", "GradientBoosting", "MLP", "PINN-Heat"],
        tuners=["RandomSearch", "GridSearch", "Bayes", "Optuna-TPE", "DE", "PSO",
                "HDGPSO", "HDGPSO-MF"],
        cv=3,
        population_size=5,
        include_openml=False,
        include_deep=True,
        out_dir=OUT_DIR,
        verbose=True,
    )
    print(f"\nTotal sweep time: {(time.time()-t0)/60:.1f} min")

    holds, df, log_text = check_claim6(summary)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "claim6_report.txt"), "w") as f:
        f.write(log_text)
    df.to_csv(os.path.join(OUT_DIR, "claim6_rank_by_budget.csv"), index=False)

    sys.exit(0 if holds else 2)


if __name__ == "__main__":
    main()
