"""Print K=7 rank table per budget after removing HDGPSO-MF."""
import pandas as pd
from hdgpso.stats import build_rank_matrix

sweep = pd.read_csv("../results_budget_sweep/summary_budget_sweep.csv")
sweep = sweep[sweep["tuner"] != "HDGPSO-MF"].reset_index(drop=True)

for b in sorted(sweep["budget"].unique()):
    sub = sweep[sweep["budget"] == b]
    rm = build_rank_matrix(sub)
    mr = rm.mean(axis=0).sort_values()
    print(f"=== budget = {b} ===")
    for t, r in mr.items():
        marker = "  <-- HDGPSO" if t == "HDGPSO" else ""
        print(f"  {t:14s} {r:.3f}{marker}")
    print()
