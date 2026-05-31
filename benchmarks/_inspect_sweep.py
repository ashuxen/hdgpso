"""Per-budget rank breakdown with both HDGPSO and HDGPSO-MF."""
import pandas as pd
from hdgpso.stats import build_rank_matrix

sweep = pd.read_csv("../results_budget_sweep/summary_budget_sweep.csv")
print(f"Sweep rows: {len(sweep)}, tuners: {sorted(sweep['tuner'].unique())}\n")

for b in sorted(sweep['budget'].unique()):
    sub = sweep[sweep['budget'] == b]
    rm = build_rank_matrix(sub)
    mr = rm.mean(axis=0).sort_values()
    print(f"=== Budget = {b}  (n = {len(rm)} cells) ===")
    for t, r in mr.items():
        marker = ""
        if t == "HDGPSO":      marker = "  <-- HDGPSO"
        if t == "HDGPSO-MF":   marker = "  <-- HDGPSO-MF"
        print(f"  {t:14s}  {r:.3f}{marker}")
    print()
