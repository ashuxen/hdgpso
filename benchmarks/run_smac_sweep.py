"""Run SMAC at budgets 20, 40 and 100 so Fig. 4 can be redrawn at K=8.

Budget 60 is reused from run_smac.py rather than re-run. Results are merged
into results/smac_sweep_all_budgets.csv.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# In the repo this file sits in benchmarks/ next to benchmark.py, with hdgpso
# pip-installed, so no path setup is needed. The snapshot directory only exists
# in the standalone revision folder.
_SNAP = os.path.join(HERE, "src_snapshot")
if os.path.isdir(_SNAP):
    sys.path.insert(0, _SNAP)
sys.path.insert(0, HERE)

warnings.filterwarnings("ignore")

import benchmark as B              # noqa: E402
import tuners as T                 # noqa: E402
from smac_tuner import SMACTuner   # noqa: E402

BUDGETS = [20, 40, 100]            # 60 is reused from run_smac.py
SEEDS = [0, 1, 2]
POP = 5                            # matches run_budget_sweep.py
CV = 3
DATASETS = ["breast_cancer", "wine", "diabetes", "pinn_heat"]
MODELS = ["RandomForest", "GradientBoosting", "MLP", "PINN-Heat"]

OUT = os.path.join(HERE, "results", "smac_sweep.csv")
B60 = os.path.join(HERE, "results", "smac_b60.csv")


def main():
    T.ALL_TUNERS["SMAC"] = SMACTuner
    B.ALL_TUNERS["SMAC"] = SMACTuner

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.dataset, r.model, int(r.seed), int(r.budget))
                for r in prev.itertuples()}
        print(f"resuming: {len(done)} cells already complete", flush=True)

    all_ds = B.get_datasets(include_openml=False, include_deep=True)
    all_ds = {k: v for k, v in all_ds.items() if k in DATASETS}

    pairs = []
    for ds_name, ds in all_ds.items():
        for model_name in MODELS:
            task = ds["task"]
            if model_name == "PINN-Heat" and ds_name != "pinn_heat":
                continue
            if ds_name == "pinn_heat" and model_name != "PINN-Heat":
                continue
            if model_name == "MLP" and task != "classification":
                continue
            if model_name in ("RandomForest", "GradientBoosting") and task == "pinn":
                continue
            pairs.append((ds_name, ds, model_name))

    total = len(pairs) * len(SEEDS) * len(BUDGETS)
    n = 0
    t_start = time.time()

    for budget in BUDGETS:
        print(f"\n{'='*62}\nBUDGET = {budget}\n{'='*62}", flush=True)
        for ds_name, ds, model_name in pairs:
            for seed in SEEDS:
                n += 1
                if (ds_name, model_name, seed, budget) in done:
                    print(f"[{n}/{total}] b={budget} {ds_name}/{model_name} "
                          f"seed={seed} -- cached", flush=True)
                    continue
                el = time.time() - t_start
                print(f"[{n}/{total}] b={budget} {ds_name}/{model_name} seed={seed} "
                      f"({el/60:.1f}m elapsed)", flush=True)
                summary, _ = B.run_one(
                    dataset_name=ds_name, X=ds["X"], y=ds["y"],
                    task_type=ds["task"], model_name=model_name,
                    tuner_name="SMAC", seed=seed, budget=budget,
                    cv=CV, population_size=POP,
                )
                summary["budget"] = budget
                pd.DataFrame([summary]).to_csv(
                    OUT, mode="a", header=not os.path.exists(OUT), index=False)
                print(f"      best_loss={summary['best_loss']:.6g} "
                      f"n_evals={summary['n_evals']} "
                      f"{summary['elapsed']:.1f}s", flush=True)

    # Fold in the already-computed b=60 rows so the output is a complete sweep.
    if os.path.exists(B60):
        b60 = pd.read_csv(B60)
        b60["budget"] = 60
        full = pd.concat([pd.read_csv(OUT), b60], ignore_index=True)
        merged = os.path.join(HERE, "results", "smac_sweep_all_budgets.csv")
        full.to_csv(merged, index=False)
        print(f"\nmerged with b=60 -> {merged} "
              f"({len(full)} rows, budgets {sorted(full.budget.unique())})",
              flush=True)

    print(f"\nSMAC sweep complete in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
