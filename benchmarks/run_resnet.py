"""ResNet-18 / CIFAR-10 benchmark: 8 tuners x 3 seeds x 60 evaluations.

Loops seed-major so an interrupted run still yields complete rank blocks.
Appends to results/resnet_cifar10.csv per cell and records the full trial
history, so the run resumes from wherever it stopped.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
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

# Protocol must be set before the objective module caches the data split.
import resnet_objective as R           # noqa: E402
R.N_TRAIN, R.N_VAL, R.EPOCHS = 20000, 5000, 5

import tuners as T                     # noqa: E402
from smac_tuner import SMACTuner       # noqa: E402

BUDGET = 60
SEEDS = [0, 1, 2]
POP = 5                                # matches run_claim_check_v5.py
DATASET = "cifar10"
MODEL = "ResNet18-CIFAR10"

TUNERS = ["GridSearch", "RandomSearch", "Bayes", "Optuna-TPE",
          "DE", "PSO", "HDGPSO", "SMAC"]

OUT = os.path.join(HERE, "results", "resnet_cifar10.csv")
PROGRESS_EVERY = 5      # heartbeat cadence, in objective evaluations


def with_progress(objective, tuner_name, seed):
    """Wrap the objective so the log gets a heartbeat during long cells.

    Without this the runner only prints when an entire cell finishes -- 60
    evaluations, over an hour -- so a live 29-hour run is indistinguishable
    from a hung one.
    """
    state = {"n": 0, "best": float("inf"), "t0": time.time()}

    def wrapped(params):
        loss = objective(params)
        state["n"] += 1
        state["best"] = min(state["best"], loss)
        if state["n"] % PROGRESS_EVERY == 0:
            el = time.time() - state["t0"]
            print(f"        {tuner_name} s{seed}: eval {state['n']:>2}/{BUDGET}  "
                  f"best_acc={-state['best']:.4f}  "
                  f"{el/60:5.1f} min  ({el/state['n']:.0f} s/eval)", flush=True)
        return loss

    return wrapped


def main():
    T.ALL_TUNERS["SMAC"] = SMACTuner
    space = R.resnet_cifar10_space()

    print(f"ResNet-18 CIFAR variant: {R.n_parameters():,} parameters", flush=True)
    print(f"protocol: n_train={R.N_TRAIN} n_val={R.N_VAL} epochs={R.EPOCHS}", flush=True)
    print(f"grid: {len(TUNERS)} tuners x {len(SEEDS)} seeds x {BUDGET} evals "
          f"= {len(TUNERS)*len(SEEDS)*BUDGET} trainings", flush=True)
    print(f"search space: {space.n_dims} dims {space.names}", flush=True)

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = {(r.tuner, int(r.seed)) for r in prev.itertuples()}
        print(f"resuming: {len(done)} of {len(TUNERS)*len(SEEDS)} cells complete",
              flush=True)

    R._load_cifar10()          # materialise the split once, outside the timing
    print("data resident on GPU\n", flush=True)

    total = len(SEEDS) * len(TUNERS)
    n = 0
    t_start = time.time()

    for seed in SEEDS:                       # seed-major: complete blocks first
        objective = R.make_resnet_cifar10_objective(seed=seed)
        for tuner_name in TUNERS:
            n += 1
            if (tuner_name, seed) in done:
                print(f"[{n}/{total}] {tuner_name} seed={seed} -- cached", flush=True)
                continue

            el = time.time() - t_start
            eta = (el / max(n - 1, 1)) * (total - n + 1) / 3600 if n > 1 else float("nan")
            print(f"[{n}/{total}] {tuner_name} seed={seed}  "
                  f"({el/3600:.2f} h elapsed, ETA {eta:.1f} h)", flush=True)

            t0 = time.time()
            try:
                tuner = T.ALL_TUNERS[tuner_name](
                    space=space,
                    objective=with_progress(objective, tuner_name, seed),
                    budget=BUDGET, seed=seed, population_size=POP,
                )
                res = tuner.optimize()
                row = {
                    "dataset": DATASET, "model": MODEL, "tuner": tuner_name,
                    "seed": seed, "best_loss": res.best_loss,
                    "n_evals": res.n_evals, "elapsed": res.elapsed_seconds,
                    "stopped": getattr(res, "stopped_reason", ""), "error": "",
                }
                # The winning configuration, one column per hyperparameter, so the
                # paper can report which values each tuner actually converged to.
                for k in space.names:
                    row[f"best_{k}"] = res.best_params.get(k)
                # Full trial trace: every configuration tried and the accuracy it
                # produced. Verified that all eight tuners populate this. Enables
                # convergence curves and analysis of which regions each explored.
                if res.history is not None and not res.history.empty:
                    h = res.history.copy()
                    h["tuner"], h["seed"] = tuner_name, seed
                    hp = os.path.join(HERE, "results", "resnet_history.csv")
                    h.to_csv(hp, mode="a", header=not os.path.exists(hp), index=False)
            except Exception as exc:                      # noqa: BLE001
                traceback.print_exc()
                row = {
                    "dataset": DATASET, "model": MODEL, "tuner": tuner_name,
                    "seed": seed, "best_loss": float("inf"), "n_evals": 0,
                    "elapsed": time.time() - t0, "stopped": "error",
                    "error": repr(exc),
                }

            pd.DataFrame([row]).to_csv(
                OUT, mode="a", header=not os.path.exists(OUT), index=False)
            acc = -row["best_loss"] if row["best_loss"] < 0 else float("nan")
            print(f"      best_loss={row['best_loss']:.6g} (val_acc={acc:.4f}) "
                  f"n_evals={row['n_evals']} {row['elapsed']/60:.1f} min", flush=True)

        print(f"--- seed {seed} complete: block is rankable ---\n", flush=True)

    print(f"ResNet benchmark complete in {(time.time()-t_start)/3600:.2f} h -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
