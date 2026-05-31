"""HDGPSO-MF example: multi-fidelity HPO on a synthetic noisy objective.

Demonstrates how to write a fidelity-aware objective and how
HDGPSO-MF uses low-fidelity probes to filter candidates before
spending the full evaluation budget.

Run:
    pip install hdgpso
    python 02_multifidelity.py
"""
import math
import time

from hdgpso import HDGPSOMF, SearchSpace, Float, Int


# Synthetic expensive objective: a smooth 4D function whose evaluation
# cost (here, time.sleep) scales with fidelity. Lower fidelity returns
# a noisier estimate of the same objective.
def objective(params, fidelity: float = 1.0):
    # Pretend each full evaluation takes 0.5s; lower fidelity is cheaper.
    time.sleep(0.5 * fidelity)
    x, y = params["x"], params["y"]
    a, b = params["a"], params["b"]
    # Smooth ground-truth objective
    truth = (1 - x) ** 2 + 100 * (y - x * x) ** 2 + 0.1 * (a * a + b * b)
    # Low-fidelity returns a noisy version of the truth
    import random
    random.seed(int((x * 1000 + y * 100 + a + b) * 1e6) % (2**31))
    noise = random.gauss(0, 1.0 - fidelity)  # noise vanishes at fidelity=1
    return truth + noise


space = SearchSpace({
    "x": Float(-2, 2),
    "y": Float(-2, 2),
    "a": Int(-5, 5),
    "b": Int(-5, 5),
})

print("Optimizing noisy synthetic objective with HDGPSO-MF...")
print("  - Each full-fidelity eval costs 1.0 budget unit")
print("  - Low-fidelity (0.3) probes cost 0.3 units each")
print("  - Budget = 30 fidelity-units total\n")

t0 = time.time()
result = HDGPSOMF(
    space=space,
    objective=objective,
    eval_budget=30,
    low_fidelity=0.3,
    verify_fraction=0.4,
    population_size=5,
    seed=0,
    verbose=True,
).optimize()

print()
print(f"Best parameters: {result.best_params}")
print(f"Best loss (full fidelity): {result.best_loss:.4f}")
print(f"Total evaluations: {result.n_evals}")
print(f"  - Low-fidelity probes: {(result.history['fidelity'] < 0.99).sum()}")
print(f"  - Full-fidelity evals: {(result.history['fidelity'] >= 0.99).sum()}")
print(f"Wall-clock: {time.time() - t0:.1f}s")
