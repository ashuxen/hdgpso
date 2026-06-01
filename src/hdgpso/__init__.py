"""hdgpso — Hybrid DE-GWO-PSO hyperparameter optimization.

:class:`HDGPSO` is the published optimizer: a three-stage hybrid
metaheuristic (Differential Evolution + Grey Wolf Optimizer + Particle
Swarm Optimization) augmented by a RandomForest surrogate filter using
tree-variance-based lower-confidence-bound acquisition.

Quick start::

    from hdgpso import HDGPSO, SearchSpace, Float, Int, Categorical

    space = SearchSpace({
        "lr":     Float(1e-5, 1e-2, log=True),
        "layers": Int(2, 8),
        "act":    Categorical(["relu", "gelu", "tanh"]),
    })

    def objective(params):
        # ... train model with given hyperparameters
        return -val_accuracy   # lower is better

    result = HDGPSO(space, objective, population_size=10,
                    iterations=15, seed=0).optimize()
    print(result.best_params, result.best_loss)

Statistical-analysis helpers for paper-grade benchmarks live in
:mod:`hdgpso.stats` (Friedman, Nemenyi, CD diagrams, Cliff's delta,
bootstrap CIs).

.. note::

   :class:`HDGPSOMF` is an *experimental* multi-fidelity variant built
   on top of HDGPSO with BOHB-style successive halving. It is provided
   for future-work exploration only and is not part of the published
   headline results; its API may change before a 1.0 release.
"""
from .core import (
    Categorical,
    Dimension,
    Float,
    HDGPSO,
    Int,
    OptimizeResult,
    SearchSpace,
)
from .multifidelity import HDGPSOMF
from ._version import __version__

__all__ = [
    # Search space
    "Categorical",
    "Dimension",
    "Float",
    "Int",
    "SearchSpace",
    # Optimizers
    "HDGPSO",
    "HDGPSOMF",
    "OptimizeResult",
    # Version
    "__version__",
]
