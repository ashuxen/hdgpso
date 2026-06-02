"""hdgpso — Hybrid DE-GWO-PSO hyperparameter optimization.

:class:`HDGPSO` is the optimizer described in the paper. It runs three
operator stages each iteration — Differential Evolution, Grey Wolf
Optimizer, and Particle Swarm Optimization — and uses a RandomForest
surrogate to filter candidates between stages via a tree-variance
lower-confidence-bound score.

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

The rank-based statistical helpers used in the paper benchmarks
(Friedman, Nemenyi, CD diagrams, Cliff's delta, bootstrap CIs) live in
:mod:`hdgpso.stats`.

.. note::

   :class:`HDGPSOMF` is an experimental multi-fidelity variant built on
   top of HDGPSO with BOHB-style successive halving. It is provided for
   future-work exploration and is not part of the published results.
   Its API is not yet stable.
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
