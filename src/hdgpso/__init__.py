"""hdgpso package. Hybrid DE-GWO-PSO hyperparameter optimization.

The :class:`HDGPSO` class is the optimizer described in the paper. It
runs three operator stages on the same population in every iteration:
Differential Evolution, Grey Wolf Optimization, and Particle Swarm
Optimization. A RandomForest surrogate is fit on the trial history
and is used to filter proposed candidates between the stages, using a
tree-variance lower-confidence-bound score.

Quick start::

    from hdgpso import HDGPSO, SearchSpace, Float, Int, Categorical

    space = SearchSpace({
        "lr":     Float(1e-5, 1e-2, log=True),
        "layers": Int(2, 8),
        "act":    Categorical(["relu", "gelu", "tanh"]),
    })

    def objective(params):
        # train a model with the given hyperparameters
        return -val_accuracy   # lower is better

    result = HDGPSO(space, objective, population_size=10,
                    iterations=15, seed=0).optimize()
    print(result.best_params, result.best_loss)

The rank-based statistical helpers used for the paper benchmark
(Friedman, Nemenyi, CD diagrams, Cliff's delta, bootstrap confidence
intervals) are available in :mod:`hdgpso.stats`.

.. note::

   :class:`HDGPSOMF` is an experimental multi-fidelity variant built on
   top of HDGPSO with BOHB-style successive halving. It is included for
   future-work exploration only. It is not part of the published
   results, and its API is not yet stable.
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
