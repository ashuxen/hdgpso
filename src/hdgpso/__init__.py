"""hdgpso — Hybrid DE-GWO-PSO hyperparameter optimization.

Two algorithms are available:

  * :class:`HDGPSO` — single-fidelity three-stage hybrid (DE + GWO + PSO)
    with optional RandomForest surrogate. Best for medium evaluation
    budgets (40-60 evals) and tree-based model HPO.

  * :class:`HDGPSOMF` — multi-fidelity extension with BOHB-style
    successive halving. Best for low evaluation budgets where each
    objective evaluation is expensive.

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

For multi-fidelity workflows the objective accepts a ``fidelity``
argument; see :class:`HDGPSOMF` and the ``hdgpso.multifidelity`` module.

Statistical-analysis helpers for paper-grade benchmarks live in
:mod:`hdgpso.stats` (Friedman, Nemenyi, CD diagrams, Cliff's delta,
bootstrap CIs).
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
