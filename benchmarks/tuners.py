"""Uniform `.optimize()` adapters around third-party hyperparameter tuners.

Every tuner here exposes:

    tuner = SomeTuner(space, objective, budget=N, seed=42)
    result = tuner.optimize()   # -> hdgpso.OptimizeResult

`budget` is the *function-evaluation budget* (number of times `objective`
is called), so all methods can be compared fairly on equal wall-clock-
independent budget.

Optional dependencies are imported lazily inside each tuner so the rest
of the benchmark still works if a package is missing.
"""
from __future__ import annotations

import time
import warnings
from itertools import product
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from hdgpso import (
    Categorical,
    Float,
    HDGPSO,
    Int,
    OptimizeResult,
    SearchSpace,
)


# ---------------------------------------------------------------------------
# Helpers shared by adapters
# ---------------------------------------------------------------------------


def _wrap_objective(
    space: SearchSpace,
    objective: Callable[[Dict[str, Any]], float],
    budget: int,
    name: str,
):
    """Return (history-recording wrapper, history-list, stop-flag dict).

    The wrapper raises StopIteration once `budget` evaluations occur so
    adapters that don't natively support stopping mid-run still honor it.
    """
    history: List[Dict[str, Any]] = []
    state = {"stop": False, "t0": time.time()}

    def wrapped(params: Dict[str, Any]) -> float:
        if state["stop"]:
            return float("inf")
        try:
            loss = float(objective(params))
        except Exception as exc:  # noqa: BLE001
            loss = float("inf")
        if not np.isfinite(loss):
            loss = float("inf")
        history.append(
            {
                "iteration": len(history) + 1,
                "loss": loss,
                "elapsed": time.time() - state["t0"],
                "optimizer": name,
                **params,
            }
        )
        if len(history) >= budget:
            state["stop"] = True
        return loss

    return wrapped, history, state


def _finalize(history: List[Dict[str, Any]], t0: float, stopped: str) -> OptimizeResult:
    df = pd.DataFrame(history)
    if df.empty:
        return OptimizeResult({}, float("inf"), df, 0, time.time() - t0, stopped)
    df["running_best"] = df["loss"].cummin()
    best_idx = df["loss"].idxmin()
    best_row = df.loc[best_idx]
    param_cols = [c for c in df.columns if c not in {"iteration", "loss", "elapsed", "optimizer", "running_best"}]
    best_params = {c: best_row[c] for c in param_cols}
    return OptimizeResult(
        best_params=best_params,
        best_loss=float(best_row["loss"]),
        history=df,
        n_evals=len(df),
        elapsed_seconds=time.time() - t0,
        stopped_reason=stopped,
    )


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------


class RandomSearchTuner:
    name = "RandomSearch"

    def __init__(self, space, objective, budget=50, seed=None, **_):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> OptimizeResult:
        wrapped, history, state = _wrap_objective(
            self.space, self.objective, self.budget, self.name
        )
        for _ in range(self.budget):
            if state["stop"]:
                break
            x = self.space.sample(self.rng, 1)[0]
            wrapped(self.space.decode(x))
        return _finalize(history, state["t0"], "budget")


# ---------------------------------------------------------------------------
# Grid search (coarse, only for low-dim spaces)
# ---------------------------------------------------------------------------


class GridSearchTuner:
    """Coarse grid that picks at most `levels` points per dimension.

    Falls back to skipping dims if total cells would exceed `budget`.
    """

    name = "GridSearch"

    def __init__(self, space, objective, budget=50, seed=None, levels=4, **_):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.levels = int(levels)

    def _grid_values(self, dim, k):
        if isinstance(dim, Float):
            if dim.log:
                return list(np.exp(np.linspace(np.log(dim.low), np.log(dim.high), k)))
            return list(np.linspace(dim.low, dim.high, k))
        if isinstance(dim, Int):
            vals = np.unique(np.round(np.linspace(dim.low, dim.high, k)).astype(int))
            return list(vals)
        if isinstance(dim, Categorical):
            return list(dim.choices)
        raise TypeError(f"Unknown dim type {type(dim)}")

    def optimize(self) -> OptimizeResult:
        # shrink levels until total combos <= budget
        k = self.levels
        while k > 1:
            total = 1
            for d in self.space.dims.values():
                total *= len(self._grid_values(d, k))
            if total <= self.budget:
                break
            k -= 1
        grids = {n: self._grid_values(d, k) for n, d in self.space.dims.items()}
        combos = list(product(*grids.values()))
        wrapped, history, state = _wrap_objective(
            self.space, self.objective, self.budget, self.name
        )
        for combo in combos:
            if state["stop"]:
                break
            wrapped(dict(zip(self.space.names, combo)))
        return _finalize(history, state["t0"], "budget")


# ---------------------------------------------------------------------------
# Bayesian (skopt)
# ---------------------------------------------------------------------------


class BayesTuner:
    name = "Bayes"

    def __init__(self, space, objective, budget=50, seed=None, **_):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.seed = seed

    def _to_skopt(self):
        from skopt.space import Categorical as SkCat
        from skopt.space import Integer as SkInt
        from skopt.space import Real as SkReal

        dims = []
        for name, d in self.space.dims.items():
            if isinstance(d, Float):
                dims.append(
                    SkReal(d.low, d.high, prior="log-uniform" if d.log else "uniform", name=name)
                )
            elif isinstance(d, Int):
                dims.append(
                    SkInt(d.low, d.high, prior="log-uniform" if d.log else "uniform", name=name)
                )
            elif isinstance(d, Categorical):
                dims.append(SkCat(d.choices, name=name))
            else:
                raise TypeError(f"Unknown dim type {type(d)}")
        return dims

    def optimize(self) -> OptimizeResult:
        try:
            from skopt import gp_minimize
            from skopt.utils import use_named_args
        except ImportError as exc:
            raise RuntimeError("scikit-optimize required: pip install scikit-optimize") from exc

        dims = self._to_skopt()
        wrapped, history, state = _wrap_objective(
            self.space, self.objective, self.budget, self.name
        )

        @use_named_args(dims)
        def fn(**params):
            if state["stop"]:
                return 1e18  # sentinel
            return wrapped(params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gp_minimize(
                    fn,
                    dims,
                    n_calls=self.budget,
                    n_initial_points=min(5, self.budget),
                    random_state=self.seed,
                )
            except StopIteration:
                pass
        return _finalize(history, state["t0"], "budget")


# ---------------------------------------------------------------------------
# Optuna (TPE)
# ---------------------------------------------------------------------------


class OptunaTuner:
    name = "Optuna-TPE"

    def __init__(self, space, objective, budget=50, seed=None, **_):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.seed = seed

    def optimize(self) -> OptimizeResult:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("optuna required: pip install optuna") from exc

        wrapped, history, state = _wrap_objective(
            self.space, self.objective, self.budget, self.name
        )

        def trial_fn(trial):
            if state["stop"]:
                raise optuna.exceptions.TrialPruned()
            params = {}
            for name, d in self.space.dims.items():
                if isinstance(d, Float):
                    params[name] = trial.suggest_float(name, d.low, d.high, log=d.log)
                elif isinstance(d, Int):
                    params[name] = trial.suggest_int(name, d.low, d.high, log=d.log)
                elif isinstance(d, Categorical):
                    params[name] = trial.suggest_categorical(name, d.choices)
            return wrapped(params)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        try:
            study.optimize(trial_fn, n_trials=self.budget, show_progress_bar=False)
        except StopIteration:
            pass
        return _finalize(history, state["t0"], "budget")


# ---------------------------------------------------------------------------
# Plain DE (scipy)
# ---------------------------------------------------------------------------


class DETuner:
    name = "DE"

    def __init__(
        self,
        space,
        objective,
        budget=50,
        seed=None,
        population_size: int = 10,
        **_,
    ):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.seed = seed
        self.population_size = int(population_size)

    def optimize(self) -> OptimizeResult:
        from scipy.optimize import differential_evolution

        wrapped, history, state = _wrap_objective(
            self.space, self.objective, self.budget, self.name
        )

        bounds = list(zip(self.space.lows, self.space.highs))

        def fn(x):
            if state["stop"]:
                return 1e18
            return wrapped(self.space.decode(x))

        # maxiter chosen so total evals ~ budget
        maxiter = max(1, self.budget // max(self.population_size, 2) - 1)
        try:
            differential_evolution(
                fn,
                bounds,
                maxiter=maxiter,
                popsize=max(2, self.population_size // self.space.n_dims),
                seed=self.seed,
                polish=False,
                tol=0,
                init="sobol",
            )
        except StopIteration:
            pass
        return _finalize(history, state["t0"], "budget")


# ---------------------------------------------------------------------------
# Plain PSO (pyswarms)
# ---------------------------------------------------------------------------


class PSOTuner:
    name = "PSO"

    def __init__(
        self,
        space,
        objective,
        budget=50,
        seed=None,
        population_size: int = 10,
        **_,
    ):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.seed = seed
        self.population_size = int(population_size)

    def optimize(self) -> OptimizeResult:
        # pyswarms imports a logging YAML at module-load time that can fail
        # on Windows ("Unable to configure handler 'file_default'"). Wrap
        # logging.config.dictConfig with a swallow-on-failure shim before
        # the first pyswarms import.
        import logging.config

        _orig = logging.config.dictConfig

        def _safe(cfg):
            try:
                _orig(cfg)
            except Exception:
                pass

        logging.config.dictConfig = _safe
        try:
            import pyswarms as ps
        except ImportError as exc:
            logging.config.dictConfig = _orig
            raise RuntimeError("pyswarms required: pip install pyswarms") from exc
        finally:
            logging.config.dictConfig = _orig

        wrapped, history, state = _wrap_objective(
            self.space, self.objective, self.budget, self.name
        )

        lb = self.space.lows
        ub = self.space.highs

        def fn(pop: np.ndarray) -> np.ndarray:
            out = np.empty(pop.shape[0])
            for i, x in enumerate(pop):
                if state["stop"]:
                    out[i] = 1e18
                else:
                    out[i] = wrapped(self.space.decode(x))
            return out

        if self.seed is not None:
            np.random.seed(self.seed)
        iters = max(1, self.budget // max(self.population_size, 2))
        opt = ps.single.GlobalBestPSO(
            n_particles=self.population_size,
            dimensions=self.space.n_dims,
            options={"c1": 1.5, "c2": 1.5, "w": 0.5},
            bounds=(lb, ub),
        )
        try:
            opt.optimize(fn, iters=iters, verbose=False)
        except StopIteration:
            pass
        return _finalize(history, state["t0"], "budget")


# ---------------------------------------------------------------------------
# HDGPSO wrappers (just normalize budget to iterations)
# ---------------------------------------------------------------------------


def _budget_to_iterations(budget: int, population_size: int) -> int:
    # Each iter evaluates 3*pop_size objective calls (DE trial + GWO + PSO).
    # Initial population evaluation is one more pop_size. So:
    # total = pop_size + iterations * 3 * pop_size  =>  iterations = (budget - pop)/(3*pop)
    # We use ceil + an extra cushion; the actual stop is enforced by eval_budget.
    iters = max(1, int(np.ceil((budget - population_size) / (3 * population_size))) + 1)
    return iters


class HDGPSOTuner:
    name = "HDGPSO"

    def __init__(
        self,
        space,
        objective,
        budget=50,
        seed=None,
        population_size: int = 10,
        early_stop_patience: Optional[int] = None,
        **_,
    ):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.seed = seed
        self.population_size = int(population_size)
        self.early_stop_patience = early_stop_patience

    def optimize(self) -> OptimizeResult:
        iters = _budget_to_iterations(self.budget, self.population_size)
        opt = HDGPSO(
            self.space,
            self.objective,
            population_size=self.population_size,
            iterations=iters,
            seed=self.seed,
            early_stop_patience=self.early_stop_patience,
            eval_budget=self.budget,
        )
        result = opt.optimize()
        # Trim history to budget for fair comparison
        if len(result.history) > self.budget:
            result.history = result.history.iloc[: self.budget].copy()
            result.history["running_best"] = result.history["loss"].cummin()
            best_idx = result.history["loss"].idxmin()
            best_row = result.history.loc[best_idx]
            param_cols = [
                c
                for c in result.history.columns
                if c not in {"iteration", "loss", "elapsed", "optimizer", "running_best"}
            ]
            result.best_params = {c: best_row[c] for c in param_cols}
            result.best_loss = float(best_row["loss"])
            result.n_evals = self.budget
        return result


class HDGPSOMFTuner:
    """HDGPSO-MF (multi-fidelity). Expects a fidelity-capable objective.

    The benchmark driver supplies a fidelity-capable objective when
    tuner_name == 'HDGPSO-MF' via make_fidelity_*_objective.
    """

    name = "HDGPSO-MF"

    def __init__(self, space, objective, budget=50, seed=None,
                 population_size: int = 5, **_):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.seed = seed
        self.population_size = int(population_size)

    def optimize(self) -> OptimizeResult:
        from hdgpso import HDGPSOMF
        # Iteration cap is generous; the fidelity budget cap stops the run cleanly.
        iters = max(8, int(np.ceil(self.budget / max(self.population_size, 1))))
        opt = HDGPSOMF(
            self.space, self.objective,
            population_size=self.population_size,
            iterations=iters,
            eval_budget=self.budget,
            seed=self.seed,
        )
        return opt.optimize()


ALL_TUNERS = {
    "RandomSearch": RandomSearchTuner,
    "GridSearch": GridSearchTuner,
    "Bayes": BayesTuner,
    "Optuna-TPE": OptunaTuner,
    "DE": DETuner,
    "PSO": PSOTuner,
    "HDGPSO": HDGPSOTuner,
    "HDGPSO-MF": HDGPSOMFTuner,
}
