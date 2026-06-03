"""HDGPSO: Hybrid DE-GWO-PSO hyperparameter optimizer.

This module implements a population-based hyperparameter tuner that
runs three search operators on the same population in every iteration.
Each operator has a different behavior, and together they cover the
explore/exploit spectrum:

  Stage 1 (Differential Evolution). For each member i in the population,
    a DE/rand/1 mutant is formed, binary crossover is applied with rate
    CR, and the resulting trial replaces the current member only if it
    improves the loss. The parameter F controls the mutation strength.

  Stage 2 (Grey Wolf Optimizer). The three best members are labeled
    alpha, beta, and delta. Every other member is updated using the
    standard GWO position update from Mirjalili (2014):

        a       linearly decreases from 2 -> 0 across iterations
        A_k     = 2*a*r1 - a    (per dim, per leader k in {alpha, beta, delta})
        C_k     = 2*r2          (per dim, per leader k)
        D_k     = |C_k * X_k - X_i|
        X_k'    = X_k - A_k * D_k
        X_new   = (X_alpha' + X_beta' + X_delta') / 3

  Stage 3 (Particle Swarm Optimization). Each particle stores its own
    best position (pbest) and the swarm stores the all-time global best
    (gbest). The velocity and position updates are:

        w   linearly decreases from w_max -> w_min across iterations
        v_i = w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
        x_i = x_i + v_i

DE is used for exploration, GWO pulls candidates toward the current top
three, and PSO contributes memory and momentum for refinement. Each
iteration costs approximately 3 * population_size objective evaluations.

Optuna-style API:

    space = {
        "lr":      Float(1e-5, 1e-2, log=True),
        "dropout": Float(0.0, 0.5),
        "layers":  Int(2, 8),
        "kernel":  Categorical(["rbf", "linear", "poly"]),
    }

    def objective(params: dict) -> float:
        model = build(**params)
        return -cross_val_score(model, X, y).mean()  # lower is better

    opt = HDGPSO(space, objective, population_size=10, iterations=30)
    result = opt.optimize()
    print(result.best_params, result.best_loss)
    history_df = result.history
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Search-space types
# ---------------------------------------------------------------------------


class Dimension:
    """Base class for a single hyperparameter dimension."""

    name: str = ""  # set when bound into a SearchSpace

    def to_internal(self, value: Any) -> float:
        raise NotImplementedError

    def from_internal(self, x: float) -> Any:
        raise NotImplementedError

    @property
    def internal_bounds(self) -> Tuple[float, float]:
        raise NotImplementedError


class Float(Dimension):
    def __init__(self, low: float, high: float, log: bool = False):
        if log and low <= 0:
            raise ValueError("log=True requires low > 0")
        if high <= low:
            raise ValueError(f"high ({high}) must be > low ({low})")
        self.low, self.high, self.log = float(low), float(high), bool(log)

    @property
    def internal_bounds(self):
        if self.log:
            return math.log(self.low), math.log(self.high)
        return self.low, self.high

    def from_internal(self, x: float) -> float:
        lo, hi = self.internal_bounds
        x = float(np.clip(x, lo, hi))
        return math.exp(x) if self.log else x

    def to_internal(self, value: float) -> float:
        return math.log(value) if self.log else float(value)


class Int(Dimension):
    def __init__(self, low: int, high: int, log: bool = False):
        if log and low <= 0:
            raise ValueError("log=True requires low > 0")
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
        self.low, self.high, self.log = int(low), int(high), bool(log)

    @property
    def internal_bounds(self):
        if self.log:
            return math.log(self.low), math.log(self.high + 1)
        return self.low - 0.4999, self.high + 0.4999

    def from_internal(self, x: float) -> int:
        lo, hi = self.internal_bounds
        x = float(np.clip(x, lo, hi))
        v = math.exp(x) if self.log else x
        return int(np.clip(round(v), self.low, self.high))

    def to_internal(self, value: int) -> float:
        return math.log(value) if self.log else float(value)


class Categorical(Dimension):
    def __init__(self, choices: Sequence[Any]):
        if len(choices) < 1:
            raise ValueError("Categorical needs at least one choice")
        self.choices = list(choices)

    @property
    def internal_bounds(self):
        return -0.4999, len(self.choices) - 1 + 0.4999

    def from_internal(self, x: float) -> Any:
        lo, hi = self.internal_bounds
        x = float(np.clip(x, lo, hi))
        idx = int(np.clip(round(x), 0, len(self.choices) - 1))
        return self.choices[idx]

    def to_internal(self, value: Any) -> float:
        return float(self.choices.index(value))


class SearchSpace:
    """Ordered collection of named Dimension objects."""

    def __init__(self, dims: Dict[str, Dimension]):
        if not dims:
            raise ValueError("SearchSpace must contain at least one dimension")
        self.dims = dict(dims)
        for name, d in self.dims.items():
            d.name = name
        self.names = list(self.dims.keys())
        self.lows = np.array([self.dims[n].internal_bounds[0] for n in self.names])
        self.highs = np.array([self.dims[n].internal_bounds[1] for n in self.names])

    @property
    def n_dims(self) -> int:
        return len(self.names)

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        return {n: self.dims[n].from_internal(x[i]) for i, n in enumerate(self.names)}

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.uniform(self.lows, self.highs, size=(n, self.n_dims))

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lows, self.highs)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OptimizeResult:
    best_params: Dict[str, Any]
    best_loss: float
    history: pd.DataFrame
    n_evals: int
    elapsed_seconds: float
    stopped_reason: str = "iterations"

    def __repr__(self) -> str:
        return (
            f"OptimizeResult(best_loss={self.best_loss:.6g}, "
            f"n_evals={self.n_evals}, elapsed={self.elapsed_seconds:.1f}s, "
            f"stopped={self.stopped_reason})"
        )


# ---------------------------------------------------------------------------
# HDGPSO
# ---------------------------------------------------------------------------


class HDGPSO:
    """Hybrid DE-GWO-PSO hyperparameter optimizer.

    Parameters
    ----------
    space : SearchSpace or dict[str, Dimension]
        The hyperparameter search space.
    objective : callable(dict) -> float
        The objective function to minimize. Lower values are better.
    population_size : int
        Number of candidates maintained per iteration.
    iterations : int
        Number of optimization iterations. Each iteration runs all
        three stages, which corresponds to approximately
        3 * population_size objective evaluations per iteration.
    F : float
        Differential evolution mutation weight.
    CR : float
        Differential evolution crossover probability.
    c1, c2 : float
        Cognitive (pbest) and social (gbest) acceleration coefficients
        used by the PSO velocity update.
    w_max, w_min : float
        Starting and ending values for the PSO inertia weight. The
        value is linearly interpolated between them across iterations,
        so the swarm explores at the start and refines at the end.
    early_stop_patience : int, optional
        Stop after this many consecutive iterations without an
        improvement in best loss. Disabled if None.
    time_budget_seconds : float, optional
        Stop when wall-clock time exceeds this number of seconds.
        Disabled if None.
    seed : int, optional
        Random seed used to initialize the internal generators.
    verbose : bool
        If True, log per-iteration progress messages.
    """

    name = "HDGPSO"

    def __init__(
        self,
        space,
        objective: Callable[[Dict[str, Any]], float],
        population_size: int = 10,
        iterations: int = 20,
        F: float = 0.8,
        CR: float = 0.5,
        c1: float = 2.0,
        c2: float = 2.0,
        w_max: float = 0.7,
        w_min: float = 0.4,
        early_stop_patience: Optional[int] = None,
        time_budget_seconds: Optional[float] = None,
        eval_budget: Optional[int] = None,
        use_surrogate: bool = True,
        surrogate_pool: int = 16,
        surrogate_refit_every: int = 4,
        surrogate_min_history: int = 12,
        surrogate_kappa: float = 0.0,
        restart_patience: Optional[int] = None,
        restart_fraction: float = 0.5,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        if not isinstance(space, SearchSpace):
            space = SearchSpace(space)
        if int(population_size) < 4:
            raise ValueError(
                "population_size must be >= 4 (DE mutation needs three distinct"
                f" individuals other than the target); got {population_size}"
            )
        self.space = space
        self.objective = objective
        self.population_size = int(population_size)
        self.iterations = int(iterations)
        self.F = float(F)
        self.CR = float(CR)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.early_stop_patience = early_stop_patience
        self.time_budget_seconds = time_budget_seconds
        self.eval_budget = eval_budget
        self.use_surrogate = bool(use_surrogate)
        self.surrogate_pool = int(surrogate_pool)
        self.surrogate_refit_every = int(surrogate_refit_every)
        self.surrogate_min_history = int(surrogate_min_history)
        self.surrogate_kappa = float(surrogate_kappa)
        self.restart_patience = restart_patience
        self.restart_fraction = float(restart_fraction)
        self._surrogate = None
        self._last_surrogate_refit_at = -1
        self._stagnation_iters = 0
        self._best_loss_at_last_check = float("inf")
        self.rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)

        self.population: Optional[np.ndarray] = None
        self._latest_losses: Optional[np.ndarray] = None
        self._velocities: Optional[np.ndarray] = None
        self._pbest: Optional[np.ndarray] = None
        self._pbest_losses: Optional[np.ndarray] = None
        self.best_x: Optional[np.ndarray] = None
        self.best_loss: float = float("inf")
        self._history: List[Dict[str, Any]] = []
        self._t0: float = 0.0

    # -- evaluation --------------------------------------------------------

    def _eval(self, x: np.ndarray, iteration: int) -> float:
        params = self.space.decode(x)
        try:
            loss = float(self.objective(params))
        except Exception as exc:  # noqa: BLE001
            loss = float("inf")
            params = {**params, "_error": repr(exc)}
        if not np.isfinite(loss):
            loss = float("inf")
        self._history.append(
            {
                "iteration": iteration,
                "loss": loss,
                "elapsed": time.time() - self._t0,
                "optimizer": self.name,
                **params,
            }
        )
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_x = x.copy()
        return loss

    def _budget_exhausted(self) -> bool:
        if self.eval_budget is not None and len(self._history) >= self.eval_budget:
            return True
        if self.time_budget_seconds is not None:
            if (time.time() - self._t0) >= self.time_budget_seconds:
                return True
        return False

    # -- Surrogate-assisted candidate selection -----------------------------

    def _refit_surrogate(self, iteration: int) -> None:
        """Refit the RandomForest surrogate on the (params, loss) history.

        The surrogate is small (30 trees) and does not require feature
        normalization, because all training points are already expressed
        in the SearchSpace internal coordinates. The refit is performed
        once every few iterations so that the model stays current
        without dominating the runtime.
        """
        if not self.use_surrogate:
            return
        if len(self._history) < self.surrogate_min_history:
            return
        if iteration == self._last_surrogate_refit_at:
            return
        if (iteration - self._last_surrogate_refit_at) < self.surrogate_refit_every:
            return
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            self.use_surrogate = False
            return

        # Reconstruct internal-coords X from history params using to_internal
        X = []
        y = []
        for rec in self._history:
            if not np.isfinite(rec.get("loss", float("inf"))):
                continue
            try:
                vec = [self.space.dims[n].to_internal(rec[n]) for n in self.space.names]
            except Exception:
                continue
            X.append(vec)
            y.append(rec["loss"])
        if len(X) < self.surrogate_min_history:
            return
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        # Cap losses at 99th percentile to dampen inf/outliers
        y_cap = np.quantile(y_arr, 0.99)
        y_arr = np.minimum(y_arr, y_cap)
        self._surrogate = RandomForestRegressor(
            n_estimators=30, max_depth=10, n_jobs=1, random_state=42
        ).fit(X_arr, y_arr)
        self._last_surrogate_refit_at = iteration

    def _surrogate_predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) predictions from the RandomForest surrogate.

        The standard deviation across trees is used as a proxy for
        epistemic uncertainty. In regions of the search space that are
        sparsely covered by training points, the individual trees tend
        to disagree, which produces a larger spread. This serves as a
        discrete analog of the predictive variance from a Gaussian
        Process and is what allows the UCB-style acquisition below.
        """
        per_tree = np.stack([t.predict(X) for t in self._surrogate.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    def _surrogate_filter(self, base: np.ndarray) -> np.ndarray:
        """Score K nearby alternatives and return the one with the
        lowest acquisition value.

        The filter generates K Gaussian-perturbed neighbors of the input
        candidate and asks the surrogate to score each one. The score is
        the lower confidence bound (LCB):

            LCB(x) = mean(x) - kappa * std(x)

        With kappa = 0 the filter picks the candidate with the lowest
        predicted loss (pure exploit). With a large kappa it picks the
        most uncertain one (pure explore). The value kappa = 1.5 is the
        common Bayesian Optimization default.

        The LCB form is used because the objective is being minimized,
        which is equivalent to the negative UCB used in BO when the
        objective is being maximized.
        """
        if self._surrogate is None or not self.use_surrogate:
            return base
        K = max(self.surrogate_pool, 2)
        scale = 0.1 * (self.space.highs - self.space.lows)
        noise = self.rng.normal(0.0, 1.0, size=(K - 1, self.space.n_dims)) * scale
        candidates = np.vstack([base.reshape(1, -1), base + noise])
        candidates = self.space.clip(candidates)
        try:
            mean, std = self._surrogate_predict_with_std(candidates)
            # LCB (lower-confidence bound) for minimization
            acq = mean - self.surrogate_kappa * std
            return candidates[int(np.argmin(acq))]
        except Exception:
            return base

    def _maybe_restart(self) -> bool:
        """Optionally restart part of the population when the search
        has stagnated.

        If the best loss has not improved for `restart_patience`
        iterations, the worst `restart_fraction` of the population is
        replaced by freshly sampled random points. The top half is
        kept, so that accumulated knowledge is preserved, while the
        bottom half is reseeded with diversity. This reduces run-to-run
        variance in cases where the search settles into a local basin
        early.
        """
        if self.restart_patience is None:
            return False
        if self.best_loss + 1e-12 < self._best_loss_at_last_check:
            self._best_loss_at_last_check = self.best_loss
            self._stagnation_iters = 0
            return False
        self._stagnation_iters += 1
        if self._stagnation_iters < self.restart_patience:
            return False

        # Restart: randomize worst restart_fraction of population
        n = self.population_size
        n_restart = max(1, int(self.restart_fraction * n))
        order = np.argsort(self._latest_losses)
        worst = order[-n_restart:]
        new_pos = self.space.sample(self.rng, n_restart)
        self.population[worst] = new_pos
        # Reset pbest and velocities for restarted members
        if self._pbest is not None:
            self._pbest[worst] = self.population[worst]
            # pbest_losses will be repaired by next eval pass
        if self._velocities is not None:
            self._velocities[worst] = 0.0
        self._stagnation_iters = 0
        return True

    # -- GWO leadership update --------------------------------------------

    def _gwo_step(self, iteration: int) -> None:
        n, d = self.population.shape
        order = np.argsort(self._latest_losses)
        alpha = self.population[order[0]]
        beta = self.population[order[1 if n > 1 else 0]]
        delta = self.population[order[2 if n > 2 else (1 if n > 1 else 0)]]

        a = 2.0 * (1.0 - iteration / max(self.iterations, 1))

        def leader_step(leader: np.ndarray) -> np.ndarray:
            r1 = self.rng.random((n, d))
            r2 = self.rng.random((n, d))
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D = np.abs(C * leader - self.population)
            return leader - A * D

        X1 = leader_step(alpha)
        X2 = leader_step(beta)
        X3 = leader_step(delta)
        self.population = (X1 + X2 + X3) / 3.0

    # -- Personal-best bookkeeping for PSO --------------------------------

    def _update_pbest(self) -> None:
        better = self._latest_losses < self._pbest_losses
        if better.any():
            self._pbest[better] = self.population[better]
            self._pbest_losses[better] = self._latest_losses[better]

    # -- PSO velocity/position update --------------------------------------

    def _pso_step(self, iteration: int) -> None:
        n, d = self.population.shape
        # Linearly decreasing inertia: explore early, refine late.
        frac = iteration / max(self.iterations, 1)
        w = self.w_max - (self.w_max - self.w_min) * frac
        r1 = self.rng.random((n, d))
        r2 = self.rng.random((n, d))
        cognitive = self.c1 * r1 * (self._pbest - self.population)
        social = self.c2 * r2 * (self.best_x - self.population)
        self._velocities = w * self._velocities + cognitive + social
        self.population = self.population + self._velocities

    # -- main loop ---------------------------------------------------------

    def optimize(self) -> OptimizeResult:
        self._t0 = time.time()
        self._history.clear()
        self.population = self.space.sample(self.rng, self.population_size)
        self.best_x = None
        self.best_loss = float("inf")
        self._latest_losses = np.array(
            [self._eval(self.population[i], 0) for i in range(self.population_size)]
        )
        # PSO state: each particle's personal best starts at its initial position
        self._pbest = self.population.copy()
        self._pbest_losses = self._latest_losses.copy()
        self._velocities = np.zeros_like(self.population)

        stagnation = 0
        prev_best = self.best_loss
        stopped_reason = "iterations"

        for it in range(1, self.iterations + 1):
            if self._budget_exhausted():
                stopped_reason = "budget"
                break

            # Refresh surrogate every K iterations
            self._refit_surrogate(it)

            # ----- Stage 1: DE -----
            new_pop = self.population.copy()
            new_losses = self._latest_losses.copy()
            for i in range(self.population_size):
                idxs = [j for j in range(self.population_size) if j != i]
                a_i, b_i, c_i = self.rng.choice(idxs, 3, replace=False)
                donor = self.population[a_i] + self.F * (
                    self.population[b_i] - self.population[c_i]
                )
                mask = self.rng.random(self.space.n_dims) < self.CR
                if not mask.any():
                    mask[self.rng.integers(self.space.n_dims)] = True
                trial = np.where(mask, donor, self.population[i])
                trial = self.space.clip(trial)
                # Surrogate-assisted refinement: pick the best neighbor of the trial
                trial = self._surrogate_filter(trial)
                trial_loss = self._eval(trial, it)
                if trial_loss < self._latest_losses[i]:
                    new_pop[i] = trial
                    new_losses[i] = trial_loss
                if self._budget_exhausted():
                    stopped_reason = "budget"
                    break
            self.population = new_pop
            self._latest_losses = new_losses
            self._update_pbest()
            if stopped_reason == "budget":
                break

            # ----- Stage 2: GWO leadership -----
            self._gwo_step(it)
            self.population = self.space.clip(self.population)
            # Apply surrogate filter to each post-GWO position
            if self.use_surrogate and self._surrogate is not None:
                self.population = np.stack(
                    [self._surrogate_filter(self.population[i])
                     for i in range(self.population_size)]
                )
                self.population = self.space.clip(self.population)
            new_losses = np.empty(self.population_size)
            for i in range(self.population_size):
                if self._budget_exhausted():
                    new_losses[i:] = self._latest_losses[i:]
                    stopped_reason = "budget"
                    break
                new_losses[i] = self._eval(self.population[i], it)
            self._latest_losses = new_losses
            self._update_pbest()
            if stopped_reason == "budget":
                break

            # ----- Stage 3: PSO refinement -----
            self._pso_step(it)
            self.population = self.space.clip(self.population)
            if self.use_surrogate and self._surrogate is not None:
                self.population = np.stack(
                    [self._surrogate_filter(self.population[i])
                     for i in range(self.population_size)]
                )
                self.population = self.space.clip(self.population)
            new_losses = np.empty(self.population_size)
            for i in range(self.population_size):
                if self._budget_exhausted():
                    new_losses[i:] = self._latest_losses[i:]
                    stopped_reason = "budget"
                    break
                new_losses[i] = self._eval(self.population[i], it)
            self._latest_losses = new_losses
            self._update_pbest()
            if stopped_reason == "budget":
                break

            # Stagnation-triggered restart (if enabled). Operates on
            # post-PSO state so it doesn't perturb the current iteration.
            if self._maybe_restart():
                # Re-evaluate the freshly-randomized members so _latest_losses
                # is consistent before the next iteration.
                for i in range(self.population_size):
                    if self._budget_exhausted():
                        stopped_reason = "budget"
                        break
                    self._latest_losses[i] = self._eval(self.population[i], it)
                self._update_pbest()
                if stopped_reason == "budget":
                    break

            if self.verbose:
                print(f"[{self.name}] iter {it}/{self.iterations}  best={self.best_loss:.6g}")
            if self.early_stop_patience is not None:
                if self.best_loss < prev_best - 1e-12:
                    stagnation = 0
                    prev_best = self.best_loss
                else:
                    stagnation += 1
                    if stagnation >= self.early_stop_patience:
                        stopped_reason = "early_stop"
                        break

        elapsed = time.time() - self._t0
        history_df = pd.DataFrame(self._history)
        if not history_df.empty:
            history_df["running_best"] = history_df["loss"].cummin()
        return OptimizeResult(
            best_params=self.space.decode(self.best_x) if self.best_x is not None else {},
            best_loss=self.best_loss,
            history=history_df,
            n_evals=len(self._history),
            elapsed_seconds=elapsed,
            stopped_reason=stopped_reason,
        )


__all__ = [
    "Float",
    "Int",
    "Categorical",
    "SearchSpace",
    "OptimizeResult",
    "HDGPSO",
]
