"""HDGPSOMF: multi-fidelity extension of HDGPSO.

Extends :class:`hdgpso.HDGPSO` with BOHB-style successive halving
(Hyperband, Li et al. 2017 / BOHB, Falkner 2018):

  - Each candidate is first evaluated at LOW fidelity (cheap, noisy
    proxy: e.g. cv=1, reduced epochs, fewer collocation points).
  - The top ``verify_fraction`` of candidates per stage are re-evaluated
    at FULL fidelity (1.0) to confirm.
  - Budget is accounted in *fidelity-units*: a low-fidelity eval at
    ``fidelity=0.3`` consumes 0.3 budget units; a full eval costs 1.0.
  - The surrogate is trained on FULL-fidelity points only (cleaner
    training signal, avoids low-fidelity miscalibration).

To use HDGPSOMF, supply an objective callable that accepts a
``fidelity`` keyword argument in [0, 1]. Example multi-fidelity
objective factories are provided in
``benchmarks/multifidelity_objectives.py`` (sklearn / MLP / PINN
flavors); users are encouraged to write their own for new problems.

Example::

    from hdgpso import HDGPSOMF, SearchSpace, Float, Int

    def objective(params, fidelity=1.0):
        # Cheap proxy: train with fewer epochs at lower fidelity
        n_epochs = max(5, int(50 * fidelity))
        return train_model(params, epochs=n_epochs)

    space = SearchSpace({"lr": Float(1e-5, 1e-2, log=True),
                          "layers": Int(2, 8)})
    result = HDGPSOMF(space, objective, eval_budget=60).optimize()
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from .core import HDGPSO, OptimizeResult, SearchSpace  # noqa: F401


class HDGPSOMF(HDGPSO):
    """HDGPSO with multi-fidelity successive halving.

    Parameters
    ----------
    space, objective : see :class:`HDGPSO`.
    eval_budget : int, optional
        Total fidelity-units permitted. One full-fidelity evaluation
        costs 1.0 unit; a 0.3-fidelity probe costs 0.3 units.
    low_fidelity : float
        Fidelity used for proposal probes (default 0.3).
    verify_fraction : float
        Fraction of proposals per stage re-evaluated at full fidelity
        (default 0.4).
    use_surrogate, surrogate_pool, surrogate_refit_every,
    surrogate_min_history, surrogate_kappa : see :class:`HDGPSO`.
    restart_patience : Optional[int]
        Disabled by default for HDGPSOMF (restart destroys converged
        candidates that the multi-fidelity surrogate has invested in).

    Notes
    -----
    The surrogate is trained only on FULL-fidelity points to avoid
    noisy low-fidelity miscalibration. This mirrors the discipline
    used by BOHB (Falkner 2018).
    """

    name = "HDGPSO-MF"

    def __init__(
        self,
        space,
        objective,
        population_size: int = 5,
        iterations: int = 30,
        F: float = 0.8,
        CR: float = 0.5,
        c1: float = 2.0,
        c2: float = 2.0,
        w_max: float = 0.7,
        w_min: float = 0.4,
        eval_budget: Optional[int] = None,
        low_fidelity: float = 0.3,
        verify_fraction: float = 0.4,
        use_surrogate: bool = True,
        surrogate_pool: int = 16,
        surrogate_refit_every: int = 4,
        surrogate_min_history: int = 6,
        surrogate_kappa: float = 0.0,
        restart_patience: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(
            space=space, objective=objective,
            population_size=population_size, iterations=iterations,
            F=F, CR=CR, c1=c1, c2=c2, w_max=w_max, w_min=w_min,
            eval_budget=None,  # we manage budget in fidelity-units
            use_surrogate=use_surrogate, surrogate_pool=surrogate_pool,
            surrogate_refit_every=surrogate_refit_every,
            surrogate_min_history=surrogate_min_history,
            surrogate_kappa=surrogate_kappa,
            restart_patience=restart_patience,
            seed=seed, verbose=verbose,
        )
        self.fidelity_budget = float(eval_budget) if eval_budget else None
        self.low_fidelity = float(low_fidelity)
        self.verify_fraction = float(verify_fraction)
        self._fidelity_used: float = 0.0

    # ---- multi-fidelity eval with cost accounting -----------------------

    def _eval(self, x: np.ndarray, iteration: int, fidelity: float = 1.0) -> float:
        params = self.space.decode(x)
        try:
            loss = float(self.objective(params, fidelity=fidelity))
        except TypeError:
            loss = float(self.objective(params))
        if not np.isfinite(loss):
            loss = float("inf")
        self._fidelity_used += fidelity
        self._history.append({
            "iteration": iteration,
            "loss": loss,
            "fidelity": fidelity,
            "elapsed": time.time() - self._t0,
            "optimizer": self.name,
            **params,
        })
        if fidelity >= 0.99 and loss < self.best_loss:
            self.best_loss = loss
            self.best_x = x.copy()
        return loss

    def _budget_exhausted(self) -> bool:
        if self.fidelity_budget is not None:
            return self._fidelity_used >= self.fidelity_budget
        return False

    def _verify_top(self, candidates: np.ndarray, low_losses: np.ndarray,
                    iteration: int) -> np.ndarray:
        n = len(candidates)
        n_verify = max(1, int(np.ceil(self.verify_fraction * n)))
        order = np.argsort(low_losses)
        top_idx = order[:n_verify]
        full_losses = low_losses.copy()
        for i in top_idx:
            if self._budget_exhausted():
                break
            full_losses[i] = self._eval(candidates[i], iteration, fidelity=1.0)
        return full_losses

    def _refit_surrogate(self, iteration: int) -> None:
        """Refit surrogate using only FULL-fidelity history points."""
        if not self.use_surrogate:
            return
        full_hist = [h for h in self._history if h.get("fidelity", 1.0) >= 0.99
                     and np.isfinite(h.get("loss", float("inf")))]
        if len(full_hist) < self.surrogate_min_history:
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
        X, y = [], []
        for rec in full_hist:
            try:
                vec = [self.space.dims[n].to_internal(rec[n]) for n in self.space.names]
            except Exception:
                continue
            X.append(vec); y.append(rec["loss"])
        if len(X) < self.surrogate_min_history:
            return
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        y_cap = np.quantile(y_arr, 0.99)
        y_arr = np.minimum(y_arr, y_cap)
        self._surrogate = RandomForestRegressor(
            n_estimators=30, max_depth=10, n_jobs=1, random_state=42,
        ).fit(X_arr, y_arr)
        self._last_surrogate_refit_at = iteration

    # ---- main optimization loop -----------------------------------------

    def optimize(self) -> OptimizeResult:
        self._t0 = time.time()
        self._history.clear()
        self._fidelity_used = 0.0
        self.population = self.space.sample(self.rng, self.population_size)
        self.best_x = None
        self.best_loss = float("inf")

        # Initial pop: evaluate at FULL fidelity for clean surrogate baseline
        self._latest_losses = np.empty(self.population_size)
        for i in range(self.population_size):
            if self._budget_exhausted():
                break
            self._latest_losses[i] = self._eval(self.population[i], 0, fidelity=1.0)

        self._pbest = self.population.copy()
        self._pbest_losses = self._latest_losses.copy()
        self._velocities = np.zeros_like(self.population)
        stopped_reason = "iterations"

        for it in range(1, self.iterations + 1):
            if self._budget_exhausted():
                stopped_reason = "budget"
                break

            self._refit_surrogate(it)

            # ----- Stage 1: DE at LOW fidelity, verify top -----
            trials = np.empty_like(self.population)
            for i in range(self.population_size):
                idxs = [j for j in range(self.population_size) if j != i]
                a_i, b_i, c_i = self.rng.choice(idxs, 3, replace=False)
                donor = self.population[a_i] + self.F * (self.population[b_i] - self.population[c_i])
                mask = self.rng.random(self.space.n_dims) < self.CR
                if not mask.any():
                    mask[self.rng.integers(self.space.n_dims)] = True
                trial = np.where(mask, donor, self.population[i])
                trial = self.space.clip(trial)
                trial = self._surrogate_filter(trial)
                trials[i] = trial
            low_losses = np.empty(self.population_size)
            for i in range(self.population_size):
                if self._budget_exhausted():
                    low_losses[i] = float("inf")
                else:
                    low_losses[i] = self._eval(trials[i], it, fidelity=self.low_fidelity)
            full_losses = self._verify_top(trials, low_losses, it)
            for i in range(self.population_size):
                if full_losses[i] < self._latest_losses[i]:
                    self.population[i] = trials[i]
                    self._latest_losses[i] = full_losses[i]
            self._update_pbest()
            if self._budget_exhausted():
                stopped_reason = "budget"; break

            # ----- Stage 2: GWO leadership at LOW fidelity, verify top -----
            self._gwo_step(it)
            self.population = self.space.clip(self.population)
            if self.use_surrogate and self._surrogate is not None:
                self.population = np.stack(
                    [self._surrogate_filter(self.population[i])
                     for i in range(self.population_size)]
                )
                self.population = self.space.clip(self.population)
            low_losses = np.empty(self.population_size)
            for i in range(self.population_size):
                if self._budget_exhausted():
                    low_losses[i] = float("inf")
                else:
                    low_losses[i] = self._eval(self.population[i], it, fidelity=self.low_fidelity)
            self._latest_losses = self._verify_top(self.population, low_losses, it)
            self._update_pbest()
            if self._budget_exhausted():
                stopped_reason = "budget"; break

            # ----- Stage 3: PSO refinement at LOW fidelity, verify top -----
            self._pso_step(it)
            self.population = self.space.clip(self.population)
            if self.use_surrogate and self._surrogate is not None:
                self.population = np.stack(
                    [self._surrogate_filter(self.population[i])
                     for i in range(self.population_size)]
                )
                self.population = self.space.clip(self.population)
            low_losses = np.empty(self.population_size)
            for i in range(self.population_size):
                if self._budget_exhausted():
                    low_losses[i] = float("inf")
                else:
                    low_losses[i] = self._eval(self.population[i], it, fidelity=self.low_fidelity)
            self._latest_losses = self._verify_top(self.population, low_losses, it)
            self._update_pbest()

            if self.verbose:
                print(f"[{self.name}] iter {it} best={self.best_loss:.6g} "
                      f"used={self._fidelity_used:.1f}/{self.fidelity_budget}")

        elapsed = time.time() - self._t0
        history_df = pd.DataFrame(self._history)
        if not history_df.empty:
            history_df["running_best"] = history_df.apply(
                lambda r: r["loss"] if r.get("fidelity", 1.0) >= 0.99 else np.nan,
                axis=1
            ).cummin().ffill()
        return OptimizeResult(
            best_params=self.space.decode(self.best_x) if self.best_x is not None else {},
            best_loss=self.best_loss,
            history=history_df,
            n_evals=len(self._history),
            elapsed_seconds=elapsed,
            stopped_reason=stopped_reason,
        )


__all__ = ["HDGPSOMF"]
