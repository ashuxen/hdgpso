"""SMAC3 adapter for the benchmark harness.

Three defaults are overridden to keep the evaluation budget matched to the
other tuners: deterministic=True, a constant-size initial design, and
overwrite=True. The surrogate is set to 30 trees / depth 10 to match HDGPSO's.
"""
from __future__ import annotations

import shutil
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ConfigSpace import (
    Categorical as CSCategorical,
    ConfigurationSpace,
    Float as CSFloat,
    Integer as CSInteger,
)
from smac import HyperparameterOptimizationFacade as HPOFacade, Scenario
from smac.runhistory import TrialValue

from hdgpso import Categorical, Float, Int, OptimizeResult, SearchSpace

# Population-matched warm-up: HDGPSO initialises N=5 candidates before its
# first operator stage, so SMAC gets a constant 5-configuration initial design.
N_INITIAL = 5

# SMAC's random forest cannot ingest non-finite costs. Failed evaluations are
# reported to the model at this penalty while the true inf is kept in history.
_FAIL_PENALTY = 1e10


class SMACTuner:
    """SMAC3 wrapped to match the benchmark harness tuner protocol."""

    name = "SMAC"

    def __init__(
        self,
        space,
        objective,
        budget: int = 60,
        seed: Optional[int] = None,
        population_size: int = 5,
        n_initial: int = N_INITIAL,
        **_,
    ):
        self.space = space if isinstance(space, SearchSpace) else SearchSpace(space)
        self.objective = objective
        self.budget = int(budget)
        self.seed = 0 if seed is None else int(seed)
        self.n_initial = int(n_initial)
        self._cat_maps: Dict[str, list] = {}

    # -- search-space translation -----------------------------------------

    def _build_config_space(self) -> ConfigurationSpace:
        """Translate the harness SearchSpace into a ConfigurationSpace.

        Categorical dimensions are encoded as index strings rather than the
        raw choices, because the harness allows mixed-type choice lists
        (e.g. max_features = ["sqrt", "log2", 0.5, 1.0]) that ConfigSpace
        would otherwise coerce to a single type.
        """
        cs = ConfigurationSpace(seed=self.seed)
        params = []
        for name in self.space.names:
            d = self.space.dims[name]
            if isinstance(d, Float):
                params.append(CSFloat(name, (d.low, d.high), log=d.log))
            elif isinstance(d, Int):
                params.append(CSInteger(name, (d.low, d.high), log=d.log))
            elif isinstance(d, Categorical):
                self._cat_maps[name] = list(d.choices)
                params.append(
                    CSCategorical(name, [str(i) for i in range(len(d.choices))])
                )
            else:
                raise TypeError(f"Unsupported dimension type for {name}: {type(d)}")
        cs.add(params)
        return cs

    def _decode(self, config) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, value in dict(config).items():
            if name in self._cat_maps:
                out[name] = self._cat_maps[name][int(value)]
            else:
                out[name] = value
        return out

    # -- main loop ---------------------------------------------------------

    def optimize(self) -> OptimizeResult:
        import time

        cs = self._build_config_space()
        out_dir = tempfile.mkdtemp(prefix="smac_")
        t0 = time.time()
        history = []
        best_loss = float("inf")
        best_params: Dict[str, Any] = {}

        try:
            scenario = Scenario(
                cs,
                name=f"cell_{self.seed}_{self.budget}",
                output_directory=out_dir,
                deterministic=True,          # trap 1
                n_trials=self.budget,
                seed=self.seed,
                use_default_config=False,
                n_workers=1,
                trial_walltime_limit=None,   # avoid pynisher subprocesses on Windows
                trial_memory_limit=None,
            )
            smac = HPOFacade(
                scenario,
                initial_design=HPOFacade.get_initial_design(
                    scenario, n_configs=self.n_initial      # trap 2
                ),
                model=HPOFacade.get_model(
                    scenario, n_trees=30, max_depth=10      # match HDGPSO's surrogate
                ),
                intensifier=HPOFacade.get_intensifier(
                    scenario, max_config_calls=1            # trap 1 (belt and braces)
                ),
                overwrite=True,                             # trap 3
                logging_level=False,
            )

            for _ in range(self.budget):
                info = smac.ask()
                params = self._decode(info.config)
                try:
                    loss = float(self.objective(params))
                except Exception:
                    loss = float("inf")
                if not np.isfinite(loss):
                    loss = float("inf")
                smac.tell(
                    info,
                    TrialValue(
                        cost=_FAIL_PENALTY if not np.isfinite(loss) else loss,
                        time=0.0,
                    ),
                )
                history.append(
                    {
                        "iteration": len(history),
                        "loss": loss,
                        "elapsed": time.time() - t0,
                        "optimizer": self.name,
                        **params,
                    }
                )
                if loss < best_loss:
                    best_loss, best_params = loss, params
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

        hist_df = pd.DataFrame(history)
        if not hist_df.empty:
            hist_df["running_best"] = hist_df["loss"].cummin()

        return OptimizeResult(
            best_params=best_params,
            best_loss=best_loss,
            history=hist_df,
            n_evals=len(history),
            elapsed_seconds=time.time() - t0,
            stopped_reason="budget",
        )
