"""Benchmark harness across tuners, models, datasets, and seeds.

This script runs every registered tuner on a configurable set of
(dataset, model) tasks and writes the per-trial history together
with a per-run summary to CSV.

Command-line usage:
    python benchmark.py --budget 60 --seeds 3 --out results/

Programmatic usage:
    from benchmark import run_benchmark
    summary, history = run_benchmark(budget=60, seeds=[0, 1, 2])
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_wine,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from hdgpso import Categorical, Float, Int, SearchSpace
from tuners import ALL_TUNERS


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


@dataclass
class Task:
    name: str
    dataset: str
    model_name: str
    X: np.ndarray
    y: np.ndarray
    task_type: str  # 'classification' or 'regression'
    cv: int = 3
    scoring: str = "accuracy"


def _safe_fetch_openml(name: str, version: int = 1):
    try:
        return fetch_openml(name=name, version=version, as_frame=False, parser="auto")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not fetch {name}: {exc}")
        return None


def get_datasets(include_openml: bool = True, include_deep: bool = False) -> Dict[str, Dict[str, Any]]:
    """Return a dictionary of available datasets keyed by name.

    Each value is a small dictionary with the input matrix X, the
    target vector y, and a ``task`` string. When ``include_deep`` is
    True, a synthetic ``pinn_heat`` placeholder is added; in that case
    X and y are dummies, because the PINN objective generates its own
    data internally.
    """
    out: Dict[str, Dict[str, Any]] = {}

    # sklearn built-ins (always available, fast)
    bc = load_breast_cancer()
    out["breast_cancer"] = {"X": bc.data, "y": bc.target, "task": "classification"}

    wine = load_wine()
    out["wine"] = {"X": wine.data, "y": wine.target, "task": "classification"}

    digits = load_digits()
    out["digits"] = {"X": digits.data, "y": digits.target, "task": "classification"}

    diab = load_diabetes()
    out["diabetes"] = {"X": diab.data, "y": diab.target, "task": "regression"}

    # California housing — slightly larger regression
    try:
        from sklearn.datasets import fetch_california_housing

        cal = fetch_california_housing()
        # subsample for speed
        idx = np.random.RandomState(0).choice(len(cal.data), 4000, replace=False)
        out["california_housing"] = {
            "X": cal.data[idx],
            "y": cal.target[idx],
            "task": "regression",
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  ! california_housing skipped: {exc}")

    if include_openml:
        # credit-g: 1000 rows, 20 features, binary classification
        ds = _safe_fetch_openml("credit-g", version=1)
        if ds is not None:
            # encode categorical features
            df = pd.DataFrame(ds.data)
            df = pd.get_dummies(df, drop_first=True)
            X = df.values.astype(float)
            y = (np.asarray(ds.target) == "good").astype(int)
            out["credit-g"] = {"X": X, "y": y, "task": "classification"}

    if include_deep:
        # PINN-Heat: synthetic dataset; objective generates collocation points internally.
        # X/y placeholders are 2 dummy rows so any model that reads them won't crash.
        out["pinn_heat"] = {
            "X": np.zeros((2, 2), dtype=np.float32),
            "y": np.zeros(2, dtype=np.float32),
            "task": "pinn",
        }

    return out


# ---------------------------------------------------------------------------
# Model search spaces
# ---------------------------------------------------------------------------


def _rf_space() -> SearchSpace:
    return SearchSpace(
        {
            "n_estimators": Int(20, 300),
            "max_depth": Int(2, 20),
            "min_samples_split": Int(2, 20),
            "min_samples_leaf": Int(1, 20),
            "max_features": Categorical(["sqrt", "log2", 0.5, 1.0]),
        }
    )


def _gbm_space() -> SearchSpace:
    return SearchSpace(
        {
            "n_estimators": Int(20, 300),
            "learning_rate": Float(1e-3, 0.3, log=True),
            "max_depth": Int(2, 10),
            "min_samples_split": Int(2, 20),
            "subsample": Float(0.5, 1.0),
        }
    )


def _xgb_space() -> SearchSpace:
    return SearchSpace(
        {
            "n_estimators": Int(20, 400),
            "learning_rate": Float(1e-3, 0.3, log=True),
            "max_depth": Int(2, 12),
            "subsample": Float(0.5, 1.0),
            "colsample_bytree": Float(0.5, 1.0),
            "reg_lambda": Float(1e-3, 10.0, log=True),
        }
    )


MODEL_SPACES = {
    "RandomForest": _rf_space,
    "GradientBoosting": _gbm_space,
    "XGBoost": _xgb_space,
}

# Deep-net models are registered lazily (only when torch is available).
try:
    from deep_objectives import DEEP_MODEL_SPACES
    MODEL_SPACES.update(DEEP_MODEL_SPACES)
except ImportError:
    pass


def _build_model(model_name: str, task_type: str, params: Dict[str, Any], seed: int):
    if model_name == "RandomForest":
        cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
        return cls(**params, random_state=seed, n_jobs=1)
    if model_name == "GradientBoosting":
        cls = (
            GradientBoostingClassifier
            if task_type == "classification"
            else GradientBoostingRegressor
        )
        return cls(**params, random_state=seed)
    if model_name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if task_type == "classification" else XGBRegressor
        return cls(
            **params,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
            use_label_encoder=False,
            eval_metric="logloss" if task_type == "classification" else "rmse",
        )
    raise ValueError(f"Unknown model {model_name}")


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------


def make_objective(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    model_name: str,
    cv: int,
    seed: int,
) -> Callable[[Dict[str, Any]], float]:
    """Return an objective function that maps hyperparameters to a loss value (lower is better)."""

    # Deep-net dispatch
    if model_name == "MLP":
        from deep_objectives import make_mlp_objective
        if task_type != "classification":
            raise ValueError("MLP only supported on classification datasets")
        return make_mlp_objective(X, y, seed=seed)
    if model_name == "PINN-Heat":
        from deep_objectives import make_pinn_heat_objective
        return make_pinn_heat_objective(seed=seed)

    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
    Xs = StandardScaler().fit_transform(X)

    def objective(params: Dict[str, Any]) -> float:
        # Some skopt/grid adapters pass numpy scalars; coerce ints.
        clean: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            else:
                clean[k] = v
        model = _build_model(model_name, task_type, clean, seed)
        try:
            scores = cross_val_score(model, Xs, y, cv=cv, scoring=scoring, n_jobs=1)
        except Exception:
            return float("inf")
        # cross_val_score returns score (higher better); we want loss (lower better)
        return -float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Run one (dataset, model, tuner, seed) cell
# ---------------------------------------------------------------------------


def run_one(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    model_name: str,
    tuner_name: str,
    seed: int,
    budget: int,
    cv: int = 3,
    population_size: int = 10,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    space = MODEL_SPACES[model_name]()
    # HDGPSO-MF requires a fidelity-aware objective; others get the standard one.
    if tuner_name == "HDGPSO-MF":
        from multifidelity_objectives import (
            make_fidelity_sklearn_objective,
            make_fidelity_mlp_objective,
            make_fidelity_pinn_objective,
        )
        if model_name == "MLP":
            objective = make_fidelity_mlp_objective(X, y, seed=seed)
        elif model_name == "PINN-Heat":
            objective = make_fidelity_pinn_objective(seed=seed)
        else:
            objective = make_fidelity_sklearn_objective(
                X, y, task_type, model_name, cv, seed,
                benchmark_make_objective=make_objective,
                benchmark_build_model=_build_model,
            )
    else:
        objective = make_objective(X, y, task_type, model_name, cv, seed)
    TunerCls = ALL_TUNERS[tuner_name]

    t0 = time.time()
    tuner = TunerCls(
        space=space,
        objective=objective,
        budget=budget,
        seed=seed,
        population_size=population_size,
    )
    try:
        result = tuner.optimize()
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {tuner_name} failed on {dataset_name}/{model_name} seed={seed}: {exc!r}")
        return (
            {
                "dataset": dataset_name,
                "model": model_name,
                "tuner": tuner_name,
                "seed": seed,
                "best_loss": float("inf"),
                "n_evals": 0,
                "elapsed": time.time() - t0,
                "error": repr(exc),
            },
            pd.DataFrame(),
        )

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "tuner": tuner_name,
        "seed": seed,
        "best_loss": result.best_loss,
        "n_evals": result.n_evals,
        "elapsed": result.elapsed_seconds,
        "stopped": result.stopped_reason,
        "error": "",
    }
    history = result.history.copy()
    history["dataset"] = dataset_name
    history["model"] = model_name
    history["tuner"] = tuner_name
    history["seed"] = seed
    return summary, history


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_benchmark(
    budget: int = 60,
    seeds: List[int] = (0, 1, 2),
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    tuners: Optional[List[str]] = None,
    cv: int = 3,
    population_size: int = 10,
    include_openml: bool = True,
    include_deep: bool = False,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full benchmark grid and return ``(summary_df, history_df)``.

    Incompatible (dataset, model) pairs are skipped automatically.
    Examples of skipped combinations are the MLP applied to regression
    datasets, or any sklearn model applied to the ``pinn_heat``
    placeholder dataset.
    """
    all_ds = get_datasets(include_openml=include_openml, include_deep=include_deep)
    if datasets is not None:
        all_ds = {k: v for k, v in all_ds.items() if k in datasets}
    if models is None:
        models = list(MODEL_SPACES.keys())
    if tuners is None:
        tuners = list(ALL_TUNERS.keys())

    # Probe XGBoost availability lazily; drop if missing
    if "XGBoost" in models:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            if verbose:
                print("  ! xgboost not installed; skipping XGBoost model")
            models = [m for m in models if m != "XGBoost"]

    # Probe optional tuners
    optional = {"Bayes": "skopt", "Optuna-TPE": "optuna", "PSO": "pyswarms"}
    for t, pkg in optional.items():
        if t in tuners:
            try:
                __import__(pkg)
            except ImportError:
                if verbose:
                    print(f"  ! {pkg} not installed; skipping {t}")
                tuners = [x for x in tuners if x != t]

    summaries: List[Dict[str, Any]] = []
    histories: List[pd.DataFrame] = []

    # Pre-filter incompatible (dataset, model) pairs so the progress count is honest.
    pairs = []
    for ds_name, ds in all_ds.items():
        for model_name in models:
            task = ds["task"]
            if model_name == "PINN-Heat" and ds_name != "pinn_heat":
                continue
            if ds_name == "pinn_heat" and model_name != "PINN-Heat":
                continue
            if model_name == "MLP" and task != "classification":
                continue
            if model_name in ("RandomForest", "GradientBoosting", "XGBoost") and task == "pinn":
                continue
            pairs.append((ds_name, ds, model_name))

    total = len(pairs) * len(tuners) * len(seeds)
    done = 0
    t_start = time.time()

    for ds_name, ds, model_name in pairs:
        for tuner_name in tuners:
            for seed in seeds:
                done += 1
                if verbose:
                    elapsed = time.time() - t_start
                    print(
                        f"[{done}/{total}] {ds_name} | {model_name} | {tuner_name} | seed={seed} "
                        f"(elapsed: {elapsed/60:.1f}m)"
                    )
                summary, hist = run_one(
                    dataset_name=ds_name,
                    X=ds["X"],
                    y=ds["y"],
                    task_type=ds["task"],
                    model_name=model_name,
                    tuner_name=tuner_name,
                    seed=seed,
                    budget=budget,
                    cv=cv,
                    population_size=population_size,
                )
                summaries.append(summary)
                if not hist.empty:
                    histories.append(hist)

    summary_df = pd.DataFrame(summaries)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
        if verbose:
            print(f"Wrote results to {out_dir}/")

    return summary_df, history_df


def run_budget_sweep(
    budgets: List[int] = (20, 40, 60, 100),
    seeds: List[int] = (0, 1, 2),
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    tuners: Optional[List[str]] = None,
    cv: int = 3,
    population_size: int = 10,
    include_openml: bool = True,
    include_deep: bool = False,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full benchmark grid once for each budget in ``budgets``.

    Each row of the resulting summary has an extra ``budget`` column,
    which makes it possible to compare rank-versus-budget curves for
    each tuner. The function returns ``(summary_df, history_df)``,
    where the frames are concatenated across the budgets evaluated.
    """
    all_summary: List[pd.DataFrame] = []
    all_history: List[pd.DataFrame] = []
    for b in budgets:
        if verbose:
            print(f"\n{'='*60}\nBudget = {b}\n{'='*60}")
        bdir = None
        if out_dir is not None:
            bdir = os.path.join(out_dir, f"budget_{b}")

        # ----- Resume-from-checkpoint (two levels) -----
        # 1) If the entire budget is already complete with all requested tuners, reuse.
        # 2) If the budget exists with SOME tuners (e.g., old 7-tuner sweep on disk and
        #    we're adding HDGPSO-MF), run ONLY the missing tuners and merge into the
        #    existing CSV. Saves 75%+ of compute when adding a new tuner.
        bs_path = os.path.join(bdir, "summary.csv") if bdir is not None else None
        bh_path = os.path.join(bdir, "history.csv") if bdir is not None else None
        cached_s = None
        cached_h = None
        if bs_path is not None and os.path.exists(bs_path):
            try:
                cached_s = pd.read_csv(bs_path)
                if bh_path and os.path.exists(bh_path):
                    cached_h = pd.read_csv(bh_path)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [checkpoint] could not read {bs_path}: {exc}; running fresh")
                cached_s = None
                cached_h = None

        requested_tuners = set(tuners) if tuners is not None else set(ALL_TUNERS.keys())
        if cached_s is not None:
            cached_tuners = set(cached_s["tuner"].unique())
            if requested_tuners.issubset(cached_tuners):
                if verbose:
                    print(f"  [checkpoint] budget={b} complete; reusing {len(cached_s)} cells")
                full_s = cached_s.copy(); full_s["budget"] = b
                full_h = cached_h.copy() if cached_h is not None else pd.DataFrame()
                if not full_h.empty:
                    full_h["budget"] = b
                all_summary.append(full_s)
                if not full_h.empty:
                    all_history.append(full_h)
                continue
            missing_tuners = sorted(requested_tuners - cached_tuners)
            if verbose:
                print(f"  [checkpoint] budget={b} exists with {sorted(cached_tuners)}; "
                      f"adding missing {missing_tuners}")
        else:
            missing_tuners = sorted(requested_tuners)

        # Run only missing tuners
        s, h = run_benchmark(
            budget=b,
            seeds=seeds,
            datasets=datasets,
            models=models,
            tuners=missing_tuners,
            cv=cv,
            population_size=population_size,
            include_openml=include_openml,
            include_deep=include_deep,
            out_dir=None,   # Write merged result after combining; avoid clobbering existing CSVs
            verbose=verbose,
        )

        # Merge new results with cached data and persist
        if cached_s is not None:
            s = pd.concat([cached_s, s], ignore_index=True)
            if cached_h is not None and not cached_h.empty and not h.empty:
                h = pd.concat([cached_h, h], ignore_index=True)
            elif cached_h is not None and not cached_h.empty:
                h = cached_h
        if bdir is not None:
            os.makedirs(bdir, exist_ok=True)
            s.to_csv(bs_path, index=False)
            if not h.empty:
                h.to_csv(bh_path, index=False)
            if verbose:
                print(f"  [checkpoint] wrote merged budget={b} summary "
                      f"({len(s)} cells, tuners={sorted(s['tuner'].unique())})")
        s = s.copy()
        s["budget"] = b
        if not h.empty:
            h = h.copy()
            h["budget"] = b
        all_summary.append(s)
        if not h.empty:
            all_history.append(h)

    summary = pd.concat(all_summary, ignore_index=True)
    history = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(os.path.join(out_dir, "summary_budget_sweep.csv"), index=False)
        history.to_csv(os.path.join(out_dir, "history_budget_sweep.csv"), index=False)
        if verbose:
            print(f"\nWrote budget-sweep results to {out_dir}/")

    return summary, history


def aggregate(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the summary across seeds and return mean and std of best_loss per (dataset, model, tuner)."""
    agg = (
        summary_df.groupby(["dataset", "model", "tuner"])["best_loss"]
        .agg(["mean", "std", "min", "count"])
        .reset_index()
    )
    return agg


def rank_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank tuners by best_loss within each (dataset, model, seed) cell and average.

    The result is a DataFrame containing the mean rank per tuner
    across all cells. Lower values indicate better performance.
    """
    df = summary_df.copy()
    df["rank"] = df.groupby(["dataset", "model", "seed"])["best_loss"].rank(method="min")
    return (
        df.groupby("tuner")["rank"]
        .agg(["mean", "std", "count"])
        .sort_values("mean")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=60, help="objective evaluations per run")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds")
    p.add_argument("--cv", type=int, default=3, help="CV folds in objective")
    p.add_argument("--pop", type=int, default=10, help="population size for swarm/DE methods")
    p.add_argument("--out", type=str, default="results", help="output directory")
    p.add_argument("--no-openml", action="store_true", help="skip OpenML datasets")
    p.add_argument("--datasets", type=str, default="", help="comma-sep dataset subset")
    p.add_argument("--models", type=str, default="", help="comma-sep model subset")
    p.add_argument("--tuners", type=str, default="", help="comma-sep tuner subset")
    args = p.parse_args()

    def _csv(s):
        return [x.strip() for x in s.split(",") if x.strip()] or None

    summary, history = run_benchmark(
        budget=args.budget,
        seeds=list(range(args.seeds)),
        cv=args.cv,
        population_size=args.pop,
        out_dir=args.out,
        include_openml=not args.no_openml,
        datasets=_csv(args.datasets),
        models=_csv(args.models),
        tuners=_csv(args.tuners),
    )
    print("\n--- Mean rank (lower better) ---")
    print(rank_table(summary).to_string(index=False))
    print("\n--- Aggregate per (dataset, model) ---")
    print(aggregate(summary).to_string(index=False))


if __name__ == "__main__":
    _main()
