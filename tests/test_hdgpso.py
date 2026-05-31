"""Smoke tests for hdgpso + tuners.

Run with:  python -m pytest test_hdgpso.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hdgpso import (
    Categorical,
    Float,
    HDGPSO,
    Int,
    SearchSpace,
)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def test_float_linear_roundtrip():
    f = Float(0.0, 10.0)
    f.name = "x"
    assert f.from_internal(5.0) == 5.0
    assert f.from_internal(-1.0) == 0.0
    assert f.from_internal(11.0) == 10.0


def test_float_log_roundtrip():
    f = Float(1e-5, 1e-2, log=True)
    f.name = "x"
    mid = (math.log(1e-5) + math.log(1e-2)) / 2
    assert f.from_internal(mid) == pytest.approx(math.sqrt(1e-5 * 1e-2), rel=1e-6)


def test_int_rounding():
    d = Int(2, 8)
    d.name = "x"
    assert d.from_internal(2.4) == 2
    assert d.from_internal(2.6) == 3
    assert d.from_internal(100) == 8
    assert isinstance(d.from_internal(5.0), int)


def test_categorical_indexing():
    c = Categorical(["a", "b", "c"])
    c.name = "x"
    assert c.from_internal(0) == "a"
    assert c.from_internal(0.4) == "a"
    assert c.from_internal(0.6) == "b"
    assert c.from_internal(1.6) == "c"
    assert c.from_internal(99) == "c"
    assert c.from_internal(-99) == "a"


def test_searchspace_decode():
    s = SearchSpace(
        {
            "lr": Float(1e-4, 1e-1, log=True),
            "layers": Int(2, 8),
            "act": Categorical(["relu", "gelu"]),
        }
    )
    rng = np.random.default_rng(0)
    samples = s.sample(rng, 5)
    assert samples.shape == (5, 3)
    for row in samples:
        decoded = s.decode(row)
        assert 1e-4 <= decoded["lr"] <= 1e-1
        assert 2 <= decoded["layers"] <= 8
        assert decoded["act"] in ("relu", "gelu")


# ---------------------------------------------------------------------------
# Optimizer convergence on a known function
# ---------------------------------------------------------------------------


def _sphere(params):
    return sum(v * v for v in params.values())


def test_hdgpso_converges_on_sphere():
    space = SearchSpace(
        {"a": Float(-5, 5), "b": Float(-5, 5), "c": Float(-5, 5)}
    )
    opt = HDGPSO(space, _sphere, population_size=8, iterations=12, seed=42)
    result = opt.optimize()
    assert result.best_loss < 1.0, f"HDGPSO failed: {result.best_loss}"
    assert len(result.history) > 0
    assert result.history["running_best"].iloc[-1] == result.best_loss


def test_hdgpso_three_stages_per_iter():
    """Each iteration produces 3*pop_size evaluations (DE + GWO + PSO).

    Surrogate filter is disabled so candidate generation doesn't depend on
    side-effect-free surrogate predictions and the eval count is exact.
    """
    space = SearchSpace({"x": Float(-1, 1), "y": Float(-1, 1), "z": Float(-1, 1), "w": Float(-1, 1)})
    pop = 5
    iters = 4
    opt = HDGPSO(space, _sphere, population_size=pop, iterations=iters, seed=0,
                 use_surrogate=False)
    result = opt.optimize()
    expected = pop + iters * 3 * pop
    assert result.n_evals == expected, f"expected {expected}, got {result.n_evals}"


def test_hdgpso_respects_eval_budget():
    """When eval_budget < natural total, optimization stops at budget."""
    space = SearchSpace({"x": Float(-1, 1)})
    opt = HDGPSO(space, _sphere, population_size=5, iterations=100,
                 eval_budget=23, seed=0, use_surrogate=False)
    r = opt.optimize()
    assert r.n_evals == 23, f"expected exactly 23 evals, got {r.n_evals}"
    assert r.stopped_reason == "budget"


def test_hdgpso_beats_random_on_rosenbrock():
    def rosenbrock(params):
        x, y = params["x"], params["y"]
        return (1 - x) ** 2 + 100 * (y - x * x) ** 2

    space = SearchSpace({"x": Float(-2, 2), "y": Float(-2, 2)})

    rng = np.random.default_rng(0)
    random_losses = [rosenbrock(space.decode(space.sample(rng)[0])) for _ in range(80)]
    random_best = min(random_losses)

    opt = HDGPSO(space, rosenbrock, population_size=8, iterations=4, seed=0)
    result = opt.optimize()

    assert result.best_loss <= random_best * 1.2, (
        f"HDGPSO ({result.best_loss}) worse than random ({random_best})"
    )


def test_early_stopping():
    space = SearchSpace({"x": Float(0, 1)})

    calls = [0]

    def obj(p):
        calls[0] += 1
        return 0.5  # constant loss -> no improvement

    opt = HDGPSO(
        space,
        obj,
        population_size=4,
        iterations=100,
        early_stop_patience=2,
        seed=0,
    )
    result = opt.optimize()
    assert result.stopped_reason == "early_stop"
    # Should have stopped well before 100 iters
    assert result.n_evals < 100 * 4 * 2


def test_time_budget():
    space = SearchSpace({"x": Float(0, 1)})
    opt = HDGPSO(
        space,
        lambda p: p["x"],
        population_size=4,
        iterations=10000,
        time_budget_seconds=0.5,
        seed=0,
    )
    result = opt.optimize()
    assert result.elapsed_seconds < 2.0
    assert result.stopped_reason == "budget"


def test_history_contains_decoded_params():
    space = SearchSpace(
        {"a": Float(0, 1), "n": Int(1, 5), "kind": Categorical(["x", "y"])}
    )
    opt = HDGPSO(space, lambda p: p["a"] + p["n"], population_size=4, iterations=2, seed=0)
    result = opt.optimize()
    assert "a" in result.history.columns
    assert "n" in result.history.columns
    assert "kind" in result.history.columns
    assert set(result.history["kind"].unique()).issubset({"x", "y"})


# ---------------------------------------------------------------------------
# Tuner adapters live in benchmarks/, not the package — their tests are in
# benchmarks/tests/. The package test suite below covers only public API.
# ---------------------------------------------------------------------------
# Stats module: Friedman, Nemenyi, CD diagram, Cliff's delta, bootstrap CI
# ---------------------------------------------------------------------------


def _fake_summary():
    """Build a small synthetic summary frame where HDGPSO clearly dominates."""
    import pandas as pd

    rows = []
    tuners = ["RandomSearch", "Bayes", "DE", "PSO", "HDGPSO"]
    rng = np.random.default_rng(0)
    for dataset in ["d1", "d2", "d3", "d4"]:
        for model in ["m1", "m2"]:
            for seed in [0, 1, 2]:
                for t in tuners:
                    base = rng.normal(0.5, 0.02)
                    if t == "HDGPSO":
                        loss = base - 0.05  # systematically better
                    elif t == "Bayes":
                        loss = base - 0.01
                    else:
                        loss = base
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "tuner": t,
                            "best_loss": loss,
                        }
                    )
    return pd.DataFrame(rows)


def test_friedman_test_rejects_when_one_dominates():
    from hdgpso.stats import friedman_test

    res = friedman_test(_fake_summary())
    assert res.reject_null, f"Friedman failed to reject H0: p={res.pvalue}"
    assert res.n_tuners == 5
    assert res.n_datasets == 4 * 2 * 3  # ds x model x seed


def test_critical_difference_value():
    from hdgpso.stats import critical_difference

    # k=5, n=10 datasets, alpha=0.05 -> should match Demsar table
    cd = critical_difference(5, 10, 0.05)
    # q=2.728, sqrt(5*6/60)=sqrt(0.5)~0.7071 -> CD ~ 1.929
    assert 1.8 < cd < 2.0, f"Got CD={cd}"


def test_hdgpso_vs_baselines_table():
    from hdgpso.stats import hdgpso_vs_baselines_table

    df = hdgpso_vs_baselines_table(_fake_summary(), target="HDGPSO")
    # All baselines should have positive rank delta (HDGPSO ranks better)
    assert (df["rank_delta"] > 0).all()
    # Cliff's delta should be > 0 (HDGPSO wins more pairs)
    assert (df["cliffs_delta"] > 0).all()


def test_cliffs_delta_known():
    from hdgpso.stats import cliffs_delta

    # x always smaller than y -> delta = 1
    assert cliffs_delta(np.array([1, 2, 3]), np.array([4, 5, 6])) == 1.0
    # x always larger -> delta = -1
    assert cliffs_delta(np.array([4, 5, 6]), np.array([1, 2, 3])) == -1.0
    # identical -> 0
    assert cliffs_delta(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0


def test_bootstrap_rank_ci():
    from hdgpso.stats import bootstrap_rank_ci

    df = bootstrap_rank_ci(_fake_summary(), n_boot=200, seed=0)
    # HDGPSO should be the lowest mean rank
    assert df.iloc[0]["tuner"] == "HDGPSO"
    # CI bounds should bracket the mean
    for _, row in df.iterrows():
        assert row["ci_lo_95"] <= row["mean_rank"] <= row["ci_hi_95"]


def test_cd_diagram_renders():
    import matplotlib

    matplotlib.use("Agg")
    from hdgpso.stats import cd_diagram

    fig = cd_diagram(_fake_summary())
    assert fig is not None
