# hdgpso

[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hybrid Differential Evolution + Grey Wolf Optimizer + Particle Swarm
Optimization** — a three-stage metaheuristic for hyperparameter
optimization, augmented by a RandomForest surrogate filter using
tree-variance-based lower-confidence-bound acquisition.

On a 189-cell benchmark (9 compatible *(dataset, model)* pairs × 7
tuners × 3 seeds at the standard 60-evaluation budget), HDGPSO achieves
the lowest mean rank (2.63), narrowly beating Optuna-TPE (2.85) and
Bayesian Optimization (2.89). It wins outright on every
Gradient-Boosted-tree cell tested.

## Install

```bash
pip install git+https://github.com/ashuxen/hdgpso.git
```

From source:

```bash
git clone https://github.com/ashuxen/hdgpso.git
cd hdgpso
pip install -e ".[stats]"                # 'stats' extra for Demsar plots
pip install -e ".[benchmarks,deep,dev]"  # full reproduction stack
```

Optional extras:

| Extra | Adds | When you need it |
|-------|------|------------------|
| `[stats]` | matplotlib | for `hdgpso.stats.cd_diagram` and `hdgpso.plots.*` |
| `[benchmarks]` | scikit-optimize, optuna, pyswarms, xgboost | to run the paper benchmark with all 7 baseline tuners |
| `[deep]` | torch | to run the MLP and PINN-Heat objectives |
| `[dev]` | pytest, ruff | for development / running tests |

## Quickstart

```python
from hdgpso import HDGPSO, SearchSpace, Float, Int, Categorical

space = SearchSpace({
    "n_estimators": Int(20, 300),
    "max_depth":    Int(2, 20),
    "max_features": Categorical(["sqrt", "log2", 0.5, 1.0]),
})

def objective(params):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    model = RandomForestClassifier(**params, random_state=0, n_jobs=1)
    return -cross_val_score(model, X, y, cv=3).mean()   # lower is better

result = HDGPSO(space, objective, population_size=10,
                iterations=15, seed=0).optimize()
print(result.best_params)
print(f"Best loss: {result.best_loss:.4f}")
print(result.history.head())
```

A user of HDGPSO sets only the search space, the objective, and the
budget pair `(population_size, iterations)`. All other knobs ship with
published defaults (`F=0.8, CR=0.5` per Storn–Price; `c1=c2=2.0` per
Kennedy–Eberhart; inertia `w: 0.7→0.4`; RandomForest surrogate refits
every 4 iterations once history reaches ~2× the population) and are
exposed via the constructor — see `help(HDGPSO)` for the full reference.

## Algorithm

A population of candidates undergoes three operator stages per
iteration:

1. **Differential Evolution** — DE/rand/1 mutation with binary crossover
   and greedy selection generates trial candidates;
2. **Grey Wolf leadership** — the top-three of the population (α, β, δ)
   guide every member's position via the canonical Mirjalili (2014)
   update;
3. **Particle Swarm refinement** — each particle's personal-best and the
   swarm's global-best drive a momentum update with linearly decreasing
   inertia.

A RandomForest surrogate is fit periodically on the trial history and
used to filter candidate proposals between stages, adding principled
exploitation while DE / GWO / PSO supply complementary exploration.

## Repository layout

```
hdgpso/
├── src/hdgpso/                 # installable Python package
│   ├── __init__.py             # public API
│   ├── core.py                 # HDGPSO + SearchSpace types
│   ├── stats.py                # Friedman / Nemenyi / CD / bootstrap
│   └── plots.py                # convergence / rank-bar / wins
├── benchmarks/                 # research reproduction scripts
│   ├── benchmark.py            # main run_benchmark() driver
│   ├── tuners.py               # uniform adapters for baseline tuners
│   ├── deep_objectives.py      # MLP + PINN-Heat objectives (torch)
│   ├── run_claim_check_v*.py   # per-iteration claim-check runs
│   └── run_budget_sweep.py     # 4-budget sweep
├── tests/test_hdgpso.py        # unit tests
└── examples/                   # standalone usage examples
```

## Reproducing the paper benchmark

```bash
pip install -e ".[benchmarks,deep]"
cd benchmarks

# Main 7-tuner comparison at budget=60
python run_claim_check_v5.py

# Budget sensitivity sweep at budgets {20, 40, 60, 100}
python run_budget_sweep.py
```

Outputs land in `results_*/` directories (gitignored).

## Statistical analysis (Demšar 2006)

`hdgpso.stats` implements the standard methodology for comparing K
algorithms over N cells:

```python
import pandas as pd
from hdgpso.stats import (
    friedman_test, nemenyi_matrix, cd_diagram,
    hdgpso_vs_baselines_table, bootstrap_rank_ci, cliffs_delta,
)

summary = pd.read_csv("results/summary.csv")
print(friedman_test(summary))           # omnibus rejection
print(bootstrap_rank_ci(summary))       # 95% CI per tuner
print(hdgpso_vs_baselines_table(summary, target="HDGPSO"))
cd_diagram(summary, save_path="fig_cd.png", title="My benchmark")
```

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Experimental: HDGPSOMF (multi-fidelity, future work)

The package also ships an in-development multi-fidelity variant,
`HDGPSOMF`, that wraps HDGPSO with BOHB-style successive halving over
fidelity-units. It is **not part of the published headline results** —
on the benchmark mix we evaluated it did not show a consistent
advantage over single-fidelity HDGPSO, because the sklearn models we
tested are already cheap at full fidelity and the low-fidelity proxies
introduced enough noise to flip candidate ranks. We expect it to be
more impactful on benchmarks where fidelity (e.g., epoch count) is
exact and informative and each full-fidelity evaluation is genuinely
expensive (large neural architecture search benchmarks, etc.).

If you want to experiment with it, see `examples/02_multifidelity.py`
and `help(hdgpso.HDGPSOMF)`. API stability is not guaranteed and the
implementation may change before a 1.0 release.

## References

- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7.
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Adv. Eng. Software 69.
- Storn, R. & Price, K. (1997). *Differential Evolution.* J. Global Optimization 11.
- Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization.*

## Citation

See [CITATION.cff](CITATION.cff). When citing, use:

```bibtex
@software{kumar2026hdgpso,
  author    = {Ashutosh Kumar},
  title     = {hdgpso: Hybrid DE-GWO-PSO hyperparameter optimization},
  year      = {2026},
  url       = {https://github.com/ashuxen/hdgpso},
  license   = {MIT}
}
```

## License

MIT. See [LICENSE](LICENSE).
