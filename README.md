# hdgpso

[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python implementation of HDGPSO, a hybrid method for hyperparameter
optimization. It combines Differential Evolution, Grey Wolf Optimization,
and Particle Swarm Optimization in one sequential search process, and
uses a RandomForest surrogate to filter candidates before they are sent
for expensive model training.

The idea behind the method is that no single optimizer is best for every
problem. DE explores the search space, GWO moves the population toward
good regions using the current top three candidates, and PSO refines
solutions with memory of past good positions. Running all three in
sequence each iteration gives the search a different behavior than any
of them alone.

On the benchmark used in the paper (9 valid *(dataset, model)* pairs,
7 tuners, 3 seeds, 60-evaluation budget), HDGPSO reaches a mean rank
of 2.63, ahead of Optuna-TPE at 2.85 and Bayesian Optimization at 2.89,
and wins outright on every Gradient-Boosted-tree cell.

## Install

```bash
pip install git+https://github.com/ashuxen/hdgpso.git
```

From source:

```bash
git clone https://github.com/ashuxen/hdgpso.git
cd hdgpso
pip install -e ".[stats]"                # adds matplotlib for plotting
pip install -e ".[benchmarks,deep,dev]"  # full reproduction stack
```

Optional extras:

| Extra | Adds | When to use it |
|-------|------|----------------|
| `[stats]` | matplotlib | `hdgpso.stats.cd_diagram` and `hdgpso.plots.*` |
| `[benchmarks]` | scikit-optimize, optuna, pyswarms, xgboost | the full 7-tuner reproduction benchmark |
| `[deep]` | torch | MLP and PINN-Heat objectives |
| `[dev]` | pytest, ruff | running tests during development |

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

In normal use you only need to provide the search space, the objective
function, and the population/iteration budget. The remaining algorithm
settings come from the published literature:

- DE: `F = 0.8`, `CR = 0.5` (within the Storn–Price recommended range)
- PSO: `c1 = c2 = 2.0` (classical Kennedy–Eberhart setting)
- Inertia: `w` decays linearly from `0.7` to `0.4` (tighter than the
  textbook `0.9 → 0.4` because DE and GWO already cover the exploration
  side)
- RandomForest surrogate: refits every 4 iterations once the trial
  history has at least 12 points (about twice the population size)

All of these can be overridden through the constructor. See
`help(HDGPSO)` for the full list.

## Algorithm

Each iteration of HDGPSO runs three stages on the same population:

1. **Differential Evolution** — for every candidate, pick three other
   members at random, form a donor vector via the DE/rand/1 mutation,
   apply binary crossover, and accept the trial only if it improves the
   loss.
2. **Grey Wolf leadership** — sort the population by current loss and
   label the top three as α, β, δ. Every other member is updated toward
   a weighted blend of these three leaders, using the canonical
   Mirjalili (2014) formula.
3. **Particle Swarm refinement** — each particle updates its velocity
   from its personal best and the swarm's global best with a linearly
   decreasing inertia weight, then moves.

Between stages, a RandomForest surrogate trained on past trials is
used to score proposed candidates and prefer ones with the best
tree-variance lower confidence bound. The surrogate is only an
inexpensive filter, never a substitute for the real objective.

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
│   ├── tuners.py               # uniform adapters for the baseline tuners
│   ├── deep_objectives.py      # MLP + PINN-Heat objectives (torch)
│   ├── run_claim_check_v*.py   # main paper run at b=60
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

# Budget sensitivity sweep over {20, 40, 60, 100}
python run_budget_sweep.py
```

Results are written into `results_*/` (gitignored).

## Statistical analysis (Demšar 2006)

The `hdgpso.stats` module implements the standard rank-based protocol
for comparing several tuners across several cells:

```python
import pandas as pd
from hdgpso.stats import (
    friedman_test, nemenyi_matrix, cd_diagram,
    hdgpso_vs_baselines_table, bootstrap_rank_ci, cliffs_delta,
)

summary = pd.read_csv("results/summary.csv")
print(friedman_test(summary))           # global rejection check
print(bootstrap_rank_ci(summary))       # 95% CI per tuner
print(hdgpso_vs_baselines_table(summary, target="HDGPSO"))
cd_diagram(summary, save_path="fig_cd.png", title="My benchmark")
```

## Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Experimental: HDGPSOMF

The package also ships an experimental multi-fidelity variant called
`HDGPSOMF` that wraps HDGPSO with BOHB-style successive halving over
fidelity units. It is not part of the published headline results: on
the benchmark mix we tested it did not show a consistent advantage,
mainly because the sklearn models are already inexpensive at full
fidelity and the low-fidelity proxies introduced enough noise to flip
candidate rankings. It is more likely to be useful in settings where
fidelity (for example, epoch count) is exact and informative and where
each full-fidelity evaluation is genuinely expensive.

If you want to try it, see `examples/02_multifidelity.py` and
`help(hdgpso.HDGPSOMF)`. The API is not yet stable.

## References

- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7.
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Adv. Eng. Software 69.
- Storn, R. & Price, K. (1997). *Differential Evolution.* J. Global Optimization 11.
- Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization.*

## Citation

See [CITATION.cff](CITATION.cff). The current BibTeX entry is:

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
