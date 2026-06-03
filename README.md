# hdgpso

[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the Python implementation of HDGPSO, a hybrid method for hyperparameter optimization. The method combines Differential Evolution, Grey Wolf Optimization, and Particle Swarm Optimization in one sequential search process. A lightweight RandomForest surrogate is also used to filter candidate solutions before expensive model training is performed.

The motivation for the design is simple. Each of the three optimizers has a useful behavior, but each one also has limitations when used alone. DE helps explore the search space. GWO moves the population toward good candidate regions. PSO refines solutions using memory of past good positions. By applying them together in a single iteration, the method tries to keep the strengths of each component and reduce the impact of their individual weaknesses.

On the benchmark used in the paper, HDGPSO achieves the lowest mean rank at the standard 60-evaluation budget. The benchmark contains nine valid (dataset, model) pairs across four datasets and four model classes, evaluated under seven tuners with three random seeds. HDGPSO obtains a mean rank of 2.63, compared to 2.85 for Optuna-TPE and 2.89 for Bayesian Optimization. On the GradientBoosting cells specifically, HDGPSO wins every tested cell.

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

| Extra | Adds | When to use |
|-------|------|-------------|
| `[stats]` | matplotlib | `hdgpso.stats.cd_diagram` and `hdgpso.plots.*` |
| `[benchmarks]` | scikit-optimize, optuna, pyswarms, xgboost | the full 7-tuner benchmark |
| `[deep]` | torch | MLP and PINN-Heat objectives |
| `[dev]` | pytest, ruff | running the tests |

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
```

In typical use, the caller only specifies the search space, the objective function, and the population and iteration budget. The remaining parameters use values taken directly from the original papers:

- For DE: `F = 0.8` and `CR = 0.5`, which sit inside the Storn–Price recommended range.
- For PSO: `c1 = c2 = 2.0`, which is the classical Kennedy–Eberhart setting.
- The inertia weight decays linearly from 0.7 to 0.4. This is slightly tighter than the canonical 0.9 to 0.4, because the preceding DE and GWO stages already supply enough exploration and PSO is used here mainly for refinement.
- The RandomForest surrogate refits every 4 iterations once at least 12 trial points are available.

All of these are exposed as constructor arguments. The complete list is available through `help(HDGPSO)`.

## What the algorithm does

Each iteration of HDGPSO runs three operator stages on the same population.

First, Differential Evolution. For each candidate, three other population members are chosen at random and combined into a donor vector using the DE/rand/1 rule. Binary crossover is then applied between the donor and the current candidate, and the trial replaces the candidate only when it improves the loss.

Second, Grey Wolf Optimizer. The population is sorted by current loss, and the top three members are labeled α, β, and δ. Every other member is updated using a weighted blend of these three leaders, following the standard Mirjalili (2014) update.

Third, Particle Swarm Optimization. Each particle maintains its personal best position, and the swarm keeps a global best. The velocity is updated using both terms, scaled by a linearly decaying inertia weight, and the particle moves accordingly.

Between the stages, a RandomForest surrogate is trained on the trial history. The surrogate is used only to filter proposed candidates by predicting their mean and tree-variance and selecting those with the best lower-confidence-bound score. It does not replace any real objective evaluation. Its only role is to screen out clearly weak proposals before they are passed to the expensive objective.

## Repository layout

```
hdgpso/
├── src/hdgpso/                 # installable package
│   ├── __init__.py             # public API
│   ├── core.py                 # HDGPSO + SearchSpace types
│   ├── stats.py                # Friedman / Nemenyi / CD / bootstrap
│   └── plots.py                # convergence / rank-bar / wins
├── benchmarks/                 # paper reproduction scripts
│   ├── benchmark.py            # main run_benchmark() driver
│   ├── tuners.py               # uniform adapters for the baseline tuners
│   ├── deep_objectives.py      # MLP + PINN-Heat objectives (torch)
│   ├── run_claim_check_v*.py   # main paper run at b = 60
│   └── run_budget_sweep.py     # budget sensitivity sweep
├── tests/test_hdgpso.py        # unit tests
└── examples/                   # standalone usage examples
```

## Reproducing the paper benchmark

```bash
pip install -e ".[benchmarks,deep]"
cd benchmarks

# Main 7-tuner comparison at budget = 60
python run_claim_check_v5.py

# Budget sensitivity sweep over {20, 40, 60, 100}
python run_budget_sweep.py
```

Results are written into `results_*/` directories, which are gitignored.

## Statistical analysis (Demšar 2006)

The `hdgpso.stats` module implements the rank-based protocol used in the paper for comparing several tuners across several cells.

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

The package also includes an experimental multi-fidelity variant, called HDGPSOMF, which wraps HDGPSO with BOHB-style successive halving over fidelity units. This variant is not part of the published headline results. On the benchmark mix used in this study it did not show a consistent advantage. The main reason is that the sklearn models are already inexpensive to train at full fidelity, and the low-fidelity proxies introduced enough noise to flip candidate rankings.

The variant may be more useful in settings where fidelity (for example, the number of training epochs) is exact and informative, and where each full-fidelity evaluation is genuinely expensive. The interface is exposed in `examples/02_multifidelity.py` and `help(hdgpso.HDGPSOMF)`. The API is not yet stable.

## References

- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7.
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Adv. Eng. Software 69.
- Storn, R. & Price, K. (1997). *Differential Evolution.* J. Global Optimization 11.
- Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization.*

## Citation

See [CITATION.cff](CITATION.cff). BibTeX entry:

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
