# hdgpso

[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hybrid Differential Evolution + Grey Wolf Optimizer + Particle Swarm
Optimization** — a 3-stage metaheuristic for hyperparameter optimization
with optional RandomForest surrogate, plus a **multi-fidelity** variant
using BOHB-style successive halving.

This repository contains two novel HPO algorithms and the full benchmark
harness used to evaluate them across 4 datasets × 4 model types × 7
baseline tuners × 3 seeds × 4 evaluation budgets.

| Algorithm | Use case | When to pick it |
|-----------|----------|-----------------|
| `HDGPSO` | Standard HPO (40-60 evaluation budget) | Tree-based / boosted classifiers; mixed continuous-discrete search spaces |
| `HDGPSOMF` | Budget-constrained HPO (≤40 evals or expensive objectives) | When each evaluation is slow or you can afford only a handful of full-fidelity runs |

## Install

```bash
pip install hdgpso
```

From source:

```bash
git clone https://github.com/ashuxen/hdgpso.git
cd hdgpso
pip install -e ".[stats]"            # add 'stats' extra for Demsar plots
pip install -e ".[benchmarks,deep,dev]"  # full reproduction stack
```

Optional extras:

| Extra | Adds | When you need it |
|-------|------|------------------|
| `[stats]` | matplotlib | for `hdgpso.stats.cd_diagram` and `hdgpso.plots.*` |
| `[benchmarks]` | scikit-optimize, optuna, pyswarms, xgboost | to run the full paper benchmark with all 7 baseline tuners |
| `[deep]` | torch | to run the MLP and PINN-Heat objectives |
| `[dev]` | pytest, ruff | for development / running tests |

## Quickstart

### HDGPSO (single-fidelity)

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

### HDGPSOMF (multi-fidelity)

The objective must accept a `fidelity` keyword in `[0, 1]`. Use it to
make the evaluation cheaper at low fidelity:

```python
from hdgpso import HDGPSOMF, SearchSpace, Float, Int

space = SearchSpace({
    "lr":          Float(1e-5, 1e-1, log=True),
    "hidden_dim":  Int(16, 256),
    "num_epochs":  Int(5, 50),
})

def objective(params, fidelity=1.0):
    # At fidelity=0.3, train for 30% of requested epochs:
    n_epochs = max(2, int(params["num_epochs"] * fidelity))
    model = build_my_model(**params)
    return train_and_validate(model, epochs=n_epochs)

# eval_budget is in fidelity-units (1 full eval = 1.0 unit;
# 1 low-fidelity eval at fidelity=0.3 = 0.3 units).
result = HDGPSOMF(space, objective, eval_budget=60,
                  low_fidelity=0.3, verify_fraction=0.4,
                  seed=0).optimize()
print(result.best_params, result.best_loss)
```

## Algorithms (one-paragraph description)

### HDGPSO

A population of candidates undergoes three operator stages per
iteration: **(1) Differential Evolution** — DE/rand/1 mutation with binary
crossover and greedy selection generates trial candidates;
**(2) Grey Wolf leadership** — the top-three of the population (α, β, δ)
guide every member's position via the canonical Mirjalili (2014) update;
**(3) Particle Swarm refinement** — each particle's personal-best and the
swarm's global-best drive a momentum update with linearly decreasing
inertia. A RandomForest surrogate is fit periodically on the trial
history and used to filter candidate proposals at each stage. The
surrogate adds principled exploitation while DE/GWO/PSO provide
complementary exploration.

### HDGPSOMF

Extends HDGPSO with **multi-fidelity successive halving** (BOHB-style):
candidate proposals are first evaluated at low fidelity (cheaper but
noisier proxy — e.g. fewer training epochs, smaller CV); only the top
fraction by low-fidelity loss are re-verified at full fidelity. Budget
is accounted in *fidelity-units* so a 0.3-fidelity probe costs 0.3 units
and a full-fidelity eval costs 1.0. The surrogate is trained on
**full-fidelity points only** (avoids noisy low-fidelity miscalibration).
For the same nominal budget, HDGPSOMF executes 2-3× more candidate
proposals than vanilla HDGPSO, dramatically improving exploration of
the search space when each evaluation is expensive.

## Repository layout

```
hdgpso/
├── src/hdgpso/                 # installable Python package
│   ├── __init__.py             # public API
│   ├── core.py                 # HDGPSO + SearchSpace types
│   ├── multifidelity.py        # HDGPSOMF
│   ├── stats.py                # Friedman / Nemenyi / CD / bootstrap
│   └── plots.py                # convergence / rank-bar / wins
├── benchmarks/                 # research reproduction scripts
│   ├── benchmark.py            # main run_benchmark() driver
│   ├── tuners.py               # uniform adapters for baseline tuners
│   ├── deep_objectives.py      # MLP + PINN-Heat objectives (torch)
│   ├── run_claim_check_v*.py   # per-iteration claim-check runs
│   └── run_budget_sweep.py     # 4-budget sweep
├── tests/test_hdgpso.py        # 20 pytest cases
├── examples/                   # standalone usage examples
└── paper/paper_draft.md        # writing companion with results
```

## Reproducing the paper benchmark

```bash
pip install -e ".[benchmarks,deep]"
cd benchmarks

# Main 8-tuner comparison at budget=60 (~3-4 hr on RTX GPU + CPU)
python run_claim_check_v5.py

# Budget sensitivity sweep at budgets {20, 40, 60, 100} (~12-16 hr)
python run_budget_sweep.py
```

Outputs land in `results_*/` directories (gitignored). The paper draft
in `paper/paper_draft.md` references the exact CSVs and figures.

## Statistical analysis (Demšar 2006)

`hdgpso.stats` implements the standard methodology for comparing N
algorithms over K cells:

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

## References

- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7.
- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Adv. Eng. Software 69.
- Storn, R. & Price, K. (1997). *Differential Evolution.* J. Global Optimization 11.
- Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization.*
- Falkner, S., Klein, A., & Hutter, F. (2018). *BOHB: Robust and Efficient Hyperparameter Optimization at Scale.* ICML.
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). *Hyperband.* JMLR 18.

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
