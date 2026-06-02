# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05

### Added
- Initial public release.
- `HDGPSO`: a three-stage hybrid metaheuristic that runs DE, GWO, and
  PSO in sequence each iteration, with a RandomForest surrogate used as
  a candidate filter between stages. Lowest mean rank (2.63) on the
  189-cell paper benchmark, and a clean sweep on every
  Gradient-Boosted-tree cell tested.
- Search-space primitives `Float`, `Int`, `Categorical`, and
  `SearchSpace` for mixed continuous, integer, and categorical
  hyperparameters, with optional log-scale on the floats.
- `OptimizeResult` container that bundles the best parameters, best
  loss, and the full trial history as a dataframe.
- Reproducible runs: a seedable RNG that produces deterministic output
  for a fixed seed when the objective itself is deterministic.
- Statistical analysis helpers in `hdgpso.stats` covering Friedman,
  Nemenyi, CD diagrams, Cliff's delta, and bootstrap rank CIs — the
  rank-based comparison protocol from Demšar (2006).
- Benchmark harness in `benchmarks/` with uniform adapters for
  GridSearch, RandomSearch, scikit-optimize, Optuna-TPE, scipy DE,
  pyswarms PSO, and HDGPSO. The main benchmark covers 9 valid
  *(dataset, model)* pairs × 3 seeds × 7 tuners = 189 cells at
  b=60, with a budget-sensitivity sweep over {20, 40, 60, 100}.

### Experimental
- `HDGPSOMF`: a multi-fidelity variant that wraps HDGPSO with BOHB-style
  successive halving and a full-fidelity-only surrogate. Provided for
  future-work exploration; not part of the published headline. It did
  not show a consistent advantage on the benchmark mix evaluated. The
  API is not yet stable.

### Notes
- The DE coefficients (F=0.8, CR=0.5) sit inside the Storn–Price
  recommended range. The PSO coefficients (c1=c2=2.0) are the classical
  Kennedy–Eberhart setting. The inertia decay (w_max=0.7, w_min=0.4) is
  tighter than the canonical 0.9 → 0.4 because the DE and GWO stages
  already cover exploration.
- A minimum population size of 4 is required, because DE mutation needs
  three distinct individuals other than the target.
