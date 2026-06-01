# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05

### Added
- Initial public release.
- `HDGPSO`: three-stage hybrid metaheuristic (DE + GWO + PSO) with
  RandomForest surrogate-assisted candidate filtering. Lowest mean rank
  (2.63) on the 189-cell paper benchmark; wins outright on every
  Gradient-Boosted-tree cell tested.
- Search-space primitives: `Float`, `Int`, `Categorical`, `SearchSpace`
  supporting mixed continuous / discrete / categorical hyperparameters
  with optional log-scale.
- `OptimizeResult` container with full trial history dataframe.
- Reproducibility: seedable RNG; deterministic output for fixed seed +
  deterministic objective.
- Statistical analysis helpers in `hdgpso.stats`: Friedman test, Nemenyi
  post-hoc, Critical Difference (CD) diagram, Cliff's delta, bootstrap
  rank CI — implementing the Demsar (2006) methodology for multi-tuner
  multi-dataset comparisons.
- Benchmark harness in `benchmarks/`: uniform adapters for GridSearch,
  RandomSearch, scikit-optimize, Optuna-TPE, scipy DE, pyswarms PSO,
  and HDGPSO; 9 *(dataset, model)* pairs × 3 seeds × 7 tuners = 189
  main cells at b=60, plus a budget-sensitivity sweep over
  {20, 40, 60, 100}.

### Experimental
- `HDGPSOMF`: multi-fidelity variant on top of HDGPSO with BOHB-style
  successive halving and a full-fidelity-only surrogate. Provided for
  future-work exploration; not part of the published headline. Did not
  show a consistent advantage on the benchmark mix evaluated. API may
  change before a 1.0 release.

### Notes
- Algorithm hyperparameters use published literature defaults: F=0.8,
  CR=0.5 (Storn–Price recommended range); c1=c2=2.0 (Kennedy–Eberhart);
  w_max=0.7, w_min=0.4 (tighter than canonical 0.9→0.4 because DE+GWO
  already supply exploration).
- Population size >= 4 required (DE mutation requires three distinct
  individuals other than the target).
