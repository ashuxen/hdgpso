# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05

### Added
- Initial public release.
- `HDGPSO`: single-fidelity three-stage hybrid metaheuristic (DE + GWO + PSO)
  with RandomForest surrogate-assisted candidate filtering.
- `HDGPSOMF`: multi-fidelity extension with BOHB-style successive halving
  and fidelity-aware surrogate (trained on full-fidelity points only).
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
  HDGPSO, HDGPSO-MF; 4-dataset / 4-model / 3-seed / 8-tuner benchmark
  + budget-sensitivity sweep ({20, 40, 60, 100} evaluations).

### Notes
- Algorithm hyperparameters tuned via grid meta-search: F=0.8, CR=0.5,
  c1=c2=2.0, w_max=0.7, w_min=0.4.
- Population size >= 4 required (DE mutation requires three distinct
  individuals other than the target).
