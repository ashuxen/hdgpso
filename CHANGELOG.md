# Changelog

This file lists notable changes between releases.

## Version 0.1.0 (May 2026)

This is the first public release of the package.

The release contains the implementation of HDGPSO, the three-stage hybrid hyperparameter optimizer described in the paper. Each iteration applies Differential Evolution, Grey Wolf Optimization, and Particle Swarm Optimization to the same population, in that order. A small RandomForest surrogate is fit on the trial history every few iterations and is used to filter proposed candidates between the operator stages before the real objective is invoked. On the 189-cell benchmark used in the paper, this version reaches a mean rank of 2.63 and wins outright on every Gradient-Boosted-tree cell tested.

The release also exposes the search space primitives `Float`, `Int`, and `Categorical`, combined into a `SearchSpace`. Floats can be log-scaled, and any combination of these types is accepted. Optimization results are returned in an `OptimizeResult` object that holds the best parameters, the best loss, the full trial history as a pandas DataFrame, the elapsed wall-clock time, and the reason the run stopped. Reproducibility is provided through a seedable random number generator: for a fixed seed and a deterministic objective, the optimization trajectory is the same on every run.

Statistical helpers for paper-style comparisons are available in `hdgpso.stats`. These include the Friedman omnibus test, the Nemenyi post-hoc test, Critical Difference diagrams, Cliff's delta effect size, and bootstrap confidence intervals on per-tuner mean rank. Together they implement the Demšar (2006) protocol used to compare several tuners across several cells.

A benchmark harness is also included under `benchmarks/`. The harness defines uniform adapters for GridSearch, RandomSearch, scikit-optimize Bayesian Optimization, Optuna-TPE, scipy Differential Evolution, pyswarms Particle Swarm, and HDGPSO. The main benchmark script runs all seven tuners across nine valid (dataset, model) pairs and three random seeds at the standard 60-evaluation budget. A separate sweep script is included for budget sensitivity at {20, 40, 60, 100} evaluations.

In addition to HDGPSO, the package ships an experimental multi-fidelity variant called `HDGPSOMF`. This variant wraps HDGPSO with BOHB-style successive halving and uses a surrogate trained only on full-fidelity points. The variant is not part of the published headline results, and on the benchmark mix used in this study it did not show a consistent advantage. Its API is not yet stable and is likely to change before a 1.0 release.

A note on default parameters: the DE coefficients are `F = 0.8` and `CR = 0.5`, which are inside the Storn–Price recommended range. The PSO coefficients `c1 = c2 = 2.0` are the classical Kennedy–Eberhart setting. The inertia weight decays linearly from 0.7 to 0.4. This is slightly tighter than the standard 0.9 to 0.4 schedule. The reason for using a tighter schedule is that the DE and GWO stages already supply enough exploration, and PSO is used here mainly for refinement. The minimum supported population size is 4, because the DE/rand/1 rule requires three distinct population members other than the target.
