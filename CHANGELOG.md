# Changelog

This file lists notable changes between releases.

## Version 0.1.0 (May 2026)

This is the first public release of the HDGPSO package.

The release includes the implementation of HDGPSO, the hybrid hyperparameter optimizer described in the paper. Each iteration updates the same population using Differential Evolution, Grey Wolf Optimization, and Particle Swarm Optimization in sequence. DE helps explore the search space, GWO guides the population toward better regions, and PSO refines candidates using personal-best and global-best memory. A small RandomForest surrogate is refit every few iterations and is used to screen candidates before the real objective function is evaluated.

The package supports mixed search spaces through `Float`, `Int`, and `Categorical` parameters, which can be combined into a `SearchSpace`. Float parameters may also be sampled on a log scale. Results are returned in an `OptimizeResult` object containing the best parameters, best loss, trial history as a pandas DataFrame, elapsed time, and stopping reason. Runs are reproducible when a fixed seed and deterministic objective are used.

This release also includes statistical helpers in `hdgpso.stats` for comparing optimizers, including Friedman testing, Nemenyi post-hoc analysis, Critical Difference diagrams, Cliff’s delta, and bootstrap confidence intervals. A benchmark harness is provided under `benchmarks/`, with adapters for GridSearch, RandomSearch, Bayesian Optimization, Optuna-TPE, scipy Differential Evolution, pyswarms Particle Swarm, and HDGPSO. The benchmark scripts reproduce the main 60-evaluation experiment and the budget sweep at 20, 40, 60, and 100 evaluations.

An experimental multi-fidelity variant, `HDGPSOMF`, is also included. It uses BOHB-style successive halving and trains the surrogate only on full-fidelity evaluations. This variant is provided for testing and is not part of the main headline results. Its API may change before a stable 1.0 release.

Default settings follow common values: `F = 0.8`, `CR = 0.5`, `c1 = c2 = 2.0`, and inertia decreasing from 0.7 to 0.4. The minimum population size is 4 because DE/rand/1 needs three other population members besides the target candidate.

