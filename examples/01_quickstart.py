"""Minimal HDGPSO example: tune a RandomForestClassifier on Breast Cancer.

Run:
    pip install hdgpso
    python 01_quickstart.py
"""
from hdgpso import HDGPSO, SearchSpace, Float, Int, Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

space = SearchSpace({
    "n_estimators": Int(20, 300),
    "max_depth": Int(2, 20),
    "min_samples_split": Int(2, 20),
    "min_samples_leaf": Int(1, 20),
    "max_features": Categorical(["sqrt", "log2", 0.5, 1.0]),
})


def objective(params):
    model = RandomForestClassifier(**params, random_state=0, n_jobs=1)
    return -cross_val_score(model, X, y, cv=3).mean()


print("Optimizing RandomForest on Breast Cancer with HDGPSO...")
result = HDGPSO(
    space=space,
    objective=objective,
    population_size=10,
    iterations=8,
    seed=0,
    verbose=True,
).optimize()

print()
print(f"Best parameters: {result.best_params}")
print(f"Best CV accuracy: {-result.best_loss:.4f}")
print(f"Total evaluations: {result.n_evals}")
print(f"Elapsed: {result.elapsed_seconds:.1f}s")
print()
print("Trial history (first 5 rows):")
print(result.history.head().to_string(index=False))
