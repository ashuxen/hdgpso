"""Fidelity-aware objective factories for the HDGPSO-MF benchmark.

These are research-code factories specific to the paper's benchmark
problems (sklearn classification/regression, MLP on tabular,
PINN-Heat 1D). They are NOT part of the installable ``hdgpso``
package — users implementing their own multi-fidelity HPO should write
their own ``objective(params, fidelity)`` callable; see the docstring
of :class:`hdgpso.HDGPSOMF` for the API contract.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np


def make_fidelity_sklearn_objective(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    model_name: str,
    cv: int,
    seed: int,
    benchmark_make_objective,
    benchmark_build_model,
):
    """Wrap sklearn objective to accept fidelity ∈ [0.1, 1.0].

    Fidelity controls CV folds: at fidelity=1.0 we use the full `cv` folds,
    at fidelity<1.0 we drop to fewer folds (cv=1 at fidelity ≤ 0.5).
    For ensemble models, fidelity additionally scales n_estimators.
    """
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"

    def objective(params: Dict[str, Any], fidelity: float = 1.0) -> float:
        clean: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            else:
                clean[k] = v

        scaled = dict(clean)
        if model_name in ("RandomForest", "GradientBoosting", "XGBoost"):
            if "n_estimators" in scaled and fidelity < 0.99:
                scaled["n_estimators"] = max(10, int(scaled["n_estimators"] * fidelity))

        eff_cv = max(2, int(round(cv * fidelity))) if fidelity >= 0.6 else 2
        model = benchmark_build_model(model_name, task_type, scaled, seed)
        try:
            if fidelity < 0.5:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xs, y, test_size=0.25, random_state=seed,
                    stratify=y if task_type == "classification" else None,
                )
                model.fit(X_tr, y_tr)
                if task_type == "classification":
                    score = model.score(X_te, y_te)
                else:
                    from sklearn.metrics import mean_squared_error
                    score = -mean_squared_error(y_te, model.predict(X_te))
                return -float(score)
            scores = cross_val_score(model, Xs, y, cv=eff_cv, scoring=scoring, n_jobs=1)
        except Exception:
            return float("inf")
        return -float(np.mean(scores))

    return objective


def make_fidelity_mlp_objective(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Wrap MLP objective with fidelity ∈ [0.1, 1.0] scaling epochs + training-data."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from deep_objectives import (
        _MLP, _ACTIVATIONS, _make_optimizer, _make_scheduler, _DEVICE,
    )

    dev = _DEVICE
    n_classes = int(np.unique(y).size)
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    yi = y.astype(np.int64)
    X_tr_full, X_va, y_tr_full, y_va = train_test_split(
        Xs, yi, test_size=0.2, stratify=yi, random_state=seed,
    )
    X_va_t = torch.from_numpy(X_va).to(dev)
    y_va_t = torch.from_numpy(y_va).to(dev)
    in_dim = Xs.shape[1]
    rng_np = np.random.default_rng(seed)

    def objective(params: Dict[str, Any], fidelity: float = 1.0) -> float:
        torch.manual_seed(seed)
        try:
            n_train = max(50, int(X_tr_full.shape[0] * fidelity))
            idx = rng_np.choice(X_tr_full.shape[0], size=n_train, replace=False)
            X_tr_t = torch.from_numpy(X_tr_full[idx]).to(dev)
            y_tr_t = torch.from_numpy(y_tr_full[idx]).to(dev)

            act_cls = _ACTIVATIONS[params["activation"]]
            model = _MLP(
                in_dim, int(params["hidden_1"]), int(params["hidden_2"]),
                n_classes, act_cls, float(params["dropout_1"]), float(params["dropout_2"]),
            ).to(dev)
            opt = _make_optimizer(
                params["optimizer"], model.parameters(),
                float(params["lr"]), float(params["weight_decay"]),
                float(params["momentum"]),
            )
            epochs = max(1, int(int(params["epochs"]) * fidelity))
            sched = _make_scheduler(params["scheduler"], opt, epochs)
            bs = int(params["batch_size"])
            criterion = nn.CrossEntropyLoss()

            n = X_tr_t.shape[0]
            best_acc, no_improve = 0.0, 0
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, bs):
                    sel = perm[i:i + bs]
                    opt.zero_grad()
                    out = model(X_tr_t[sel])
                    criterion(out, y_tr_t[sel]).backward()
                    opt.step()
                if sched is not None:
                    sched.step()
                model.eval()
                with torch.no_grad():
                    acc = (model(X_va_t).argmax(dim=1) == y_va_t).float().mean().item()
                if acc > best_acc:
                    best_acc, no_improve = acc, 0
                else:
                    no_improve += 1
                    if no_improve >= 5:
                        break
            return -float(best_acc)
        except Exception:
            return float("inf")

    return objective


def make_fidelity_pinn_objective(seed: int = 0):
    """Wrap PINN objective with fidelity scaling epochs + collocation."""
    import torch
    from deep_objectives import (
        _MLPSinusoidal, _ACTIVATIONS, _make_optimizer, _make_scheduler,
        _DEVICE, _KAPPA, _T_MAX, _analytic_heat,
    )

    dev = _DEVICE
    g = np.linspace(0, 1, 40); tg = np.linspace(0, _T_MAX, 20)
    XX, TT = np.meshgrid(g, tg)
    X_val = torch.from_numpy(XX.flatten().astype(np.float32).reshape(-1, 1)).to(dev)
    T_val = torch.from_numpy(TT.flatten().astype(np.float32).reshape(-1, 1)).to(dev)
    U_val = _analytic_heat(X_val, T_val).detach()

    def objective(params: Dict[str, Any], fidelity: float = 1.0) -> float:
        torch.manual_seed(seed)
        try:
            act_cls = _ACTIVATIONS[params["activation"]]
            model = _MLPSinusoidal(int(params["hidden_dim"]), int(params["num_layers"]),
                                    act_cls).to(dev)
            opt = _make_optimizer(params["optimizer"], model.parameters(),
                                  float(params["lr"]), float(params["weight_decay"]))
            epochs = max(20, int(int(params["epochs"]) * fidelity))
            sched = _make_scheduler(params["scheduler"], opt, epochs)
            n_coll = max(100, int(int(params["n_collocation"]) * fidelity))
            pw = float(params["physics_weight"])
            bw = float(params["bc_weight"])

            rng = torch.Generator(device=dev).manual_seed(seed)
            x_int = torch.rand((n_coll, 1), generator=rng, device=dev).requires_grad_(True)
            t_int = torch.rand((n_coll, 1), generator=rng, device=dev).mul(_T_MAX).requires_grad_(True)
            x_ic = torch.linspace(0, 1, 80, device=dev).reshape(-1, 1)
            t_ic = torch.zeros_like(x_ic)
            u_ic_true = torch.sin(math.pi * x_ic)
            t_bc = torch.linspace(0, _T_MAX, 60, device=dev).reshape(-1, 1)
            x_bc_left = torch.zeros_like(t_bc); x_bc_right = torch.ones_like(t_bc)

            best_val, no_improve = float("inf"), 0
            for ep in range(epochs):
                model.train()
                opt.zero_grad()
                u = model(x_int, t_int)
                u_t = torch.autograd.grad(u, t_int, grad_outputs=torch.ones_like(u),
                                          create_graph=True, retain_graph=True)[0]
                u_x = torch.autograd.grad(u, x_int, grad_outputs=torch.ones_like(u),
                                          create_graph=True, retain_graph=True)[0]
                u_xx = torch.autograd.grad(u_x, x_int, grad_outputs=torch.ones_like(u_x),
                                            create_graph=True, retain_graph=True)[0]
                phys = ((u_t - _KAPPA * u_xx) ** 2).mean()
                ic = ((model(x_ic, t_ic) - u_ic_true) ** 2).mean()
                bc = (model(x_bc_left, t_bc) ** 2).mean() + (model(x_bc_right, t_bc) ** 2).mean()
                loss = pw * phys + ic + bw * bc
                if not torch.isfinite(loss):
                    return float("inf")
                loss.backward(); opt.step()
                if sched is not None: sched.step()
                if ep % max(epochs // 20, 1) == 0 or ep == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        val_mse = ((model(X_val, T_val) - U_val) ** 2).mean().item()
                    if val_mse < best_val:
                        best_val, no_improve = val_mse, 0
                    else:
                        no_improve += 1
                        if no_improve >= 50:
                            break
            return float(best_val)
        except Exception:
            return float("inf")

    return objective
