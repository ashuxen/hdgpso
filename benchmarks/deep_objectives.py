"""Deep-network and PINN objectives used in the HDGPSO benchmark.

Each factory in this module returns an ``objective(params) -> float``
callable that the tuner can invoke directly. Both objectives are
designed to take about 2 to 5 seconds per evaluation on a GPU.

The search spaces here are intentionally high-dimensional and
mixed-type, with more than 10 parameters spanning floats, integers,
and categoricals. This is the regime in which surrogate-based methods
such as Bayesian Optimization and TPE tend to struggle, and in which
metaheuristics such as HDGPSO are expected to be competitive.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hdgpso import Categorical, Float, Int, SearchSpace


# Auto-select device once at module load
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Activation / optimizer factories shared by both objectives
# ---------------------------------------------------------------------------


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def _make_optimizer(name: str, params, lr: float, weight_decay: float, momentum: float = 0.9):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def _make_scheduler(name: str, opt, epochs: int):
    name = name.lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    if name == "step":
        return optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.5)
    if name == "none":
        return None
    raise ValueError(f"unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# MLP for tabular / image-tabular classification
# ---------------------------------------------------------------------------


def _mlp_space() -> SearchSpace:
    """12 hyperparameters: 6 floats (3 log), 5 ints, 3 categoricals."""
    return SearchSpace(
        {
            "lr": Float(1e-5, 1e-1, log=True),
            "weight_decay": Float(1e-6, 1e-2, log=True),
            "batch_size": Int(16, 256, log=True),
            "hidden_1": Int(32, 512),
            "hidden_2": Int(16, 256),
            "dropout_1": Float(0.0, 0.6),
            "dropout_2": Float(0.0, 0.6),
            "activation": Categorical(["relu", "gelu", "tanh", "elu"]),
            "optimizer": Categorical(["adam", "adamw", "sgd", "rmsprop"]),
            "scheduler": Categorical(["cosine", "step", "none"]),
            "momentum": Float(0.0, 0.99),
            "epochs": Int(5, 30),
        }
    )


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_1, hidden_2, out_dim, act_cls, dropout_1, dropout_2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            act_cls(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_1, hidden_2),
            act_cls(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_mlp_objective(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    device: Optional[str] = None,
    val_frac: float = 0.2,
) -> Callable[[Dict[str, Any]], float]:
    """Returns objective(params) -> -val_accuracy for an MLP on (X, y)."""
    dev = torch.device(device) if device else _DEVICE
    n_classes = int(np.unique(y).size)
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    yi = y.astype(np.int64)
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xs, yi, test_size=val_frac, stratify=yi, random_state=seed
    )
    X_tr_t = torch.from_numpy(X_tr).to(dev)
    y_tr_t = torch.from_numpy(y_tr).to(dev)
    X_va_t = torch.from_numpy(X_va).to(dev)
    y_va_t = torch.from_numpy(y_va).to(dev)
    in_dim = Xs.shape[1]

    def objective(params: Dict[str, Any]) -> float:
        torch.manual_seed(seed)
        try:
            act_cls = _ACTIVATIONS[params["activation"]]
            model = _MLP(
                in_dim,
                int(params["hidden_1"]),
                int(params["hidden_2"]),
                n_classes,
                act_cls,
                float(params["dropout_1"]),
                float(params["dropout_2"]),
            ).to(dev)
            opt = _make_optimizer(
                params["optimizer"],
                model.parameters(),
                float(params["lr"]),
                float(params["weight_decay"]),
                float(params["momentum"]),
            )
            epochs = int(params["epochs"])
            sched = _make_scheduler(params["scheduler"], opt, epochs)
            bs = int(params["batch_size"])
            criterion = nn.CrossEntropyLoss()

            n = X_tr_t.shape[0]
            best_acc = 0.0
            patience = 5
            no_improve = 0
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, bs):
                    idx = perm[i : i + bs]
                    opt.zero_grad()
                    out = model(X_tr_t[idx])
                    loss = criterion(out, y_tr_t[idx])
                    loss.backward()
                    opt.step()
                if sched is not None:
                    sched.step()

                model.eval()
                with torch.no_grad():
                    pred = model(X_va_t).argmax(dim=1)
                    acc = (pred == y_va_t).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            return -float(best_acc)
        except Exception:
            return float("inf")

    return objective


# ---------------------------------------------------------------------------
# PINN: 1D heat equation
#   u_t = kappa * u_xx          on  x in [0, 1], t in [0, T]
#   u(x, 0) = sin(pi * x)        initial condition
#   u(0, t) = u(1, t) = 0         boundary condition
#   Analytic: u(x, t) = exp(-kappa * pi^2 * t) * sin(pi * x)
# ---------------------------------------------------------------------------


_KAPPA = 0.1
_T_MAX = 1.0


def _pinn_space() -> SearchSpace:
    """11 hyperparameters covering arch, optimizer, training, physics weighting."""
    return SearchSpace(
        {
            "lr": Float(1e-5, 1e-1, log=True),
            "weight_decay": Float(1e-7, 1e-3, log=True),
            "hidden_dim": Int(16, 128),
            "num_layers": Int(2, 6),
            "activation": Categorical(["tanh", "gelu", "elu"]),
            "optimizer": Categorical(["adam", "adamw", "rmsprop"]),
            "scheduler": Categorical(["cosine", "step", "none"]),
            "physics_weight": Float(1e-2, 1e2, log=True),
            "bc_weight": Float(1e-1, 1e1, log=True),
            "n_collocation": Int(500, 2500),
            "epochs": Int(100, 500),
        }
    )


class _MLPSinusoidal(nn.Module):
    def __init__(self, hidden_dim, num_layers, act_cls):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), act_cls()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def _analytic_heat(x, t, kappa=_KAPPA):
    return torch.exp(-kappa * math.pi ** 2 * t) * torch.sin(math.pi * x)


def make_pinn_heat_objective(
    seed: int = 0, device: Optional[str] = None
) -> Callable[[Dict[str, Any]], float]:
    """Returns objective(params) -> validation MSE between PINN and analytic
    solution of the 1D heat equation. Lower is better."""
    dev = torch.device(device) if device else _DEVICE

    # Pre-sample a deterministic validation grid (same for all evaluations)
    g = np.linspace(0, 1, 40)
    tg = np.linspace(0, _T_MAX, 20)
    XX, TT = np.meshgrid(g, tg)
    X_val = torch.from_numpy(XX.flatten().astype(np.float32).reshape(-1, 1)).to(dev)
    T_val = torch.from_numpy(TT.flatten().astype(np.float32).reshape(-1, 1)).to(dev)
    U_val = _analytic_heat(X_val, T_val).detach()

    def objective(params: Dict[str, Any]) -> float:
        torch.manual_seed(seed)
        try:
            act_cls = _ACTIVATIONS[params["activation"]]
            model = _MLPSinusoidal(
                int(params["hidden_dim"]),
                int(params["num_layers"]),
                act_cls,
            ).to(dev)
            opt = _make_optimizer(
                params["optimizer"],
                model.parameters(),
                float(params["lr"]),
                float(params["weight_decay"]),
            )
            epochs = int(params["epochs"])
            sched = _make_scheduler(params["scheduler"], opt, epochs)
            n_coll = int(params["n_collocation"])
            pw = float(params["physics_weight"])
            bw = float(params["bc_weight"])

            # Interior collocation
            rng = torch.Generator(device=dev).manual_seed(seed)
            x_int = torch.rand((n_coll, 1), generator=rng, device=dev).requires_grad_(True)
            t_int = torch.rand((n_coll, 1), generator=rng, device=dev).mul(_T_MAX).requires_grad_(True)

            # IC: u(x, 0) = sin(pi x)
            x_ic = torch.linspace(0, 1, 80, device=dev).reshape(-1, 1)
            t_ic = torch.zeros_like(x_ic)
            u_ic_true = torch.sin(math.pi * x_ic)

            # BC: u(0, t) = u(1, t) = 0
            t_bc = torch.linspace(0, _T_MAX, 60, device=dev).reshape(-1, 1)
            x_bc_left = torch.zeros_like(t_bc)
            x_bc_right = torch.ones_like(t_bc)

            best_val = float("inf")
            patience = 50
            no_improve = 0
            for ep in range(epochs):
                model.train()
                opt.zero_grad()

                # Physics residual
                u = model(x_int, t_int)
                u_t = torch.autograd.grad(
                    u, t_int, grad_outputs=torch.ones_like(u),
                    create_graph=True, retain_graph=True
                )[0]
                u_x = torch.autograd.grad(
                    u, x_int, grad_outputs=torch.ones_like(u),
                    create_graph=True, retain_graph=True
                )[0]
                u_xx = torch.autograd.grad(
                    u_x, x_int, grad_outputs=torch.ones_like(u_x),
                    create_graph=True, retain_graph=True
                )[0]
                res = u_t - _KAPPA * u_xx
                phys = (res ** 2).mean()

                # IC + BC
                ic = ((model(x_ic, t_ic) - u_ic_true) ** 2).mean()
                bc = ((model(x_bc_left, t_bc)) ** 2).mean() + ((model(x_bc_right, t_bc)) ** 2).mean()

                loss = pw * phys + ic + bw * bc
                if not torch.isfinite(loss):
                    return float("inf")
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()

                if ep % max(epochs // 20, 1) == 0 or ep == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        u_pred = model(X_val, T_val)
                        val_mse = ((u_pred - U_val) ** 2).mean().item()
                    if val_mse < best_val:
                        best_val = val_mse
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break
            return float(best_val)
        except Exception:
            return float("inf")

    return objective


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


DEEP_MODEL_SPACES = {
    "MLP": _mlp_space,
    "PINN-Heat": _pinn_space,
}


__all__ = [
    "make_mlp_objective",
    "make_pinn_heat_objective",
    "DEEP_MODEL_SPACES",
]
