"""Model builders, validation-set tuning and fitting for all six models.

Every model is trained on the training subset only. The validation subset is
used for hyperparameter selection (all models) and early stopping (ANNs).
The test subset is never touched here.
"""
import copy
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from . import config as C


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def iter_grid(grid: dict, fast: bool = False):
    if fast:
        grid = {k: v[:1] for k, v in grid.items()}
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


@dataclass
class FitResult:
    name: str
    params: dict
    model: object
    train_time_s: float
    size: int
    size_unit: str
    val_rmse: float
    history: dict | None = field(default=None, repr=False)


# ---------------------------------------------------------------- sklearn
def _build_sklearn(name: str, params: dict, seed: int):
    if name == "LinearRegression":
        return LinearRegression(**params)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "SVR":
        return SVR(kernel="rbf", **params)
    raise KeyError(name)


def _sklearn_size(name: str, model):
    if name == "LinearRegression":
        return int(model.coef_.size + 1), "coefficients"
    if name == "RandomForest":
        return int(sum(t.tree_.node_count for t in model.estimators_)), "tree nodes"
    return int(model.support_.size), "support vectors"


def fit_sklearn(name, params, seed, data) -> FitResult:
    model = _build_sklearn(name, params, seed)
    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    elapsed = time.perf_counter() - t0
    size, unit = _sklearn_size(name, model)
    return FitResult(name, dict(params), model, elapsed, size, unit,
                     rmse(data.y_val, model.predict(data.X_val)))


# ---------------------------------------------------------------- PyTorch MLPs
class TorchMLP:
    """Wraps a trained torch network so it exposes predict() like sklearn."""

    def __init__(self, net):
        self.net = net

    def predict(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()


def build_mlp(n_features, hidden, dropout=0.0):
    """Dense ReLU hidden layers (+ optional dropout) and one linear output neuron."""
    from torch import nn

    layers, width = [], n_features
    for units in hidden:
        layers += [nn.Linear(width, units), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        width = units
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def l2_penalty(net, l2):
    """l2 * sum(W^2) over hidden-layer weight matrices (biases and output layer excluded)."""
    from torch import nn

    if not l2:
        return 0.0
    linears = [m for m in net if isinstance(m, nn.Linear)][:-1]
    return l2 * sum((m.weight ** 2).sum() for m in linears)


def fit_mlp(name, params, seed, data, fast=False) -> FitResult:
    """Adam + MSE, mini-batches, early stopping on validation loss with best-weight restore."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.set_num_threads(1)          # tiny network: single thread is faster and deterministic
    max_epochs = C.FAST_MAX_EPOCHS if fast else C.MAX_EPOCHS
    patience = C.FAST_PATIENCE if fast else C.PATIENCE
    l2, bs = params.get("l2", 0.0), params["batch_size"]

    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)
    X_tr, y_tr = f32(data.X_train), f32(data.y_train).unsqueeze(1)
    X_va, y_va = f32(data.X_val), f32(data.y_val).unsqueeze(1)

    net = build_mlp(X_tr.shape[1], C.MLP_ARCHS[name], params.get("dropout", 0.0))
    opt = torch.optim.Adam(net.parameters(), lr=params["lr"])
    gen = torch.Generator().manual_seed(seed)

    history = {"loss": [], "val_loss": [], "rmse": [], "val_rmse": []}
    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.perf_counter()
    for _ in range(max_epochs):
        net.train()
        perm, sse = torch.randperm(len(X_tr), generator=gen), 0.0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i + bs]
            mse = F.mse_loss(net(X_tr[idx]), y_tr[idx])
            loss = mse + l2_penalty(net, l2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sse += mse.item() * len(idx)
        train_mse = sse / len(X_tr)   # running training-mode MSE (dropout active)

        net.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(net(X_va), y_va).item()
            pen = float(l2_penalty(net, l2))
        val_loss = val_mse + pen      # monitored quantity includes the L2 term
        history["loss"].append(train_mse + pen)
        history["val_loss"].append(val_loss)
        history["rmse"].append(train_mse ** 0.5)
        history["val_rmse"].append(val_mse ** 0.5)

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    elapsed = time.perf_counter() - t0

    net.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
    wrapped = TorchMLP(net)
    n_params = sum(p.numel() for p in net.parameters())
    return FitResult(name, dict(params), wrapped, elapsed, int(n_params), "weights",
                     rmse(data.y_val, wrapped.predict(data.X_val)), history)


# ---------------------------------------------------------------- public API
def fit(name, params, seed, data, fast=False) -> FitResult:
    if name in C.MLP_NAMES:
        return fit_mlp(name, params, seed, data, fast)
    return fit_sklearn(name, params, seed, data)


def search_space(name, base_params=None, fast=False):
    if name == "LinearRegression":
        return [{}]
    if name == "RandomForest":
        return list(iter_grid(C.RF_GRID, fast))
    if name == "SVR":
        return list(iter_grid(C.SVR_GRID, fast))
    if name in ("MLP-A", "MLP-B"):
        return list(iter_grid(C.ANN_GRID, fast))
    if name == "MLP-C":
        if base_params is None:
            raise ValueError("MLP-C needs MLP-B's best lr/batch_size as base_params")
        base = {k: base_params[k] for k in ("lr", "batch_size")}
        return [{**base, **reg} for reg in iter_grid(C.REG_GRID, fast)]
    raise KeyError(name)


def tune(name, seed, data, base_params=None, fast=False, log=print):
    """Fit every candidate on train, keep the one with the lowest validation RMSE."""
    best, table = None, []
    candidates = search_space(name, base_params, fast)
    for i, params in enumerate(candidates, 1):
        res = fit(name, params, seed, data, fast)
        row = {**params, "val_rmse": res.val_rmse, "train_time_s": res.train_time_s}
        if res.history:
            row["epochs_run"] = len(res.history["loss"])
            row["best_epoch"] = res.history["best_epoch"]
        table.append(row)
        if name in C.MLP_NAMES:
            log(f"    [{i}/{len(candidates)}] {params} -> val RMSE {res.val_rmse:.3f}")
        if best is None or res.val_rmse < best.val_rmse:
            best = res
    return best, table
