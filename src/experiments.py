"""MLP configurable, entrenamiento reproducible y evaluacion de experimentos."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ExperimentConfig:
    id: int
    name: str
    hidden_layers: tuple[int, ...] = (64, 32)
    activation: str = "ReLU"
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    dropout: float = 0.0
    batch_norm: bool = False
    l1_lambda: float = 0.0
    weight_decay: float = 0.0
    momentum: float = 0.0
    alpha: float = 0.99


class RegressionMLP(nn.Module):
    """Perceptron multicapa para una salida continua."""

    def __init__(self, input_dim: int, config: ExperimentConfig):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in config.hidden_layers:
            layers.append(nn.Linear(previous, width))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(make_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def make_activation(name: str) -> nn.Module:
    activations = {
        "ReLU": nn.ReLU,
        "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01),
        "Tanh": nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"Activacion no soportada: {name}")
    return activations[name]()


def make_optimizer(model: nn.Module, config: ExperimentConfig):
    common = {
        "params": model.parameters(),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if config.optimizer == "Adam":
        return torch.optim.Adam(**common)
    if config.optimizer == "SGD":
        return torch.optim.SGD(**common, momentum=config.momentum)
    if config.optimizer == "RMSprop":
        return torch.optim.RMSprop(**common, alpha=config.alpha)
    raise ValueError(f"Optimizador no soportado: {config.optimizer}")


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": math.sqrt(mse),
    }


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(X, dtype=torch.float32)
    return model(tensor).cpu().numpy()


def train_experiment(
    config: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> tuple[dict, dict, dict[str, torch.Tensor]]:
    """Entrena una configuracion y conserva su checkpoint de menor RMSE val."""

    set_deterministic(seed)
    model = RegressionMLP(X_train.shape[1], config)
    optimizer = make_optimizer(model, config)
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    history = {"train_mse": [], "train_objective": [], "val_mse": []}
    best_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        sum_mse = 0.0
        sum_objective = 0.0
        seen = 0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            mse_loss = criterion(outputs, targets)
            l1_penalty = torch.zeros((), dtype=torch.float32)
            if config.l1_lambda > 0:
                l1_penalty = sum(parameter.abs().sum() for parameter in model.parameters())
            objective = mse_loss + config.l1_lambda * l1_penalty
            objective.backward()
            optimizer.step()
            batch_n = len(features)
            seen += batch_n
            sum_mse += float(mse_loss.item()) * batch_n
            sum_objective += float(objective.item()) * batch_n

        train_mse = sum_mse / seen
        train_objective = sum_objective / seen
        val_predictions = predict(model, X_val)
        val_mse = float(mean_squared_error(y_val, val_predictions))
        val_rmse = math.sqrt(val_mse)
        history["train_mse"].append(train_mse)
        history["train_objective"].append(train_objective)
        history["val_mse"].append(val_mse)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    duration = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No se genero un checkpoint valido")
    model.load_state_dict(best_state)
    best_predictions = predict(model, X_val)
    metrics = regression_metrics(y_val, best_predictions)
    result = {
        **asdict(config),
        "architecture": "-".join(map(str, config.hidden_layers)),
        "regularization": regularization_label(config),
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": duration,
        "val_mse": metrics["mse"],
        "val_mae": metrics["mae"],
        "val_rmse": metrics["rmse"],
        "generalization_gap": history["val_mse"][best_epoch - 1]
        - history["train_mse"][best_epoch - 1],
    }
    return result, history, best_state


def regularization_label(config: ExperimentConfig) -> str:
    labels = []
    if config.l1_lambda:
        labels.append(f"L1={config.l1_lambda:g}")
    if config.weight_decay:
        labels.append(f"L2={config.weight_decay:g}")
    if config.dropout:
        labels.append(f"Dropout={config.dropout:g}")
    if config.batch_norm:
        labels.append("BatchNorm")
    return "+".join(labels) if labels else "Ninguna"


def experiment_catalog() -> list[ExperimentConfig]:
    """Catalogo ordenado: baseline y 16 variaciones controladas."""

    base = dict(hidden_layers=(64, 32), activation="ReLU", optimizer="Adam")
    return [
        ExperimentConfig(1, "Baseline", **base),
        ExperimentConfig(2, "Arquitectura pequena", hidden_layers=(32,)),
        ExperimentConfig(3, "Arquitectura profunda", hidden_layers=(128, 64, 32)),
        ExperimentConfig(4, "Activacion LeakyReLU", activation="LeakyReLU"),
        ExperimentConfig(5, "Activacion Tanh", activation="Tanh"),
        ExperimentConfig(6, "Optimizador SGD", optimizer="SGD", learning_rate=1e-2, momentum=0.9),
        ExperimentConfig(7, "Optimizador RMSprop", optimizer="RMSprop", alpha=0.99),
        ExperimentConfig(8, "Learning rate alto", learning_rate=1e-2),
        ExperimentConfig(9, "Learning rate bajo", learning_rate=1e-4),
        ExperimentConfig(10, "Batch pequeno", batch_size=32),
        ExperimentConfig(11, "Batch grande", batch_size=256),
        ExperimentConfig(12, "Menos epochs", epochs=50),
        ExperimentConfig(13, "Mas epochs", epochs=150),
        ExperimentConfig(14, "Regularizacion L1", l1_lambda=1e-5),
        ExperimentConfig(15, "Regularizacion L2", weight_decay=1e-4),
        ExperimentConfig(16, "Dropout", dropout=0.2),
        ExperimentConfig(17, "Batch normalization", batch_norm=True),
    ]


def run_experiments(
    configs: Iterable[ExperimentConfig],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict], dict[int, dict[str, torch.Tensor]]]:
    rows, histories, states = [], {}, {}
    for config in configs:
        print(f"[{config.id:02d}] {config.name}...", flush=True)
        result, history, state = train_experiment(config, X_train, y_train, X_val, y_val)
        rows.append(result)
        histories[str(config.id)] = history
        states[config.id] = state
        print(
            f"     epoch={result['best_epoch']} RMSE={result['val_rmse']:.4f} "
            f"MAE={result['val_mae']:.4f} tiempo={result['training_seconds']:.1f}s",
            flush=True,
        )
    results = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    return results, histories, states


def select_best(results: pd.DataFrame) -> pd.Series:
    return results.sort_values(
        ["val_rmse", "val_mae", "parameters", "training_seconds"],
        ascending=True,
    ).iloc[0]


def select_curve_ids(results: pd.DataFrame) -> list[int]:
    best = int(results.loc[results.val_rmse.idxmin(), "id"])
    worst = int(results.loc[results.val_rmse.idxmax(), "id"])
    gap = int(results.loc[results.generalization_gap.idxmax(), "id"])
    return list(dict.fromkeys([1, best, worst, gap]))


def plot_selected_curves(
    results: pd.DataFrame,
    histories: dict[str, dict],
    output: str | Path,
) -> list[int]:
    ids = select_curve_ids(results)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, experiment_id in zip(axes.flat, ids):
        history = histories[str(experiment_id)]
        row = results.loc[results.id == experiment_id].iloc[0]
        epochs = np.arange(1, len(history["train_mse"]) + 1)
        ax.plot(epochs, history["train_mse"], label="Train MSE", color="#2563eb")
        ax.plot(epochs, history["val_mse"], label="Validacion MSE", color="#dc2626")
        ax.axvline(row.best_epoch, color="#64748b", linestyle="--", linewidth=1)
        ax.set(title=f"E{experiment_id}: {row['name']}", xlabel="Epoch", ylabel="MSE")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(ids):]:
        ax.axis("off")
    fig.suptitle("Curvas de perdida seleccionadas", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    return ids


def retrain_and_test_once(
    config: ExperimentConfig,
    epochs: int,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 42,
) -> tuple[dict, StandardScaler, dict[str, torch.Tensor]]:
    """Reajusta con train+val y evalua test una sola vez."""

    set_deterministic(seed)
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    final_config = ExperimentConfig(**{**asdict(config), "epochs": int(epochs)})
    model = RegressionMLP(X_train_val_scaled.shape[1], final_config)
    optimizer = make_optimizer(model, final_config)
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(X_train_val_scaled, dtype=torch.float32),
        torch.as_tensor(y_train_val.to_numpy(dtype=np.float32), dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=final_config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    started = time.perf_counter()
    for _ in range(final_config.epochs):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, targets)
            if final_config.l1_lambda:
                loss = loss + final_config.l1_lambda * sum(
                    parameter.abs().sum() for parameter in model.parameters()
                )
            loss.backward()
            optimizer.step()
    predictions = predict(model, X_test_scaled)
    metrics = regression_metrics(y_test.to_numpy(), predictions)
    metrics.update(
        {
            "epochs": final_config.epochs,
            "training_seconds": time.perf_counter() - started,
            "mae_usd": metrics["mae"] * 100_000,
            "rmse_usd": metrics["rmse"] * 100_000,
        }
    )
    return metrics, scaler, copy.deepcopy(model.state_dict())


def save_artifacts(
    output_dir: str | Path,
    results: pd.DataFrame,
    histories: dict[str, dict],
    best: pd.Series,
    final_metrics: dict,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "experiments.csv", index=False)
    (output / "histories.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )
    summary = {
        "best_experiment": int(best["id"]),
        "best_name": str(best["name"]),
        "best_epoch": int(best["best_epoch"]),
        "validation": {
            "mse": float(best["val_mse"]),
            "mae": float(best["val_mae"]),
            "rmse": float(best["val_rmse"]),
        },
        "test": {key: float(value) for key, value in final_metrics.items()},
    }
    (output / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

