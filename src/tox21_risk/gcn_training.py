"""Masked multi-task training and evaluation for the Tox21 GCN."""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tox21_risk.gcn_model import GCNConfig, Tox21GCN
from tox21_risk.lr_baseline import binary_probability_metrics


@dataclass(frozen=True)
class EvaluationResult:
    """Predictions and endpoint metrics from one deterministic model pass."""

    endpoint_metrics: pd.DataFrame
    labels: np.ndarray
    probabilities: np.ndarray
    mol_ids: list[str]
    macro_endpoint_bce: float
    macro_auroc: float
    macro_auprc: float
    macro_brier: float


@dataclass(frozen=True)
class FitResult:
    """Best restored model and its validation training trace."""

    model: Tox21GCN
    history: pd.DataFrame
    best_epoch: int
    best_validation_bce: float


def masked_bce_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """Average BCE over observed targets only."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    observed = ~torch.isnan(targets)
    if not torch.any(observed):
        raise ValueError("batch contains no observed labels")
    return F.binary_cross_entropy_with_logits(logits[observed], targets[observed])


def evaluate_model(
    model: Tox21GCN,
    loader: DataLoader,
    tasks: Sequence[str],
    device: torch.device,
) -> EvaluationResult:
    """Evaluate one checkpoint with dropout disabled."""

    model.eval()
    label_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    mol_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            label_parts.append(batch.y.detach().cpu().numpy())
            probability_parts.append(torch.sigmoid(logits).cpu().numpy())
            mol_ids.extend(str(mol_id) for mol_id in batch.mol_id)

    if not label_parts:
        raise ValueError("evaluation loader contains no graphs")
    labels = np.concatenate(label_parts, axis=0)
    probabilities = np.concatenate(probability_parts, axis=0)
    if labels.shape != probabilities.shape or labels.shape[1] != len(tasks):
        raise ValueError("evaluation predictions do not match endpoint order")

    metric_rows: list[dict[str, float | int | str]] = []
    for endpoint_index, endpoint in enumerate(tasks):
        endpoint_labels = labels[:, endpoint_index]
        endpoint_probabilities = probabilities[:, endpoint_index]
        observed = ~np.isnan(endpoint_labels)
        observed_labels = endpoint_labels[observed]
        observed_probabilities = endpoint_probabilities[observed]
        try:
            metrics = binary_probability_metrics(
                observed_labels, observed_probabilities
            )
        except ValueError as error:
            raise ValueError(f"{endpoint}: {error}") from error
        clipped = np.clip(observed_probabilities, 1e-7, 1.0 - 1e-7)
        endpoint_bce = float(
            -np.mean(
                observed_labels * np.log(clipped)
                + (1.0 - observed_labels) * np.log(1.0 - clipped)
            )
        )
        metric_rows.append(
            {"endpoint": endpoint, **metrics, "bce": endpoint_bce}
        )

    endpoint_metrics = pd.DataFrame(metric_rows)
    return EvaluationResult(
        endpoint_metrics=endpoint_metrics,
        labels=labels,
        probabilities=probabilities,
        mol_ids=mol_ids,
        macro_endpoint_bce=float(endpoint_metrics["bce"].mean()),
        macro_auroc=float(endpoint_metrics["auroc"].mean()),
        macro_auprc=float(endpoint_metrics["auprc"].mean()),
        macro_brier=float(endpoint_metrics["brier"].mean()),
    )


def fit_model(
    train_graphs: Sequence[Data],
    validation_graphs: Sequence[Data],
    tasks: Sequence[str],
    config: GCNConfig,
    seed: int,
    split_strategy: str,
    checkpoint_path: Path,
) -> FitResult:
    """Train one GCN and restore the validation-BCE-selected checkpoint."""

    if not train_graphs or not validation_graphs:
        raise ValueError("training and validation graph sets must be non-empty")
    set_reproducible_seed(seed)
    device = torch.device(config.device)
    input_width = int(train_graphs[0].num_node_features)
    model = Tox21GCN(
        input_width,
        len(tasks),
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        list(train_graphs),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        list(validation_graphs),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    best_validation_bce = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_observed = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            observed_count = int((~torch.isnan(batch.y)).sum().item())
            loss = masked_bce_with_logits(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * observed_count
            total_observed += observed_count
        if total_observed == 0:
            raise ValueError("training set contains no observed labels")

        validation = evaluate_model(model, validation_loader, tasks, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_observed,
                "validation_macro_bce": validation.macro_endpoint_bce,
                "validation_macro_auroc": validation.macro_auroc,
            }
        )
        if validation.macro_endpoint_bce < best_validation_bce:
            best_validation_bce = validation.macro_endpoint_bce
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                best_state,
                config,
                seed,
                split_strategy,
                input_width,
                tasks,
                best_epoch,
                best_validation_bce,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(
        model=model,
        history=pd.DataFrame(history_rows),
        best_epoch=best_epoch,
        best_validation_bce=best_validation_bce,
    )


def set_reproducible_seed(seed: int) -> None:
    """Seed every random source used by the CPU training workflow."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _save_checkpoint(
    path: Path,
    state: dict[str, Tensor],
    config: GCNConfig,
    seed: int,
    split_strategy: str,
    input_width: int,
    tasks: Sequence[str],
    best_epoch: int,
    best_validation_bce: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_values = asdict(config)
    config_values["seeds"] = list(config.seeds)
    torch.save(
        {
            "model_state_dict": state,
            "config": config_values,
            "seed": seed,
            "split_strategy": split_strategy,
            "input_width": input_width,
            "endpoint_order": list(tasks),
            "best_epoch": best_epoch,
            "best_validation_bce": best_validation_bce,
        },
        path,
    )
