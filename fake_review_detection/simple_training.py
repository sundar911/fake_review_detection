from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluation import Metrics, aggregate_batches, compute_metrics


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SimpleTrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    device: torch.device = field(default_factory=default_device)


def train_simple_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    collate_fn,
    config: SimpleTrainingConfig,
) -> Dict[str, list]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
    }

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    model = model.to(config.device)

    for _ in range(config.epochs):
        train_loss, train_metrics = _run_epoch(model, train_loader, criterion, optimizer, config.device, config.grad_clip)
        val_loss, val_metrics = _run_epoch(model, val_loader, criterion, None, config.device, None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)
    return history


def evaluate_simple_model(
    model: nn.Module,
    dataset: Dataset,
    collate_fn,
    config: SimpleTrainingConfig,
) -> tuple[float, Metrics]:
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    return _run_epoch(model.to(config.device), loader, criterion, None, config.device, None)


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: Optional[float],
) -> tuple[float, Metrics]:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    metric_batches: list[Metrics] = []

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(tokens)
            loss = criterion(logits, labels)
        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item()
        metric_batches.append(compute_metrics(logits.detach(), labels.detach()))

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = aggregate_batches(metric_batches)
    return avg_loss, metrics
