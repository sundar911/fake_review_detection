from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import collate_hierarchical
from .evaluation import Metrics, aggregate_batches, compute_metrics


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 5.0
    device: torch.device = field(default_factory=default_device)


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> tuple[float, Metrics]:
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    metric_batches: list[Metrics] = []

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        word_mask = batch["word_mask"].to(device)
        sentence_mask = batch["sentence_mask"].to(device)
        labels = batch["labels"].to(device)

        model_inputs = {
            "tokens": tokens,
            "word_mask": word_mask,
            "sentence_mask": sentence_mask,
        }
        with torch.set_grad_enabled(training):
            outputs = model(model_inputs)
            logits = outputs["logits"]
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


def build_dataloader(dataset: Dataset, config: TrainingConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_hierarchical,
    )


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainingConfig,
) -> Dict[str, list]:
    model = model.to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
    }
    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)

    for _ in range(config.epochs):
        train_loss, train_metrics = _run_epoch(
            model, train_loader, criterion, config.device, optimizer, grad_clip=config.grad_clip
        )
        val_loss, val_metrics = _run_epoch(model, val_loader, criterion, config.device, optimizer=None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)
    return history


def evaluate_model(model: nn.Module, dataset: Dataset, config: TrainingConfig) -> tuple[float, Metrics]:
    loader = build_dataloader(dataset, config, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    return _run_epoch(model.to(config.device), loader, criterion, config.device, optimizer=None)
