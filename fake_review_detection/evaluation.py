from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Metrics:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(labels.dtype)

    true_positive = torch.sum((preds == 1) & (labels == 1)).item()
    true_negative = torch.sum((preds == 0) & (labels == 0)).item()
    false_positive = torch.sum((preds == 1) & (labels == 0)).item()
    false_negative = torch.sum((preds == 0) & (labels == 1)).item()

    accuracy = (true_positive + true_negative) / max(labels.numel(), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def aggregate_batches(metric_batches: Iterable[Metrics]) -> Metrics:
    total = Metrics(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0)
    count = 0
    for metric in metric_batches:
        total.accuracy += metric.accuracy
        total.precision += metric.precision
        total.recall += metric.recall
        total.f1 += metric.f1
        count += 1
    if count == 0:
        return total
    return Metrics(
        accuracy=total.accuracy / count,
        precision=total.precision / count,
        recall=total.recall / count,
        f1=total.f1 / count,
    )
