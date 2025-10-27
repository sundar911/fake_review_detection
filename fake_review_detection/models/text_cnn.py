from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TextCNNConfig:
    vocab_size: int
    embedding_dim: int = 128
    num_filters: int = 100
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    dropout: float = 0.5
    num_classes: int = 1
    pad_index: int = 0


class TextCNN(nn.Module):
    def __init__(self, config: TextCNNConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_index)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.embedding_dim,
                    out_channels=config.num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in config.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.num_filters * len(config.kernel_sizes), config.num_classes)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor with shape [batch, sequence_length]
        Returns:
            logits: Tensor with shape [batch]
        """
        embedded = self.embedding(tokens)  # [batch, seq_len, embed_dim]
        embedded = embedded.transpose(1, 2)  # [batch, embed_dim, seq_len]
        feature_maps = []
        for conv in self.convs:
            activated = torch.relu(conv(embedded))
            pooled = torch.max(activated, dim=-1).values
            feature_maps.append(pooled)
        features = torch.cat(feature_maps, dim=1)
        features = self.dropout(features)
        logits = self.classifier(features).squeeze(-1)
        return logits
