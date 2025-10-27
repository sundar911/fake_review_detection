from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TextRNNConfig:
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.5
    bidirectional: bool = True
    num_classes: int = 1
    pad_index: int = 0


class TextBiLSTM(nn.Module):
    def __init__(self, config: TextRNNConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_index)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        lstm_output_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(lstm_output_dim, config.num_classes)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        outputs, _ = self.lstm(embedded)
        pooled = torch.mean(outputs, dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
