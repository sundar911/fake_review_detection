from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .han import HAN, HANConfig


@dataclass
class MultiHeadHANConfig(HANConfig):
    word_heads: int = 4
    word_head_dim: Optional[int] = None
    sentence_heads: int = 4
    sentence_head_dim: Optional[int] = None
    attention_dropout: float = 0.1


class MultiHeadAggregator(nn.Module):
    """
    Multi-head additive attention with learned global queries per head.
    """

    def __init__(self, hidden_dim: int, heads: int, head_dim: Optional[int], dropout: float) -> None:
        super().__init__()
        head_dim = head_dim or hidden_dim // heads
        if head_dim * heads != hidden_dim:
            raise ValueError("hidden_dim must be divisible by heads when head_dim is None.")
        self.heads = heads
        self.head_dim = head_dim
        total_dim = heads * head_dim
        self.key = nn.Linear(hidden_dim, total_dim, bias=False)
        self.value = nn.Linear(hidden_dim, total_dim, bias=False)
        self.query = nn.Parameter(torch.randn(heads, head_dim))
        self.out = nn.Linear(total_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, hidden_dim = inputs.shape
        mask = mask.bool()
        invalid = ~mask.any(dim=1)
        if invalid.any():
            mask = mask.clone()
            mask[invalid, 0] = True
        key = self.key(inputs).view(batch_size, time_steps, self.heads, self.head_dim)
        value = self.value(inputs).view(batch_size, time_steps, self.heads, self.head_dim)

        query = self.query.view(1, self.heads, self.head_dim).expand(batch_size, -1, -1)
        scores = torch.einsum("bthd,bhd->bth", key, query) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        attention = self.dropout(attention)
        context = torch.einsum("bth,bthd->bhd", attention, value).reshape(batch_size, -1)
        output = self.out(context)
        mean_attention = attention.mean(dim=-1)
        return output, mean_attention


class MultiHeadHAN(HAN):
    """
    Multi-head variant of the HAN model that replaces the additive attention
    layers with a multi-head scaled dot-product aggregator.
    """

    def __init__(self, config: MultiHeadHANConfig) -> None:
        base_config = HANConfig(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            word_hidden_dim=config.word_hidden_dim,
            sentence_hidden_dim=config.sentence_hidden_dim,
            embedding_dropout=config.embedding_dropout,
            word_dropout=config.word_dropout,
            sentence_dropout=config.sentence_dropout,
            num_classes=config.num_classes,
            pad_index=config.pad_index,
        )
        super().__init__(base_config)
        self.word_attention = MultiHeadAggregator(
            hidden_dim=config.word_hidden_dim * 2,
            heads=config.word_heads,
            head_dim=config.word_head_dim,
            dropout=config.attention_dropout,
        )
        self.sentence_attention = MultiHeadAggregator(
            hidden_dim=config.sentence_hidden_dim * 2,
            heads=config.sentence_heads,
            head_dim=config.sentence_head_dim,
            dropout=config.attention_dropout,
        )
