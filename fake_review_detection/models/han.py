from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HANConfig:
    vocab_size: int
    embedding_dim: int = 128
    word_hidden_dim: int = 64
    sentence_hidden_dim: int = 64
    embedding_dropout: float = 0.1
    word_dropout: float = 0.1
    sentence_dropout: float = 0.1
    num_classes: int = 1
    pad_index: int = 0


class Attention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        if mask.ndim == 2:
            invalid = ~mask.any(dim=1)
            if invalid.any():
                mask = mask.clone()
                mask[invalid, 0] = True
        scores = self.context_vector(torch.tanh(self.projection(inputs))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.sum(inputs * weights.unsqueeze(-1), dim=1)
        return attended, weights


class HAN(nn.Module):
    """
    Hierarchical Attention Network for document classification.
    """

    def __init__(self, config: HANConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_index,
        )
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        self.word_encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.word_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.word_dropout = nn.Dropout(config.word_dropout)
        self.word_attention = Attention(hidden_dim=config.word_hidden_dim * 2)

        self.sentence_encoder = nn.GRU(
            input_size=config.word_hidden_dim * 2,
            hidden_size=config.sentence_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.sentence_dropout = nn.Dropout(config.sentence_dropout)
        self.sentence_attention = Attention(hidden_dim=config.sentence_hidden_dim * 2)

        self.classifier = nn.Linear(config.sentence_hidden_dim * 2, config.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = batch["tokens"]
        word_mask = batch["word_mask"]
        sentence_mask = batch["sentence_mask"]

        batch_size, max_sentences, max_words = tokens.size()
        reshaped_tokens = tokens.view(batch_size * max_sentences, max_words)
        reshaped_word_mask = word_mask.view(batch_size * max_sentences, max_words)

        embedded = self.embedding(reshaped_tokens)
        embedded = self.embedding_dropout(embedded)

        word_outputs, _ = self.word_encoder(embedded)
        word_outputs = self.word_dropout(word_outputs)

        sentence_vectors, word_attentions = self.word_attention(
            word_outputs, reshaped_word_mask
        )
        sentence_vectors = sentence_vectors.view(batch_size, max_sentences, -1)
        word_attentions = word_attentions.view(batch_size, max_sentences, max_words)

        sentence_outputs, _ = self.sentence_encoder(sentence_vectors)
        sentence_outputs = self.sentence_dropout(sentence_outputs)

        document_vector, sentence_attentions = self.sentence_attention(
            sentence_outputs, sentence_mask
        )
        logits = self.classifier(document_vector).squeeze(-1)
        return {
            "logits": logits,
            "word_attentions": word_attentions,
            "sentence_attentions": sentence_attentions,
        }
