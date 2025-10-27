import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
TOKEN_REGEX = re.compile(r"\w+|[^\w\s]")


def read_reviews(csv_path: Path) -> Tuple[List[str], List[int]]:
    """
    Load review texts and labels from a CSV file.

    The expected format is a header row with at least two columns named
    ``review_text`` and ``label``.
    """
    texts: List[str] = []
    labels: List[int] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "review_text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'review_text' and 'label' columns.")
        for row in reader:
            texts.append(row["review_text"].strip())
            labels.append(int(float(row["label"])))
    return texts, labels


class Vocabulary:
    """Minimal vocabulary wrapper for token to index lookups."""

    def __init__(self, min_freq: int = 1, max_size: Optional[int] = None) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.token_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_token = [self.pad_token, self.unk_token]

    def build(self, texts: Sequence[str]) -> None:
        counter = Counter()
        for text in texts:
            for token in tokenize(text):
                counter[token] += 1
        items = [
            token
            for token, freq in counter.most_common()
            if freq >= self.min_freq
        ]
        if self.max_size:
            items = items[: self.max_size]
        for token in items:
            self.add_token(token)

    def add_token(self, token: str) -> None:
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

    def token_to_index(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def __len__(self) -> int:
        return len(self.idx_to_token)


def tokenize(text: str) -> List[str]:
    sentences = SENTENCE_SPLIT_REGEX.split(text.strip().lower())
    output: List[str] = []
    for sent in sentences:
        output.extend(TOKEN_REGEX.findall(sent))
    return output


def split_dataset(
    texts: Sequence[str],
    labels: Sequence[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return lists of indices for train, validation, and test splits.
    """
    assert len(texts) == len(labels)
    size = len(texts)
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    train_end = int(size * train_ratio)
    val_end = train_end + int(size * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


@dataclass
class HierarchicalSample:
    tokens: torch.LongTensor
    sentence_mask: torch.BoolTensor
    word_mask: torch.BoolTensor
    label: torch.FloatTensor


def sentence_tokenize(text: str) -> List[List[str]]:
    sentences = [s for s in SENTENCE_SPLIT_REGEX.split(text.strip()) if s]
    return [TOKEN_REGEX.findall(sent.lower()) for sent in sentences if sent]


class ReviewDataset(Dataset[HierarchicalSample]):
    """
    Dataset that converts reviews into a padded word-by-sentence tensor suitable
    for Hierarchical Attention Networks.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        indices: Sequence[int],
        vocab: Vocabulary,
        max_sentences: int = 15,
        max_words: int = 30,
    ) -> None:
        self.vocab = vocab
        self.samples: List[HierarchicalSample] = []
        for idx in indices:
            encoded, sent_mask, word_mask = self._encode(texts[idx], vocab, max_sentences, max_words)
            label_tensor = torch.tensor(float(labels[idx]), dtype=torch.float32)
            self.samples.append(
                HierarchicalSample(
                    tokens=encoded,
                    sentence_mask=sent_mask,
                    word_mask=word_mask,
                    label=label_tensor,
                )
            )

    @staticmethod
    def _encode(
        text: str,
        vocab: Vocabulary,
        max_sentences: int,
        max_words: int,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor]:
        sentences = sentence_tokenize(text)
        encoded = torch.zeros((max_sentences, max_words), dtype=torch.long)
        sentence_mask = torch.zeros(max_sentences, dtype=torch.bool)
        word_mask = torch.zeros((max_sentences, max_words), dtype=torch.bool)
        for s_idx in range(min(len(sentences), max_sentences)):
            tokens = sentences[s_idx][:max_words]
            if not tokens:
                continue
            sentence_mask[s_idx] = True
            for w_idx, token in enumerate(tokens):
                encoded[s_idx, w_idx] = vocab.token_to_index(token)
                word_mask[s_idx, w_idx] = True
        return encoded, sentence_mask, word_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> HierarchicalSample:
        return self.samples[index]


def collate_hierarchical(batch: Sequence[HierarchicalSample]) -> dict[str, torch.Tensor]:
    tokens = torch.stack([sample.tokens for sample in batch], dim=0)
    sent_mask = torch.stack([sample.sentence_mask for sample in batch], dim=0)
    word_mask = torch.stack([sample.word_mask for sample in batch], dim=0)
    labels = torch.stack([sample.label for sample in batch], dim=0)
    return {
        "tokens": tokens,
        "sentence_mask": sent_mask,
        "word_mask": word_mask,
        "labels": labels,
    }


def build_datasets(
    csv_path: Path,
    max_sentences: int = 15,
    max_words: int = 30,
    min_freq: int = 2,
    max_vocab_size: Optional[int] = 20000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[ReviewDataset, ReviewDataset, ReviewDataset, Vocabulary]:
    texts, labels = read_reviews(csv_path)
    vocab = Vocabulary(min_freq=min_freq, max_size=max_vocab_size)
    vocab.build(texts)
    train_idx, val_idx, test_idx = split_dataset(texts, labels, train_ratio, val_ratio, seed)
    train_dataset = ReviewDataset(texts, labels, train_idx, vocab, max_sentences, max_words)
    val_dataset = ReviewDataset(texts, labels, val_idx, vocab, max_sentences, max_words)
    test_dataset = ReviewDataset(texts, labels, test_idx, vocab, max_sentences, max_words)
    return train_dataset, val_dataset, test_dataset, vocab


def get_text_splits(
    csv_path: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, List]:
    texts, labels = read_reviews(csv_path)
    train_idx, val_idx, test_idx = split_dataset(texts, labels, train_ratio, val_ratio, seed)
    return {
        "texts": texts,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


class SequenceDataset(Dataset[Tuple[torch.LongTensor, torch.FloatTensor]]):
    """
    Dataset that returns flattened token sequences with padding.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        indices: Sequence[int],
        vocab: Vocabulary,
        max_tokens: int = 200,
    ) -> None:
        self.samples: List[Tuple[torch.LongTensor, torch.FloatTensor]] = []
        for idx in indices:
            tokens = tokenize(texts[idx])[:max_tokens]
            encoded = torch.zeros(max_tokens, dtype=torch.long)
            for t_idx, token in enumerate(tokens):
                encoded[t_idx] = vocab.token_to_index(token)
            label_tensor = torch.tensor(float(labels[idx]), dtype=torch.float32)
            self.samples.append((encoded, label_tensor))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        return self.samples[index]


def build_sequence_datasets(
    csv_path: Path,
    max_tokens: int = 200,
    min_freq: int = 2,
    max_vocab_size: Optional[int] = 20000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset, Vocabulary]:
    texts, labels = read_reviews(csv_path)
    vocab = Vocabulary(min_freq=min_freq, max_size=max_vocab_size)
    vocab.build(texts)
    train_idx, val_idx, test_idx = split_dataset(texts, labels, train_ratio, val_ratio, seed)
    train_dataset = SequenceDataset(texts, labels, train_idx, vocab, max_tokens)
    val_dataset = SequenceDataset(texts, labels, val_idx, vocab, max_tokens)
    test_dataset = SequenceDataset(texts, labels, test_idx, vocab, max_tokens)
    return train_dataset, val_dataset, test_dataset, vocab


def collate_sequences(batch: Sequence[Tuple[torch.LongTensor, torch.FloatTensor]]) -> dict[str, torch.Tensor]:
    tokens = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return {"tokens": tokens, "labels": labels}
