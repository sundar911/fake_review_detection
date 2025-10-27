from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from fake_review_detection.config import BEST_CONFIGS
from fake_review_detection.data import (
    build_datasets,
    build_sequence_datasets,
    collate_sequences,
    get_text_splits,
)
from fake_review_detection.models import (
    HAN,
    HANConfig,
    MultiHeadHAN,
    MultiHeadHANConfig,
    TextBiLSTM,
    TextCNN,
    TextCNNConfig,
    TextRNNConfig,
)
from fake_review_detection.simple_training import (
    SimpleTrainingConfig,
    evaluate_simple_model,
    train_simple_model,
)
from fake_review_detection.training import TrainingConfig, evaluate_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the best-performing configuration for the selected fake review detection model."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "myle_ott.csv",
        help="Path to the review dataset CSV file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "textcnn", "textbilstm", "han", "multihead-han", "all"],
        default="all",
        help="Which model to run. Use 'all' to execute every best configuration.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducibility.")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_logistic(csv_path: Path, seed: int) -> None:
    cfg = BEST_CONFIGS["logistic_regression"]
    splits = get_text_splits(
        csv_path,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=seed,
    )
    texts = splits["texts"]
    labels = splits["labels"]
    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(cfg["vectorizer"]["ngram_range"]),
        max_features=cfg["vectorizer"]["max_features"],
        min_df=cfg["vectorizer"]["min_df"],
    )
    x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    x_val = vectorizer.transform([texts[i] for i in val_idx])
    x_test = vectorizer.transform([texts[i] for i in test_idx])
    y_train = np.array([labels[i] for i in train_idx])
    y_val = np.array([labels[i] for i in val_idx])
    y_test = np.array([labels[i] for i in test_idx])

    model = LogisticRegression(**cfg["model"])
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)

    print("\n=== Logistic Regression (TF-IDF) ===")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test precision: {precision:.4f} | recall: {recall:.4f} | F1: {f1:.4f}")


def _run_simple_model(
    name: str,
    model_cfg: dict,
    data_cfg: dict,
    train_cfg_dict: dict,
    csv_path: Path,
    seed: int,
) -> None:
    train_ds, val_ds, test_ds, vocab = build_sequence_datasets(
        csv_path,
        max_tokens=data_cfg["max_tokens"],
        min_freq=data_cfg.get("min_freq", 1),
        seed=seed,
    )
    if name == "TextCNN":
        model = TextCNN(TextCNNConfig(vocab_size=len(vocab), **model_cfg))
    elif name == "TextBiLSTM":
        model = TextBiLSTM(TextRNNConfig(vocab_size=len(vocab), **model_cfg))
    else:
        raise ValueError(f"Unknown simple model '{name}'")

    train_cfg = SimpleTrainingConfig(**train_cfg_dict)
    history = train_simple_model(model, train_ds, val_ds, collate_sequences, train_cfg)
    val_metrics = history["val_metrics"][-1]
    _, test_metrics = evaluate_simple_model(model, test_ds, collate_sequences, train_cfg)

    print(f"\n=== {name} ===")
    print(f"Validation accuracy: {val_metrics.accuracy:.4f}")
    print(f"Test accuracy: {test_metrics.accuracy:.4f}")
    print(
        "Test precision: "
        f"{test_metrics.precision:.4f} | recall: {test_metrics.recall:.4f} | F1: {test_metrics.f1:.4f}"
    )


def _run_han_variant(
    name: str,
    model_cls,
    model_cfg_cls,
    config_key: str,
    csv_path: Path,
    seed: int,
) -> None:
    cfg = BEST_CONFIGS[config_key]
    train_ds, val_ds, test_ds, vocab = build_datasets(
        csv_path,
        max_sentences=cfg["data"]["max_sentences"],
        max_words=cfg["data"]["max_words"],
        min_freq=cfg["data"].get("min_freq", 1),
        seed=seed,
    )
    model = model_cls(model_cfg_cls(vocab_size=len(vocab), pad_index=0, **cfg["model"]))
    train_cfg = TrainingConfig(**cfg["training"])
    history = train_model(model, train_ds, val_ds, train_cfg)
    val_metrics = history["val_metrics"][-1]
    _, test_metrics = evaluate_model(model, test_ds, train_cfg)

    print(f"\n=== {name} ===")
    print(f"Validation accuracy: {val_metrics.accuracy:.4f}")
    print(f"Test accuracy: {test_metrics.accuracy:.4f}")
    print(
        "Test precision: "
        f"{test_metrics.precision:.4f} | recall: {test_metrics.recall:.4f} | F1: {test_metrics.f1:.4f}"
    )


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    targets = (
        ["logistic", "textcnn", "textbilstm", "han", "multihead-han"] if args.model == "all" else [args.model]
    )
    for model_name in targets:
        if model_name == "logistic":
            run_logistic(args.csv_path, args.seed)
        elif model_name == "textcnn":
            cfg = BEST_CONFIGS["textcnn"]
            _run_simple_model(
                "TextCNN",
                model_cfg=cfg["model"],
                data_cfg=cfg["data"],
                train_cfg_dict=cfg["training"],
                csv_path=args.csv_path,
                seed=args.seed,
            )
        elif model_name == "textbilstm":
            cfg = BEST_CONFIGS["textbilstm"]
            _run_simple_model(
                "TextBiLSTM",
                model_cfg=cfg["model"],
                data_cfg=cfg["data"],
                train_cfg_dict=cfg["training"],
                csv_path=args.csv_path,
                seed=args.seed,
            )
        elif model_name == "han":
            _run_han_variant(
                "Hierarchical Attention Network",
                HAN,
                HANConfig,
                "han",
                args.csv_path,
                args.seed,
            )
        elif model_name == "multihead-han":
            _run_han_variant(
                "Multi-Head HAN",
                MultiHeadHAN,
                MultiHeadHANConfig,
                "multihead_han",
                args.csv_path,
                args.seed,
            )
        else:
            raise ValueError(f"Unsupported model name '{model_name}'")


if __name__ == "__main__":
    main()
