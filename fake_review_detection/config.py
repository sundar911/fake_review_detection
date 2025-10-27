from __future__ import annotations

BEST_CONFIGS = {
    "logistic_regression": {
        "vectorizer": {"ngram_range": (1, 2), "max_features": 5000, "min_df": 2},
        "model": {"solver": "liblinear", "C": 1.0, "max_iter": 1000},
        "train_ratio": 0.7,
        "val_ratio": 0.15,
    },
    "textcnn": {
        "data": {"max_tokens": 150, "min_freq": 2},
        "model": {"embedding_dim": 200, "num_filters": 128, "dropout": 0.3},
        "training": {"epochs": 3, "batch_size": 32, "learning_rate": 1e-3},
    },
    "textbilstm": {
        "data": {"max_tokens": 150, "min_freq": 2},
        "model": {"embedding_dim": 200, "hidden_dim": 64, "dropout": 0.3},
        "training": {"epochs": 3, "batch_size": 32, "learning_rate": 1e-3},
    },
    "han": {
        "data": {"max_sentences": 15, "max_words": 30, "min_freq": 2},
        "model": {
            "embedding_dim": 500,
            "word_hidden_dim": 32,
            "sentence_hidden_dim": 32,
            "embedding_dropout": 0.4,
            "word_dropout": 0.4,
            "sentence_dropout": 0.4,
        },
        "training": {"epochs": 5, "batch_size": 16, "learning_rate": 1e-3},
    },
    "multihead_han": {
        "data": {"max_sentences": 15, "max_words": 30, "min_freq": 2},
        "model": {
            "embedding_dim": 320,
            "word_hidden_dim": 96,
            "sentence_hidden_dim": 96,
            "embedding_dropout": 0.2,
            "word_dropout": 0.2,
            "sentence_dropout": 0.2,
            "word_heads": 4,
            "sentence_heads": 4,
            "attention_dropout": 0.2,
        },
        "training": {"epochs": 5, "batch_size": 16, "learning_rate": 0.002},
    },
}

