"""Модель и метрики классификации товаров."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


def build_model(*, random_state: int = 42) -> Pipeline:
    """Построить word/char TF-IDF модель."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> float:
    """Посчитать top-k для допустимого числа кандидатов."""
    class_count = probabilities.shape[1]
    if not 1 <= k <= class_count:
        raise ValueError(f"k должен быть в диапазоне [1, {class_count}], получено: {k}")
    top_indices = np.argsort(probabilities, axis=1)[:, -k:]
    top_labels = classes[top_indices]
    return float(np.mean([truth in row for truth, row in zip(y_true, top_labels, strict=True)]))


def classification_metrics(
    model: Pipeline,
    texts: Any,
    labels: Any,
    *,
    top_k: int = 2,
) -> dict[str, Any]:
    """Рассчитать основные метрики."""
    y_true = np.asarray(labels)
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    classes = model.named_steps["classifier"].classes_
    return {
        "rows": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        f"top_{top_k}_accuracy": top_k_accuracy(y_true, probabilities, classes, top_k),
        "labels": [str(label) for label in classes],
    }
