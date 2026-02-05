"""Запуск воспроизводимого synthetic-эксперимента классификации товаров."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .data import combine_text, split_by_seller
from .demo_data import make_smoke_data
from .model import build_model, classification_metrics


def select_text(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "both":
        return combine_text(frame)
    if mode not in {"title", "description"}:
        raise ValueError("text_field_mode должна быть both, title или description")
    return frame[mode].astype(str).str.strip()


def perturb_text(texts: pd.Series, *, rate: float, seed: int) -> pd.Series:
    if not 0 <= rate < 0.8:
        raise ValueError("text_noise_rate должна быть в диапазоне [0, 0.8)")
    rng = np.random.default_rng(seed + 70_001)
    result = []
    for text in texts.astype(str):
        tokens = text.split()
        changed = [token if rng.random() >= rate else f"#{len(token)}" for token in tokens]
        result.append(" ".join(changed))
    return pd.Series(result, index=texts.index, dtype="string")


def perturb_labels(labels: pd.Series, *, rate: float, seed: int) -> pd.Series:
    if not 0 <= rate < 0.5:
        raise ValueError("label_noise_rate должна быть в диапазоне [0, 0.5)")
    result = labels.copy()
    if rate == 0:
        return result
    rng = np.random.default_rng(seed + 80_009)
    classes = sorted(result.unique().tolist())
    mapping = {label: classes[(index + 1) % len(classes)] for index, label in enumerate(classes)}
    mask = rng.random(len(result)) < rate
    result.loc[mask] = result.loc[mask].map(mapping)
    return result


def run_experiment(
    *,
    sellers_per_category: int,
    data_seed: int,
    split_seed: int,
    model_seed: int,
    hypothesis: str,
    regularization_c: float = 1.0,
    feature_mode: str = "union",
    word_ngram_max: int = 2,
    char_ngram_max: int = 5,
    text_field_mode: str = "both",
    label_noise_rate: float = 0.0,
    text_noise_rate: float = 0.0,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    """Обучить модель и сохранить seller-disjoint validation/test диагностику."""
    if not hypothesis.strip():
        raise ValueError("hypothesis не должна быть пустой")
    frame = make_smoke_data(
        sellers_per_category=sellers_per_category,
        seed=data_seed,
    )
    train, validation, test, manifest = split_by_seller(
        frame,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        random_state=split_seed,
    )
    model = build_model(
        random_state=model_seed,
        regularization_c=regularization_c,
        feature_mode=feature_mode,
        word_ngram_max=word_ngram_max,
        char_ngram_max=char_ngram_max,
    )
    train_text = select_text(train, text_field_mode)
    validation_text = perturb_text(
        select_text(validation, text_field_mode),
        rate=text_noise_rate,
        seed=data_seed + split_seed,
    )
    test_text = perturb_text(
        select_text(test, text_field_mode),
        rate=text_noise_rate,
        seed=data_seed + split_seed + 1,
    )
    train_labels = perturb_labels(train["category"], rate=label_noise_rate, seed=data_seed)
    model.fit(train_text, train_labels)
    validation_metrics = classification_metrics(
        model,
        validation_text,
        validation["category"],
    )
    test_metrics = classification_metrics(model, test_text, test["category"])
    prediction = model.predict(test_text)
    labels = sorted(frame["category"].unique().tolist())

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "synthetic_product_category_sensitivity",
        "hypothesis": hypothesis.strip(),
        "parameters": {
            "sellers_per_category": sellers_per_category,
            "data_seed": data_seed,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "regularization_c": regularization_c,
            "feature_mode": feature_mode,
            "word_ngram_max": word_ngram_max,
            "char_ngram_max": char_ngram_max,
            "text_field_mode": text_field_mode,
            "label_noise_rate": label_noise_rate,
            "text_noise_rate": text_noise_rate,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        },
        "dataset": {
            "mode": "synthetic",
            "rows": len(frame),
            "sellers": int(frame["seller_id"].nunique()),
            "split_rows": manifest["split"].value_counts().sort_index().to_dict(),
            "split_sellers": manifest.groupby("split")["seller_id"].nunique().to_dict(),
            "feature_count": int(model.named_steps["features"].transform(train_text).shape[1]),
            "noisy_train_labels": int((train_labels != train["category"]).sum()),
        },
        "validation": validation_metrics,
        "test": {
            **test_metrics,
            "confusion_matrix": confusion_matrix(
                test["category"], prediction, labels=labels
            ).tolist(),
            "per_class": classification_report(
                test["category"],
                prediction,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        },
    }
    if baseline is not None:
        baseline_validation = baseline.get("validation")
        baseline_test = baseline.get("test")
        baseline_dataset = baseline.get("dataset")
        if not all(
            isinstance(value, dict)
            for value in (baseline_validation, baseline_test, baseline_dataset)
        ):
            raise ValueError("baseline не соответствует схеме эксперимента")
        baseline_feature_count = baseline_dataset.get("feature_count")
        result["comparison"] = {
            "validation_macro_f1_delta": (
                float(validation_metrics["macro_f1"])
                - float(baseline_validation["macro_f1"])
            ),
            "test_macro_f1_delta": (
                float(test_metrics["macro_f1"]) - float(baseline_test["macro_f1"])
            ),
            "test_top_2_accuracy_delta": (
                float(test_metrics["top_2_accuracy"])
                - float(baseline_test["top_2_accuracy"])
            ),
            "feature_count_delta": (
                int(result["dataset"]["feature_count"]) - int(baseline_feature_count)
                if baseline_feature_count is not None
                else None
            ),
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--sellers-per-category", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=["word", "char", "union"], default="union")
    parser.add_argument("--word-ngram-max", type=int, default=2)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--text-field-mode",
        choices=["both", "title", "description"],
        default="both",
    )
    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--text-noise-rate", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    baseline = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    result = run_experiment(
        sellers_per_category=args.sellers_per_category,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        hypothesis=args.hypothesis,
        regularization_c=args.regularization_c,
        feature_mode=args.feature_mode,
        word_ngram_max=args.word_ngram_max,
        char_ngram_max=args.char_ngram_max,
        text_field_mode=args.text_field_mode,
        label_noise_rate=args.label_noise_rate,
        text_noise_rate=args.text_noise_rate,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        baseline=baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Результат эксперимента сохранён в {args.output}")


if __name__ == "__main__":
    main()
