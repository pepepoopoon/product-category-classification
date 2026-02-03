"""Запуск воспроизводимого synthetic-эксперимента классификации товаров."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from .data import combine_text, split_by_seller
from .demo_data import make_smoke_data
from .model import build_model, classification_metrics


def run_experiment(
    *,
    sellers_per_category: int,
    data_seed: int,
    split_seed: int,
    model_seed: int,
    hypothesis: str,
) -> dict[str, object]:
    """Обучить модель и сохранить seller-disjoint validation/test диагностику."""
    if not hypothesis.strip():
        raise ValueError("hypothesis не должна быть пустой")
    frame = make_smoke_data(
        sellers_per_category=sellers_per_category,
        seed=data_seed,
    )
    train, validation, test, manifest = split_by_seller(frame, random_state=split_seed)
    model = build_model(random_state=model_seed)
    model.fit(combine_text(train), train["category"])
    validation_metrics = classification_metrics(
        model,
        combine_text(validation),
        validation["category"],
    )
    test_metrics = classification_metrics(model, combine_text(test), test["category"])
    prediction = model.predict(combine_text(test))
    labels = sorted(frame["category"].unique().tolist())

    return {
        "schema_version": 1,
        "experiment": "synthetic_product_category_sensitivity",
        "hypothesis": hypothesis.strip(),
        "parameters": {
            "sellers_per_category": sellers_per_category,
            "data_seed": data_seed,
            "split_seed": split_seed,
            "model_seed": model_seed,
        },
        "dataset": {
            "mode": "synthetic",
            "rows": len(frame),
            "sellers": int(frame["seller_id"].nunique()),
            "split_rows": manifest["split"].value_counts().sort_index().to_dict(),
            "split_sellers": manifest.groupby("split")["seller_id"].nunique().to_dict(),
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--sellers-per-category", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = run_experiment(
        sellers_per_category=args.sellers_per_category,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        hypothesis=args.hypothesis,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Результат эксперимента сохранён в {args.output}")


if __name__ == "__main__":
    main()
