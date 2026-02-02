"""CLI обучения товарного классификатора."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from .data import combine_text, load_csv, split_by_seller
from .io import write_json
from .model import build_model, classification_metrics

LOGGER = logging.getLogger(__name__)


def train(data_path: str | Path, output_dir: str | Path, *, random_state: int = 42) -> dict:
    """Обучить модель и сохранить воспроизводимые артефакты."""
    data = load_csv(data_path)
    train_frame, validation_frame, test_frame, manifest = split_by_seller(
        data, random_state=random_state
    )
    model = build_model(random_state=random_state)
    model.fit(combine_text(train_frame), train_frame["category"])
    validation_metrics = classification_metrics(
        model,
        combine_text(validation_frame),
        validation_frame["category"],
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "schema_version": 1,
        "random_state": random_state,
        "text_fields": ["title", "description"],
    }
    joblib.dump(bundle, destination / "model.joblib")
    manifest.to_csv(destination / "split_manifest.csv", index=False)
    validation_frame.to_csv(destination / "validation.csv", index=False)
    test_frame.to_csv(destination / "test.csv", index=False)
    metadata = {
        "schema_version": 1,
        "random_state": random_state,
        "split_strategy": "seller-group with dominant-category balancing",
        "split_rows": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "validation_metrics": validation_metrics,
    }
    write_json(destination / "metadata.json", metadata)
    LOGGER.info("Артефакты сохранены в %s", destination)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Путь к входному CSV")
    parser.add_argument("--output-dir", required=True, help="Каталог артефактов")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(args.data, args.output_dir, random_state=args.seed)


if __name__ == "__main__":
    main()
