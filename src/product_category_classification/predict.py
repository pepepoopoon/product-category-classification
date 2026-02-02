"""CLI инференса товарной категории."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_bundle


def predict(model_path: str | Path, text: str, *, top_k: int = 2) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("Текст для предсказания не может быть пустым")
    if top_k < 1:
        raise ValueError("top_k должен быть положительным")
    model = load_bundle(model_path)["model"]
    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_
    indices = np.argsort(probabilities)[::-1][: min(top_k, len(classes))]
    candidates = [
        {"category": str(classes[index]), "probability": float(probabilities[index])}
        for index in indices
    ]
    return {"prediction": candidates[0]["category"], "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    result = predict(args.model, args.text, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
