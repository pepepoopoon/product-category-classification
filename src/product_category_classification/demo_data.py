"""Генерация детерминированного синтетического набора."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_smoke_data() -> pd.DataFrame:
    categories = {
        "electronics": [
            "wireless gaming mouse",
            "usb c charging cable",
            "portable bluetooth speaker",
        ],
        "home": [
            "cotton kitchen towel",
            "ceramic coffee mug",
            "wooden storage shelf",
        ],
        "sports": [
            "fitness yoga mat",
            "training football ball",
            "lightweight running bottle",
        ],
    }
    rows: list[dict[str, str]] = []
    product_number = 1
    for category, titles in categories.items():
        for seller_number in range(1, 5):
            seller = f"{category[:2]}-seller-{seller_number}"
            for title_number, title in enumerate(titles, start=1):
                rows.append(
                    {
                        "product_id": f"p-{product_number:03d}",
                        "seller_id": seller,
                        "title": f"{title} model {seller_number}{title_number}",
                        "description": f"synthetic {category} catalog example",
                        "category": category,
                    }
                )
                product_number += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_smoke_data().to_csv(destination, index=False)


if __name__ == "__main__":
    main()
