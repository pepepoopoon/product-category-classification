"""Загрузка, проверка и разбиение товарных данных."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("product_id", "seller_id", "title", "description", "category")


def validate_frame(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Проверить схему и вернуть нормализованную копию."""
    required = set(REQUIRED_COLUMNS if require_target else REQUIRED_COLUMNS[:-1])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {sorted(missing)}")

    data = frame.copy()
    text_columns = ["product_id", "seller_id", "title", "description"]
    if require_target:
        text_columns.append("category")
    for column in text_columns:
        if data[column].isna().any() and column != "description":
            raise ValueError(f"Столбец {column!r} содержит пропуски")
        data[column] = data[column].fillna("").astype(str).str.strip()

    if data["product_id"].eq("").any() or data["seller_id"].eq("").any():
        raise ValueError("product_id и seller_id не могут быть пустыми")
    if data["product_id"].duplicated().any():
        raise ValueError("product_id должен быть уникальным")
    if (data["title"] + data["description"]).str.strip().eq("").any():
        raise ValueError("У каждого товара должен быть непустой title или description")
    if require_target:
        if data["category"].eq("").any():
            raise ValueError("category не может быть пустой")
    return data


def load_csv(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    """Загрузить UTF-8 CSV и проверить его схему."""
    return validate_frame(pd.read_csv(path), require_target=require_target)


def combine_text(frame: pd.DataFrame) -> pd.Series:
    """Объединить заголовок и описание без изменения строк."""
    return (frame["title"] + " " + frame["description"]).str.strip()


def split_by_seller(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделить продавцов целиком, балансируя по их доминирующей категории."""
    data = validate_frame(frame)
    if data["category"].nunique() < 2:
        raise ValueError("Для обучения нужны как минимум две категории")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Доли validation и test должны быть положительными")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Сумма долей validation и test должна быть меньше 1")

    seller_category_counts = (
        data.groupby(["seller_id", "category"], sort=True).size().rename("count").reset_index()
    )
    dominant = (
        seller_category_counts.sort_values(
            ["seller_id", "count", "category"], ascending=[True, False, True]
        )
        .drop_duplicates("seller_id")
        .set_index("seller_id")["category"]
    )
    per_category = dominant.value_counts()
    scarce = per_category[per_category < 3]
    if not scarce.empty:
        raise ValueError(
            "Для seller-group split нужны минимум три доминирующих продавца на категорию: "
            f"{scarce.to_dict()}"
        )

    rng = np.random.default_rng(random_state)
    assignments: dict[str, str] = {}
    for category in sorted(dominant.unique()):
        sellers = dominant[dominant.eq(category)].index.to_numpy(copy=True)
        rng.shuffle(sellers)
        count = len(sellers)
        test_count = max(1, int(round(count * test_fraction)))
        validation_count = max(1, int(round(count * validation_fraction)))
        if test_count + validation_count >= count:
            validation_count = 1
            test_count = 1
        for seller in sellers[:test_count]:
            assignments[str(seller)] = "test"
        for seller in sellers[test_count : test_count + validation_count]:
            assignments[str(seller)] = "validation"
        for seller in sellers[test_count + validation_count :]:
            assignments[str(seller)] = "train"

    split_names = data["seller_id"].map(assignments)
    if split_names.isna().any():
        raise RuntimeError("Не всем продавцам назначена часть данных")

    offer_keys = combine_text(data).str.casefold().str.replace(r"\s+", " ", regex=True)
    offer_split_counts = (
        pd.DataFrame({"offer_key": offer_keys, "split": split_names})
        .groupby("offer_key")["split"]
        .nunique()
    )
    leaked_keys = offer_split_counts[offer_split_counts > 1].index
    if len(leaked_keys) > 0:
        leaked_product_ids = data.loc[offer_keys.isin(leaked_keys), "product_id"].tolist()
        raise ValueError(
            "Обнаружены дубликаты товарного текста в разных split; "
            f"product_id: {leaked_product_ids[:10]}"
        )

    manifest = data[["product_id", "seller_id", "category"]].copy()
    manifest["split"] = split_names
    parts = tuple(
        data.loc[split_names.eq(name)].reset_index(drop=True)
        for name in (
            "train",
            "validation",
            "test",
        )
    )
    return (*parts, manifest)
