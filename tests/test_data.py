import pandas as pd
import pytest

from product_category_classification.data import split_by_seller, validate_frame
from product_category_classification.demo_data import make_smoke_data


def test_validation_rejects_duplicate_product_ids() -> None:
    frame = make_smoke_data()
    frame.loc[1, "product_id"] = frame.loc[0, "product_id"]

    with pytest.raises(ValueError, match="уникальным"):
        validate_frame(frame)


def test_split_has_no_seller_leakage_and_is_deterministic() -> None:
    first = split_by_seller(make_smoke_data(), random_state=17)
    second = split_by_seller(make_smoke_data(), random_state=17)

    train, validation, test, manifest = first
    assert set(train["seller_id"]).isdisjoint(validation["seller_id"])
    assert set(train["seller_id"]).isdisjoint(test["seller_id"])
    assert set(validation["seller_id"]).isdisjoint(test["seller_id"])
    assert set(manifest["split"]) == {"train", "validation", "test"}
    pd.testing.assert_frame_equal(manifest, second[3])
