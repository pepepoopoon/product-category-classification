import numpy as np
import pytest

from product_category_classification.model import top_k_accuracy


@pytest.mark.parametrize("invalid_k", [0, 3])
def test_top_k_accuracy_rejects_k_outside_class_count(invalid_k: int) -> None:
    labels = np.asarray(["electronics", "home"])
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    classes = np.asarray(["electronics", "home"])

    with pytest.raises(ValueError, match=r"k должен быть в диапазоне \[1, 2\]"):
        top_k_accuracy(labels, probabilities, classes, invalid_k)
