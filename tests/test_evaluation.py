import numpy as np

from digit_nn.evaluation import (
    build_confusion_matrix,
    calculate_per_class_accuracy,
)


def test_build_confusion_matrix() -> None:
    labels = np.array([0, 0, 1, 1, 2])
    predictions = np.array([0, 1, 1, 2, 2])

    result = build_confusion_matrix(
        labels,
        predictions,
        number_of_classes=3,
    )

    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_calculate_per_class_accuracy() -> None:
    confusion = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ]
    )

    result = calculate_per_class_accuracy(confusion)

    np.testing.assert_allclose(result, [0.5, 0.5, 1.0])
