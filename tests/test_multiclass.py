import numpy as np

from digit_nn.data import select_all_digits
from digit_nn.model import MulticlassClassifier
from digit_nn.training import one_hot


def test_select_all_digits_preserves_ten_labels() -> None:
    images = np.zeros((10, 2, 2))
    labels = np.arange(10)

    result = select_all_digits(images, labels)

    np.testing.assert_array_equal(result.labels, labels)
    assert result.images.shape == (10, 2, 2)


def test_multiclass_classifier_outputs_ten_probabilities() -> None:
    model = MulticlassClassifier(
        input_size=4,
        hidden_size=6,
        output_size=10,
        seed=42,
    )
    inputs = np.ones((3, 4))

    probabilities = model.forward(inputs)

    assert probabilities.shape == (3, 10)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(3),
    )


def test_one_hot_supports_ten_classes() -> None:
    labels = np.array([0, 4, 9])

    result = one_hot(labels, number_of_classes=10)

    assert result.shape == (3, 10)
    np.testing.assert_array_equal(result.sum(axis=1), np.ones(3))
    np.testing.assert_array_equal(
        result[np.arange(3), labels],
        np.ones(3),
    )
