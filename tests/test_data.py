import numpy as np

from digit_nn.data import (
    flatten_and_normalize,
    select_digits,
)


def test_flatten_and_normalize() -> None:
    images = np.array([[[0, 255], [128, 64]]])

    result = flatten_and_normalize(images)

    assert result.shape == (1, 4)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    np.testing.assert_allclose(
        result,
        [[0.0, 1.0, 128 / 255, 64 / 255]],
    )


def test_select_digits() -> None:
    images = np.zeros((4, 2, 2))
    labels = np.array([1, 2, 3, 2])

    result = select_digits(images, labels)

    assert result.images.shape == (3, 2, 2)
    np.testing.assert_array_equal(result.labels, [0, 1, 1])
