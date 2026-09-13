import numpy as np

from digit_nn.losses import (
    cross_entropy,
    softmax_cross_entropy_gradient,
)


def test_cross_entropy_matches_known_value() -> None:
    probabilities = np.array(
        [
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    targets = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    result = cross_entropy(probabilities, targets)

    expected = (-np.log(0.8) - np.log(0.9)) / 2

    np.testing.assert_allclose(result, expected)


def test_softmax_cross_entropy_gradient() -> None:
    probabilities = np.array(
        [
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )

    targets = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    result = softmax_cross_entropy_gradient(
        probabilities,
        targets,
    )

    expected = np.array(
        [
            [0.1, -0.1],
            [-0.15, 0.15],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_cross_entropy_handles_zero_probability() -> None:
    probabilities = np.array([[0.0, 1.0]])
    targets = np.array([[1.0, 0.0]])

    result = cross_entropy(probabilities, targets)

    assert np.isfinite(result)
