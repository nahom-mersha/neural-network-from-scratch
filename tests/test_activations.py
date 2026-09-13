import numpy as np

from digit_nn.activations import (
    relu,
    relu_derivative,
    sigmoid,
    softmax,
)


def test_relu() -> None:
    values = np.array([-2.0, 0.0, 3.0])

    np.testing.assert_array_equal(
        relu(values),
        [0.0, 0.0, 3.0],
    )

    np.testing.assert_array_equal(
        relu_derivative(values),
        [0.0, 0.0, 1.0],
    )


def test_sigmoid_is_bounded_and_stable() -> None:
    values = np.array([-100.0, 0.0, 100.0])

    result = sigmoid(values)

    assert np.all(np.isfinite(result))

    np.testing.assert_allclose(
        result,
        [0.0, 0.5, 1.0],
        atol=1e-12,
    )


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array(
        [
            [1000.0, 0.0],
            [1.0, 2.0],
        ]
    )

    probabilities = softmax(logits)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(2),
    )

    assert np.all(np.isfinite(probabilities))
