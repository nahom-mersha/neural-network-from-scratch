import numpy as np

from digit_nn.model import BinaryClassifier


def _loss(
    model: BinaryClassifier,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> float:
    model.forward(inputs)
    return model.loss(targets)


def numerical_gradient(
    model: BinaryClassifier,
    inputs: np.ndarray,
    targets: np.ndarray,
    parameter: np.ndarray,
    index: tuple[int, ...],
    epsilon: float = 1e-5,
) -> float:
    """Calculate one numerical gradient using central differences."""
    original_value = parameter[index]

    parameter[index] = original_value + epsilon
    loss_plus = _loss(model, inputs, targets)

    parameter[index] = original_value - epsilon
    loss_minus = _loss(model, inputs, targets)

    parameter[index] = original_value

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def gradient_check(
    model: BinaryClassifier,
    inputs: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-5,
) -> float:
    """Return the largest difference between analytical and numerical gradients."""
    model.forward(inputs)
    model.backward(targets)

    parameters_and_gradients = [
        (model.dense1.weights, model.dense1.weights_gradient),
        (model.dense1.biases, model.dense1.biases_gradient),
        (model.dense2.weights, model.dense2.weights_gradient),
        (model.dense2.biases, model.dense2.biases_gradient),
    ]

    largest_difference = 0.0

    for parameter, analytical_gradient in parameters_and_gradients:
        for index in np.ndindex(parameter.shape):
            numerical = numerical_gradient(
                model,
                inputs,
                targets,
                parameter,
                index,
                epsilon,
            )

            analytical = analytical_gradient[index]
            difference = abs(analytical - numerical)

            largest_difference = max(
                largest_difference,
                difference,
            )

    return float(largest_difference)
