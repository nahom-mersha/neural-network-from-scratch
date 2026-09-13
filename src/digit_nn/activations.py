import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """Apply ReLU element by element."""
    return np.maximum(0.0, values)


def relu_derivative(values: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (values > 0).astype(float)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply sigmoid with numerical stability."""
    result = np.empty_like(values, dtype=float)

    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)

    return result


def sigmoid_derivative(values: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid."""
    probabilities = sigmoid(values)
    return probabilities * (1.0 - probabilities)


def softmax(
    logits: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Convert logits into probabilities safely."""
    shifted_logits = logits - np.max(
        logits,
        axis=axis,
        keepdims=True,
    )

    exponentials = np.exp(shifted_logits)

    return exponentials / np.sum(
        exponentials,
        axis=axis,
        keepdims=True,
    )
