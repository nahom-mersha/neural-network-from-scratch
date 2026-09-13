import numpy as np


def cross_entropy(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate mean multiclass cross-entropy loss."""
    clipped_probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    losses_per_example = -np.sum(
        targets * np.log(clipped_probabilities),
        axis=1,
    )

    return float(np.mean(losses_per_example))


def softmax_cross_entropy_gradient(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Calculate dL/d_logits for mean softmax cross-entropy."""
    batch_size = probabilities.shape[0]

    return (probabilities - targets) / batch_size
