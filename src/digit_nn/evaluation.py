from dataclasses import dataclass

import numpy as np

from digit_nn.model import BinaryClassifier


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    per_class_accuracy: np.ndarray
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    incorrect_indices: np.ndarray


def build_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    number_of_classes: int,
) -> np.ndarray:
    """Count true-class versus predicted-class pairs."""
    matrix = np.zeros(
        (number_of_classes, number_of_classes),
        dtype=np.int64,
    )

    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def calculate_per_class_accuracy(
    confusion: np.ndarray,
) -> np.ndarray:
    """Calculate accuracy separately for each true class."""
    correct = np.diag(confusion)
    totals = confusion.sum(axis=1)

    return np.divide(
        correct,
        totals,
        out=np.zeros(confusion.shape[0], dtype=float),
        where=totals != 0,
    )


def evaluate(
    model: BinaryClassifier,
    inputs: np.ndarray,
    labels: np.ndarray,
    number_of_classes: int,
) -> EvaluationResult:
    """Evaluate predictions and collect basic error-analysis metrics."""
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)

    probabilities = model.forward(inputs)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    confusion = build_confusion_matrix(
        labels,
        predictions,
        number_of_classes,
    )

    incorrect_indices = np.flatnonzero(predictions != labels)

    return EvaluationResult(
        accuracy=float(np.mean(predictions == labels)),
        per_class_accuracy=calculate_per_class_accuracy(confusion),
        confusion_matrix=confusion,
        predictions=predictions,
        confidence=confidence,
        incorrect_indices=incorrect_indices,
    )
