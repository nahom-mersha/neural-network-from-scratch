from dataclasses import dataclass

import numpy as np

from digit_nn.model import BinaryClassifier


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    train_accuracy: list[float]
    validation_accuracy: list[float]


def one_hot(
    labels: np.ndarray,
    number_of_classes: int | None = None,
) -> np.ndarray:
    """Convert integer labels into one-hot vectors."""
    labels = labels.astype(int)

    if number_of_classes is None:
        number_of_classes = int(labels.max()) + 1

    result = np.zeros(
        (labels.shape[0], number_of_classes),
        dtype=float,
    )

    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def accuracy(
    model: BinaryClassifier,
    inputs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Calculate classification accuracy."""
    predictions = model.predict(inputs)

    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)

    return float(np.mean(predictions == labels))


def train(
    model: BinaryClassifier,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    validation_inputs: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    seed: int | None = 42,
) -> TrainingHistory:
    """Train the model using mini-batch gradient descent."""
    train_targets = (
        train_labels
        if train_labels.ndim == 2
        else one_hot(train_labels, model.dense2.biases.shape[0])
    )

    validation_targets = None

    if validation_labels is not None:
        validation_targets = (
            validation_labels
            if validation_labels.ndim == 2
            else one_hot(
                validation_labels,
                model.dense2.biases.shape[0],
            )
        )

    rng = np.random.default_rng(seed)

    history = TrainingHistory(
        train_loss=[],
        validation_loss=[],
        train_accuracy=[],
        validation_accuracy=[],
    )

    number_of_examples = train_inputs.shape[0]

    for _ in range(epochs):
        shuffled_indices = rng.permutation(number_of_examples)

        for start in range(0, number_of_examples, batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]

            model.train_step(
                train_inputs[batch_indices],
                train_targets[batch_indices],
                learning_rate,
            )

        model.forward(train_inputs)
        epoch_train_loss = model.loss(train_targets)

        history.train_loss.append(epoch_train_loss)
        history.train_accuracy.append(accuracy(model, train_inputs, train_labels))

        if (
            validation_inputs is not None
            and validation_labels is not None
            and validation_targets is not None
        ):
            model.forward(validation_inputs)
            epoch_validation_loss = model.loss(validation_targets)

            history.validation_loss.append(epoch_validation_loss)
            history.validation_accuracy.append(
                accuracy(model, validation_inputs, validation_labels)
            )

    return history
