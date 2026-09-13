from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np

from digit_nn.model import BinaryClassifier
from digit_nn.training import train

Configuration: TypeAlias = dict[str, int | float]


@dataclass
class ExperimentResult:
    configuration: Configuration
    training_time_seconds: float
    final_train_loss: float
    final_train_accuracy: float
    final_validation_loss: float | None
    final_validation_accuracy: float | None


def run_experiment(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    configuration: Configuration,
    validation_inputs: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
) -> ExperimentResult:
    """Run one controlled experiment."""
    model = BinaryClassifier(
        input_size=train_inputs.shape[1],
        hidden_size=int(configuration.get("hidden_size", 32)),
        output_size=2,
        seed=int(configuration.get("seed", 42)),
    )

    start_time = perf_counter()

    history = train(
        model=model,
        train_inputs=train_inputs,
        train_labels=train_labels,
        epochs=int(configuration.get("epochs", 10)),
        batch_size=int(configuration.get("batch_size", 32)),
        learning_rate=float(configuration.get("learning_rate", 0.01)),
        validation_inputs=validation_inputs,
        validation_labels=validation_labels,
        seed=int(configuration.get("seed", 42)),
    )

    elapsed_time = perf_counter() - start_time

    validation_loss = None
    validation_accuracy = None

    if validation_inputs is not None and validation_labels is not None:
        validation_loss = history.validation_loss[-1]
        validation_accuracy = history.validation_accuracy[-1]

    return ExperimentResult(
        configuration=configuration,
        training_time_seconds=elapsed_time,
        final_train_loss=history.train_loss[-1],
        final_train_accuracy=history.train_accuracy[-1],
        final_validation_loss=validation_loss,
        final_validation_accuracy=validation_accuracy,
    )
