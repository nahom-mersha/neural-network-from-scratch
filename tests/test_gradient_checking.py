import numpy as np

from digit_nn.gradient_checking import gradient_check
from digit_nn.model import BinaryClassifier


def test_binary_classifier_gradients_are_correct() -> None:
    model = BinaryClassifier(
        input_size=3,
        hidden_size=4,
        output_size=2,
        seed=42,
    )

    inputs = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.5, 0.3, 0.1],
        ]
    )

    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    largest_difference = gradient_check(
        model,
        inputs,
        targets,
    )

    assert largest_difference < 1e-5
