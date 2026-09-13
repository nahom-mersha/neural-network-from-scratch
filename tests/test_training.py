import numpy as np

from digit_nn.model import BinaryClassifier
from digit_nn.training import one_hot, train


def test_one_hot_encoding() -> None:
    labels = np.array([0, 1, 1, 0])

    result = one_hot(labels)

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_training_reduces_loss() -> None:
    inputs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    labels = np.array([0, 1])

    model = BinaryClassifier(
        input_size=2,
        hidden_size=8,
        output_size=2,
        seed=42,
    )

    history = train(
        model=model,
        train_inputs=inputs,
        train_labels=labels,
        epochs=100,
        batch_size=2,
        learning_rate=0.1,
        seed=42,
    )

    assert history.train_loss[-1] < history.train_loss[0]
    assert history.train_accuracy[-1] >= 0.5
