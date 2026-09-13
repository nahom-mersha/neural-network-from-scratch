import numpy as np

from digit_nn.model import BinaryClassifier


def test_binary_classifier_forward_shape() -> None:
    model = BinaryClassifier(
        input_size=3,
        hidden_size=4,
        output_size=2,
        seed=42,
    )

    inputs = np.ones((5, 3))

    probabilities = model.forward(inputs)

    assert probabilities.shape == (5, 2)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(5),
    )


def test_binary_classifier_backward_shapes() -> None:
    model = BinaryClassifier(
        input_size=3,
        hidden_size=4,
        output_size=2,
        seed=42,
    )

    inputs = np.ones((5, 3))
    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    model.forward(inputs)
    model.backward(targets)

    assert model.dense1.weights_gradient.shape == (3, 4)
    assert model.dense1.biases_gradient.shape == (4,)
    assert model.dense2.weights_gradient.shape == (4, 2)
    assert model.dense2.biases_gradient.shape == (2,)


def test_binary_classifier_can_reduce_loss() -> None:
    model = BinaryClassifier(
        input_size=2,
        hidden_size=8,
        output_size=2,
        seed=42,
    )

    inputs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    targets = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    model.forward(inputs)
    initial_loss = model.loss(targets)

    for _ in range(300):
        model.train_step(
            inputs,
            targets,
            learning_rate=0.1,
        )

    model.forward(inputs)
    final_loss = model.loss(targets)

    assert final_loss < initial_loss
