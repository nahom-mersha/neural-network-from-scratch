import numpy as np

from digit_nn.layers import Dense


def test_dense_forward_shape() -> None:
    layer = Dense(input_size=3, output_size=2, seed=42)
    inputs = np.ones((4, 3))

    outputs = layer.forward(inputs)

    assert outputs.shape == (4, 2)


def test_dense_backward_shapes() -> None:
    layer = Dense(input_size=3, output_size=2, seed=42)
    inputs = np.ones((4, 3))
    output_gradient = np.ones((4, 2))

    layer.forward(inputs)
    input_gradient = layer.backward(output_gradient)

    assert input_gradient.shape == (4, 3)
    assert layer.weights_gradient.shape == (3, 2)
    assert layer.biases_gradient.shape == (2,)


def test_dense_update_changes_parameters() -> None:
    layer = Dense(input_size=3, output_size=2, seed=42)
    original_weights = layer.weights.copy()

    inputs = np.ones((4, 3))
    layer.forward(inputs)
    layer.backward(np.ones((4, 2)))
    layer.update(learning_rate=0.1)

    assert not np.array_equal(layer.weights, original_weights)
