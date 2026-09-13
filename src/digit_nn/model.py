import numpy as np

from digit_nn.activations import relu, relu_derivative, softmax
from digit_nn.layers import Dense
from digit_nn.losses import (
    cross_entropy,
    softmax_cross_entropy_gradient,
)


class BinaryClassifier:
    """Two-layer neural network for classifying digits 1 and 2."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 32,
        output_size: int = 2,
        seed: int | None = None,
    ) -> None:
        self.dense1 = Dense(
            input_size=input_size,
            output_size=hidden_size,
            seed=seed,
        )

        self.dense2 = Dense(
            input_size=hidden_size,
            output_size=output_size,
            seed=None if seed is None else seed + 1,
        )

        self.z1: np.ndarray | None = None
        self.a1: np.ndarray | None = None
        self.z2: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the complete forward pass."""
        self.z1 = self.dense1.forward(inputs)
        self.a1 = relu(self.z1)

        self.z2 = self.dense2.forward(self.a1)
        self.probabilities = softmax(self.z2)

        return self.probabilities

    def loss(
        self,
        targets: np.ndarray,
    ) -> float:
        """Calculate the loss using the latest predictions."""
        if self.probabilities is None:
            raise RuntimeError("forward must be called before loss")

        return cross_entropy(self.probabilities, targets)

    def backward(self, targets: np.ndarray) -> None:
        """Run the complete backward pass."""
        if self.z1 is None or self.a1 is None:
            raise RuntimeError("forward must be called before backward")

        if self.probabilities is None:
            raise RuntimeError("forward must be called before backward")

        # Softmax + cross-entropy gradient.
        d_z2 = softmax_cross_entropy_gradient(
            self.probabilities,
            targets,
        )

        # Dense layer 2:
        # dW2, db2 are stored inside dense2.
        d_a1 = self.dense2.backward(d_z2)

        # ReLU:
        d_z1 = d_a1 * relu_derivative(self.z1)

        # Dense layer 1:
        # dW1, db1 are stored inside dense1.
        self.dense1.backward(d_z1)

    def update(self, learning_rate: float) -> None:
        """Update all weights and biases."""
        self.dense1.update(learning_rate)
        self.dense2.update(learning_rate)

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        """Perform one forward, backward, and update step."""
        self.forward(inputs)
        current_loss = self.loss(targets)
        self.backward(targets)
        self.update(learning_rate)

        return current_loss

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the predicted class index for each example."""
        probabilities = self.forward(inputs)
        return np.argmax(probabilities, axis=1)
