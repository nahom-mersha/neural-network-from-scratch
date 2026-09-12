import numpy as np


class Dense:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        # He initialization: useful before ReLU.
        self.weights = rng.standard_normal((input_size, output_size)) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros(output_size)

        self.inputs: np.ndarray | None = None
        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward")

        self.weights_gradient = self.inputs.T @ output_gradient
        self.biases_gradient = output_gradient.sum(axis=0)

        # Gradient passed to the previous layer.
        input_gradient = output_gradient @ self.weights.T

        return input_gradient

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient
