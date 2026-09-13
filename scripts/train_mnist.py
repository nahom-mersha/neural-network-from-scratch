from pathlib import Path

import numpy as np

from digit_nn.data import prepare_multiclass_dataset
from digit_nn.evaluation import evaluate
from digit_nn.model import MulticlassClassifier
from digit_nn.training import train

DATA_PATH = Path("data/raw/mnist.npz")
MODEL_PATH = Path("artifacts/mnist_model.npz")


def main() -> None:
    print("Preparing the complete MNIST dataset...")

    dataset = prepare_multiclass_dataset(
        path=DATA_PATH,
        seed=42,
    )

    model = MulticlassClassifier(
        input_size=784,
        hidden_size=32,
        output_size=10,
        seed=42,
    )

    print("Training the ten-digit model...")

    history = train(
        model=model,
        train_inputs=dataset.train.images,
        train_labels=dataset.train.labels,
        epochs=10,
        batch_size=128,
        learning_rate=0.1,
        validation_inputs=dataset.validation.images,
        validation_labels=dataset.validation.labels,
        seed=42,
    )

    result = evaluate(
        model=model,
        inputs=dataset.test.images,
        labels=dataset.test.labels,
        number_of_classes=10,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        MODEL_PATH,
        dense1_weights=model.dense1.weights,
        dense1_biases=model.dense1.biases,
        dense2_weights=model.dense2.weights,
        dense2_biases=model.dense2.biases,
    )

    print(f"Final training loss: {history.train_loss[-1]:.4f}")
    print(f"Final validation loss: {history.validation_loss[-1]:.4f}")
    print(f"Test accuracy: {result.accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
