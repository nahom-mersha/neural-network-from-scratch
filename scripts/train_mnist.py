import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from digit_nn.data import prepare_multiclass_dataset
from digit_nn.evaluation import evaluate
from digit_nn.model import MulticlassClassifier
from digit_nn.training import train

DATA_PATH = Path("data/raw/mnist.npz")
MODEL_PATH = Path("artifacts/mnist_model.npz")
METRICS_PATH = Path("artifacts/training_metrics.json")


CONFIG = {
    "input_size": 784,
    "hidden_size": 32,
    "output_size": 10,
    "epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.1,
    "seed": 42,
}


def save_metrics(record: dict) -> None:
    if METRICS_PATH.exists():
        records = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        records = []

    records.append(record)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    METRICS_PATH.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("Preparing the complete MNIST dataset...")

    dataset = prepare_multiclass_dataset(
        path=DATA_PATH,
        seed=CONFIG["seed"],
    )

    model = MulticlassClassifier(
        input_size=CONFIG["input_size"],
        hidden_size=CONFIG["hidden_size"],
        output_size=CONFIG["output_size"],
        seed=CONFIG["seed"],
    )

    print("Training the ten-digit model...")

    history = train(
        model=model,
        train_inputs=dataset.train.images,
        train_labels=dataset.train.labels,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        validation_inputs=dataset.validation.images,
        validation_labels=dataset.validation.labels,
        seed=CONFIG["seed"],
    )

    result = evaluate(
        model=model,
        inputs=dataset.test.images,
        labels=dataset.test.labels,
        number_of_classes=CONFIG["output_size"],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        MODEL_PATH,
        dense1_weights=model.dense1.weights,
        dense1_biases=model.dense1.biases,
        dense2_weights=model.dense2.weights,
        dense2_biases=model.dense2.biases,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    confusion_matrix_path = Path(f"artifacts/confusion_matrix_{run_id}.csv")

    np.savetxt(
        confusion_matrix_path,
        result.confusion_matrix,
        delimiter=",",
        fmt="%d",
    )

    record = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "configuration": CONFIG,
        "training": {
            "loss_history": history.train_loss,
            "accuracy_history": history.train_accuracy,
            "final_loss": history.train_loss[-1],
            "final_accuracy": history.train_accuracy[-1],
        },
        "validation": {
            "loss_history": history.validation_loss,
            "accuracy_history": history.validation_accuracy,
            "final_loss": history.validation_loss[-1],
            "final_accuracy": history.validation_accuracy[-1],
        },
        "test": {
            "accuracy": result.accuracy,
            "per_class_accuracy": (result.per_class_accuracy.tolist()),
            "confusion_matrix_file": str(confusion_matrix_path),
            "mean_confidence": float(np.mean(result.confidence)),
            "minimum_confidence": float(np.min(result.confidence)),
            "maximum_confidence": float(np.max(result.confidence)),
            "incorrect_count": int(len(result.incorrect_indices)),
        },
    }

    save_metrics(record)

    print(f"Final training loss: {history.train_loss[-1]:.4f}")
    print(f"Final validation loss: {history.validation_loss[-1]:.4f}")
    print(f"Test accuracy: {result.accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
