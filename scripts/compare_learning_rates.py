"""Compare learning rates on a small XOR classification problem.

Controlled experiment:
- same dataset;
- same hidden-layer size;
- same number of epochs;
- same batch size;
- same random seed;
- only the learning rate changes.

The goal is to observe how the learning rate affects learning speed and
final performance.
"""

import numpy as np

from digit_nn.experiments import run_experiment


def main() -> None:
    # XOR requires a hidden layer because it is not linearly separable.
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    labels = np.array([0, 1, 1, 0])

    learning_rates = [0.001, 0.01, 0.1]

    results = []

    for learning_rate in learning_rates:
        configuration = {
            "learning_rate": learning_rate,
            "hidden_size": 8,
            "batch_size": 4,
            "epochs": 300,
            "seed": 42,
        }

        result = run_experiment(
            train_inputs=inputs,
            train_labels=labels,
            configuration=configuration,
        )

        results.append(result)

    print("| Learning rate | Final loss | Final accuracy | Time (seconds) |")
    print("|---:|---:|---:|---:|")

    for result in results:
        learning_rate = result.configuration["learning_rate"]

        print(
            f"| {learning_rate} "
            f"| {result.final_train_loss:.4f} "
            f"| {result.final_train_accuracy:.2f} "
            f"| {result.training_time_seconds:.4f} |"
        )

    best_result = min(
        results,
        key=lambda result: result.final_train_loss,
    )

    print()
    print("Best configuration by final training loss:")
    print(best_result.configuration)


if __name__ == "__main__":
    main()
