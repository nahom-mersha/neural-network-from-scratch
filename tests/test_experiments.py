import numpy as np

from digit_nn.experiments import run_experiment


def test_experiment_records_results() -> None:
    inputs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )

    labels = np.array([0, 1, 1, 0])

    configuration = {
        "learning_rate": 0.1,
        "hidden_size": 8,
        "batch_size": 2,
        "epochs": 10,
        "seed": 42,
    }

    result = run_experiment(
        train_inputs=inputs,
        train_labels=labels,
        configuration=configuration,
    )

    assert result.configuration == configuration
    assert result.training_time_seconds >= 0.0
    assert np.isfinite(result.final_train_loss)
    assert 0.0 <= result.final_train_accuracy <= 1.0
