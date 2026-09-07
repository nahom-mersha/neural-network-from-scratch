from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    images: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


def load_mnist(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path) as data:
        return (
            data["x_train"],
            data["y_train"],
            data["x_test"],
            data["y_test"],
        )


def select_digits(
    images: np.ndarray,
    labels: np.ndarray,
    first_digit: int = 1,
    second_digit: int = 2,
) -> DatasetSplit:
    mask = np.isin(labels, [first_digit, second_digit])

    selected_images = images[mask]
    selected_labels = labels[mask]

    # Map digit 1 → class 0 and digit 2 → class 1.
    binary_labels = (selected_labels == second_digit).astype(np.int64)

    return DatasetSplit(selected_images, binary_labels)


def flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    flattened = images.reshape(images.shape[0], -1)
    return flattened.astype(np.float32) / 255.0


def split_train_validation(
    dataset: DatasetSplit,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DatasetSplit, DatasetSplit]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset.labels))

    validation_size = int(len(indices) * validation_ratio)
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train = DatasetSplit(
        dataset.images[train_indices],
        dataset.labels[train_indices],
    )
    validation = DatasetSplit(
        dataset.images[validation_indices],
        dataset.labels[validation_indices],
    )

    return train, validation


def prepare_binary_dataset(
    path: Path,
    first_digit: int = 1,
    second_digit: int = 2,
    seed: int = 42,
) -> PreparedDataset:
    x_train, y_train, x_test, y_test = load_mnist(path)

    train_and_validation = select_digits(
        x_train,
        y_train,
        first_digit,
        second_digit,
    )
    test = select_digits(
        x_test,
        y_test,
        first_digit,
        second_digit,
    )

    train, validation = split_train_validation(
        train_and_validation,
        seed=seed,
    )

    train = DatasetSplit(flatten_and_normalize(train.images), train.labels)
    validation = DatasetSplit(
        flatten_and_normalize(validation.images),
        validation.labels,
    )
    test = DatasetSplit(flatten_and_normalize(test.images), test.labels)

    return PreparedDataset(train, validation, test)
