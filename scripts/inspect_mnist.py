from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data/raw")
DATA_PATH = DATA_DIR / "mnist.npz"
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        print("Downloading MNIST...")
        urlretrieve(URL, DATA_PATH)

    with np.load(DATA_PATH) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")
    print(f"Pixel range: {x_train.min()} to {x_train.max()}")
    print(f"Classes: {np.unique(y_train)}")

    class_counts = np.bincount(y_train, minlength=10)
    print(f"Training examples per class: {class_counts}")

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for image, label, axis in zip(x_train[:10], y_train[:10], axes.ravel()):
        axis.imshow(image, cmap="gray")
        axis.set_title(f"Label: {label}")
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
