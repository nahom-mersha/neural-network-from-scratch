from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from digit_nn.model import MulticlassClassifier

MODEL_PATH = Path("artifacts/mnist_model.npz")
DATA_PATH = Path("data/raw/mnist.npz")

IMAGE_SIZE = 28
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_SIZE = 32
NUMBER_OF_CLASSES = 10


@st.cache_data
def load_digit_samples() -> dict[int, np.ndarray]:
    with np.load(DATA_PATH) as data:
        images = data["x_train"]
        labels = data["y_train"]

    samples: dict[int, np.ndarray] = {}

    for digit in range(NUMBER_OF_CLASSES):
        index = np.flatnonzero(labels == digit)[0]
        samples[digit] = images[index]

    return samples


@st.cache_resource
def load_model() -> MulticlassClassifier:
    model = MulticlassClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=NUMBER_OF_CLASSES,
        seed=42,
    )

    with np.load(MODEL_PATH) as saved:
        model.dense1.weights = saved["dense1_weights"]
        model.dense1.biases = saved["dense1_biases"]
        model.dense2.weights = saved["dense2_weights"]
        model.dense2.biases = saved["dense2_biases"]

    return model


def preprocess_drawing(image_data: np.ndarray) -> np.ndarray:
    """Convert a canvas drawing into one normalized MNIST-style input."""

    # Convert RGB channels into one grayscale image.
    grayscale = image_data[:, :, :3].mean(axis=2)

    # Canvas: black digit on white background.
    # MNIST: white digit on black background.
    digit = 255.0 - grayscale

    # Find the drawn digit.
    mask = digit > 20

    if not np.any(mask):
        return np.zeros((1, INPUT_SIZE), dtype=np.float32)

    coordinates = np.argwhere(mask)

    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)

    # Crop around the digit.
    cropped = digit[top : bottom + 1, left : right + 1]

    # Scale the digit to fit inside approximately 20 × 20 pixels.
    height, width = cropped.shape
    scale = 20 / max(height, width)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    # Create a black 28 × 28 MNIST-style image.
    mnist_image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
    )

    # Center the resized digit.
    top_offset = (IMAGE_SIZE - new_height) // 2
    left_offset = (IMAGE_SIZE - new_width) // 2

    mnist_image[
        top_offset : top_offset + new_height,
        left_offset : left_offset + new_width,
    ] = np.asarray(resized)

    # Normalize pixels to the range [0, 1].
    normalized = mnist_image.astype(np.float32) / 255.0

    # Add batch dimension:
    # (28, 28) → (1, 784)
    return normalized.reshape(1, INPUT_SIZE)


def predict_digit(
    model: MulticlassClassifier,
    inputs: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Return prediction, confidence, and all class probabilities."""

    probabilities = model.forward(inputs)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    return prediction, confidence, probabilities


st.set_page_config(
    page_title="Handwritten Digit Classifier",
    page_icon="🔢",
    layout="wide",
)

st.title("Handwritten Digit Classifier")
st.write("Draw a digit from 0 to 9 and let the neural network classify it.")

if not DATA_PATH.exists():
    st.error(f"MNIST data file not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


st.subheader("MNIST training examples")

samples = load_digit_samples()
sample_columns = st.columns(5)

for digit in range(NUMBER_OF_CLASSES):
    column = sample_columns[digit % 5]

    with column:
        st.image(
            samples[digit],
            caption=f"Digit {digit}",
            width=110,
        )


st.divider()

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Draw a digit")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="digit_canvas",
    )

with right_column:
    st.subheader("Processed 28 × 28 image")

    if canvas_result.image_data is None:
        st.info("Draw a digit to see its processed image.")
        processed_inputs = None
    else:
        processed_inputs = preprocess_drawing(canvas_result.image_data)

        processed_image = processed_inputs.reshape(
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        st.image(
            processed_image,
            caption="This is the image given to the model",
            width=280,
            clamp=True,
        )

        st.caption(f"Model input shape: {processed_inputs.shape}")


st.divider()

if st.button(
    "Predict",
    type="primary",
    use_container_width=False,
):
    if processed_inputs is None:
        st.warning("Please draw a digit first.")
        st.stop()

    model = load_model()

    prediction, confidence, probabilities = predict_digit(
        model,
        processed_inputs,
    )

    st.subheader(f"Prediction: {prediction}")
    st.metric("Confidence", f"{confidence:.2%}")

    st.subheader("Probability distribution")

    probability_table = {
        "Digit": [f"Digit {digit}" for digit in range(NUMBER_OF_CLASSES)],
        "Probability": [f"{probability:.2%}" for probability in probabilities],
    }

    st.dataframe(
        probability_table,
        hide_index=True,
        use_container_width=True,
    )

    st.bar_chart(
        probabilities,
        x_label="Digit",
        y_label="Probability",
    )
