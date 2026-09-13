from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from digit_nn.model import MulticlassClassifier

MODEL_PATH = Path("artifacts/mnist_model.npz")


@st.cache_resource
def load_model() -> MulticlassClassifier:
    model = MulticlassClassifier(
        input_size=784,
        hidden_size=32,
        output_size=10,
        seed=42,
    )

    saved = np.load(MODEL_PATH)

    model.dense1.weights = saved["dense1_weights"]
    model.dense1.biases = saved["dense1_biases"]
    model.dense2.weights = saved["dense2_weights"]
    model.dense2.biases = saved["dense2_biases"]

    return model


def preprocess_drawing(image_data: np.ndarray) -> np.ndarray:
    # Use the RGB channels and convert black ink on white background
    # into bright digit pixels on a dark background like MNIST.
    grayscale = image_data[:, :, :3].mean(axis=2)
    digit = 255.0 - grayscale

    image = Image.fromarray(digit.astype(np.uint8))
    image = image.resize((28, 28))

    normalized = np.asarray(image, dtype=np.float32) / 255.0

    return normalized.reshape(1, 784)


st.title("Handwritten Digit Classifier")
st.write("Draw a digit from 0 to 9, then click Predict.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    return_image_data=True,
)

if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
        st.stop()

    model = load_model()
    inputs = preprocess_drawing(canvas_result.image_data)

    probabilities = model.forward(inputs)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence:.2%}")

    st.subheader("Probability distribution")

    st.bar_chart(probabilities)

    st.write(
        {
            f"Digit {digit}": f"{probability:.2%}"
            for digit, probability in enumerate(probabilities)
        }
    )
