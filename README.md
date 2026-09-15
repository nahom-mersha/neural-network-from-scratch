# Neural Network From Scratch for Handwritten Digits

An educational fully connected neural network built from scratch with NumPy
for handwritten-digit classification.

The project begins with binary classification of digits `1` and `2`, then
expands to all ten MNIST classes. It focuses on vectorized computation, tensor
shapes, forward propagation, backpropagation, and gradient-based learning
without using PyTorch.

## Project status

Complete.

The project includes:

- manual forward propagation and backpropagation;
- NumPy dense layers;
- ReLU, sigmoid, and softmax activations;
- cross-entropy loss;
- mini-batch gradient descent;
- train, validation, and test evaluation;
- analytical and numerical gradient checking;
- confusion-matrix error analysis;
- model saving and loading;
- a Streamlit drawing application;
- automated tests and continuous integration.

## Architecture

```text
28 × 28 image
    ↓
784 flattened pixel values
    ↓
Dense layer: 784 → 32
    ↓
ReLU
    ↓
Dense layer: 32 → 10
    ↓
Softmax
    ↓
Probabilities for digits 0–9
```

The forward pass is:

```text
Z1 = X @ W1 + b1
A1 = ReLU(Z1)
Z2 = A1 @ W2 + b2
P  = softmax(Z2)
```

## Project 7 connection

Project 7 implemented automatic differentiation for scalar values:

```text
scalar values → scalar derivatives → computational graphs
```

Project 8 extends the same chain-rule idea to arrays and matrices:

```text
arrays and matrices → vectorized derivatives → neural-network training
```

The central backpropagation rule remains:

```text
parent gradient += local derivative × upstream gradient
```

## Results

Final configuration:

```text
Input size:     784
Hidden size:    32
Output size:    10
Epochs:         10
Batch size:     128
Learning rate:  0.1
Random seed:    42
```

Final results:

```text
Training accuracy:   96.38%
Validation accuracy: 95.60%
Test accuracy:       95.78%
```

## Learning-rate experiment

A small controlled experiment compared learning rates on XOR. The dataset,
architecture, batch size, epoch count, and random seed remained fixed.

| Learning rate | Final loss | Final accuracy |
|---:|---:|---:|
| 0.001 | 0.8703 | 75% |
| 0.01 | 0.6267 | 50% |
| 0.1 | 0.0853 | 100% |

This was an illustrative experiment rather than a general hyperparameter
search. It demonstrated that the learning rate strongly affects training.

## Installation

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

To run the Streamlit application, install the app dependencies too:

```bash
python -m pip install -e ".[dev,app]"
```

## Running the tests

```bash
python -m pytest
```

## Training the model

```bash
python -m scripts.train_mnist
```

The trained model is saved to:

```text
artifacts/mnist_model.npz
```

Training metrics are saved to:

```text
artifacts/training_metrics.json
```

## Running the Streamlit application

```bash
python -m streamlit run app.py
```

The application performs this pipeline:

```text
user drawing
→ grayscale conversion
→ color inversion
→ cropping and centering
→ resizing to 28 × 28
→ normalization to [0, 1]
→ flattening to (1, 784)
→ model forward pass
→ prediction and confidence
```

## Model persistence

The NumPy archive contains the learned parameters:

```text
dense1_weights
dense1_biases
dense2_weights
dense2_biases
```

The application recreates the same architecture and loads these parameters
for inference.

## Limitations

The model performs well on MNIST-style test images but may perform worse on
drawings made in the Streamlit canvas. User drawings can differ in stroke
thickness, position, scale, and handwriting style.

A convolutional neural network and stronger data augmentation would likely
improve robustness to different drawing styles. Those improvements are
reserved for a later PyTorch project.

## Main learning outcome

This project connects the mathematics of neural networks to a working AI
system:

```text
data preparation
→ vectorized forward pass
→ manual backpropagation
→ parameter updates
→ evaluation
→ model persistence
→ user-facing prediction application
```
