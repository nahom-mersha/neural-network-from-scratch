# Neural Network From Scratch for Handwritten Digits

An educational NumPy neural network built from scratch for handwritten-digit classification.

The project begins by distinguishing handwritten digits `1` and `2`, then expands to all ten MNIST classes. The implementation focuses on understanding vectorized computation, tensor shapes, forward propagation, backpropagation, and gradient-based learning without using PyTorch.

## AI-assisted learning

This is an AI-assisted learning project. I directed the work, reviewed and tested the implementation, and documented the concepts I learned.

## Planned features

- Data preparation and inspection
- Train, validation, and test splits
- Image normalization and visualization
- NumPy dense layers
- ReLU and sigmoid activations
- Softmax and cross-entropy loss
- Manual forward propagation
- Manual backpropagation
- Mini-batch gradient descent
- Parameter initialization
- Analytical versus numerical gradient checking
- Architecture and hyperparameter experiments
- Deliberate gradient-debugging exercises
- A drawing application for handwritten-digit predictions

## Engineering features

- `src` package layout
- `pytest` tests
- Ruff formatting and linting
- Logging
- YAML configuration
- GitHub Actions continuous integration
- Docker support

## Quick Start

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Build the Docker image:

```bash
docker build -t neural-network-from-scratch .
```

Run the test suite in Docker:

```bash
docker run --rm neural-network-from-scratch
```

## Learning focus

Project 7 implemented automatic differentiation for scalar values. This project extends that understanding to arrays and matrices:

```text
Project 7: scalar values → scalar derivatives → computational graphs
Project 8: arrays and matrices → vectorized derivatives → neural-network training
```

The main goal is genuine understanding. The final code should be explainable from the data representation through the forward pass, backward pass, experiments, and user-facing prediction.

## Project status

This project is currently in the setup stage. Results and completed features will be added as they are implemented and verified.

