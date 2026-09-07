import numpy as np

# Two examples, each with three input features.
X = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

# Three input features connected to two neurons.
W = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
    ]
)

# One bias for each neuron.
b = np.array([1.0, -1.0])

# Linear part.
Z = X @ W + b

# ReLU activation.
A = np.maximum(0.0, Z)

print("X shape:", X.shape)
print("W shape:", W.shape)
print("b shape:", b.shape)
print("Z shape:", Z.shape)
print("A shape:", A.shape)
print("Z:\n", Z)
print("A:\n", A)

assert X.shape == (2, 3)
assert W.shape == (3, 2)
assert b.shape == (2,)
assert Z.shape == (2, 2)
assert A.shape == (2, 2)
