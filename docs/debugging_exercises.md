# Step 12 — Deliberate Debugging Exercises

These exercises intentionally describe common neural-network bugs and their symptoms.

## 1. Wrong transpose in the dense layer

Correct:

```python
dX = dZ @ W.T
```

Possible bug:

```python
dX = dZ @ W
```

Typical symptom:

```text
shape mismatch error
```

Reason:

```text
dZ:  (batch_size, output_size)
W.T: (output_size, input_size)
dX:  (batch_size, input_size)
```

---

## 2. Wrong sign in the softmax-cross-entropy gradient

Correct:

```python
dZ = (P - Y) / batch_size
```

Possible bug:

```python
dZ = (Y - P) / batch_size
```

Typical symptom:

```text
the model updates in the wrong direction
loss increases instead of decreases
```

---

## 3. Forgetting the batch-size division

Correct:

```python
dZ = (P - Y) / batch_size
```

Possible bug:

```python
dZ = P - Y
```

Typical symptom:

```text
gradients are too large
training becomes unstable
loss may oscillate or become NaN
```

---

## 4. Numerically unstable softmax

Correct:

```python
shifted_logits = logits - np.max(
    logits,
    axis=1,
    keepdims=True,
)
```

Possible bug:

```python
exponentials = np.exp(logits)
```

Typical symptom:

```text
overflow warning
NaN probabilities
NaN loss
```

The shift does not change the final probabilities, but prevents very large exponentials.

---

## 5. Wrong ReLU derivative

Correct:

```python
dZ1 = dA1 * (Z1 > 0)
```

Possible bug:

```python
dZ1 = dA1 * (Z1 < 0)
```

Typical symptom:

```text
active neurons stop learning
loss does not decrease properly
```

ReLU passes gradients only where:

```text
Z1 > 0
```

---

## 6. Debugging procedure

When the network behaves incorrectly:

1. Check tensor shapes.
2. Check for NaN or infinite values.
3. Check whether the loss decreases.
4. Check whether probabilities sum to one.
5. Run numerical gradient checking.
6. Inspect the sign and scale of gradients.
7. Compare the implementation with the mathematical formula.

Useful checks:

```python
assert np.all(np.isfinite(probabilities))
assert np.isfinite(loss)
assert probabilities.shape == targets.shape
```

The main lesson is:

> A neural network can have correct-looking code but still fail because of one wrong sign, transpose, axis, or scaling factor.
