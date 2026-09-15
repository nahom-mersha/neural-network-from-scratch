# Learning-Rate XOR Experiment

## Script

This experiment is implemented in:

```text
scripts/compare_learning_rates.py
```

The script compares three learning rates on a small XOR classification problem.

## Experimental design

The script uses the same:

- XOR dataset;
- neural-network architecture;
- hidden-layer size of 8;
- batch size of 4;
- epoch count of 300;
- random seed of 42.

Only the learning rate changes:

```python
learning_rates = [0.001, 0.01, 0.1]
```

The script trains one model for each learning rate and prints:

- final training loss;
- final training accuracy;
- training time;
- best configuration by final training loss.

## Results

| Learning rate | Final loss | Final accuracy | Time |
|---:|---:|---:|---:|
| 0.001 | 0.8703 | 75% | 0.0376 s |
| 0.01 | 0.6267 | 50% | 0.0336 s |
| 0.1 | 0.0853 | 100% | 0.0285 s |

## Interpretation

A learning rate of `0.001` produced very small parameter updates, so the
model learned slowly.

A learning rate of `0.01` did not converge successfully in this particular
run.

A learning rate of `0.1` produced the lowest final loss and reached 100%
accuracy on the XOR training set.

## Conclusion

This experiment shows that the learning rate strongly affects whether and how
quickly the network learns. However, this is an illustrative controlled
experiment, not a general hyperparameter search.

The result applies only to this small XOR dataset, this architecture, this
initialization seed, and these training settings. Therefore, `0.1` should not
be considered the universally optimal learning rate.

The experiment demonstrates the importance of changing one variable at a time
and observing its effect on training.
