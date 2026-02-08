# MLP From Scratch — Cats vs Dogs Binary Classification

A fully connected deep neural network implemented entirely in NumPy — no PyTorch, no TensorFlow, no autograd. Every component (forward pass, backpropagation, weight updates) is written by hand to understand what's actually happening under the hood.

## What this is

A modular multi-layer perceptron trained on 15,000 images (can add more) from the [microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs) dataset. The network takes a flattened 64×64 RGB image (12,288 features) and outputs a binary prediction: cat (0) or dog (1).

## Architecture

The network is fully configurable at instantiation — number of layers, neurons per layer, learning rate, and activation function per layer are all parameters:

```python
model = NN(
    L=3,
    nx=np.array([10, 9, 1]),
    learning_rate=0.001,
    activation_funcs=[ReLU, ReLU, Sigmoid]
)
```

```
Input (12,288) → Dense(10, ReLU) → Dense(9, ReLU) → Dense(1, Sigmoid) → BCE Loss
```

## What's implemented from scratch

- **Data preprocessing**: HuggingFace loading, RGB conversion (handles grayscale/RGBA edge cases), resize to 64×64, normalization to [0,1], flattening
- **He initialization**: weights scaled by √(2/n_in) to prevent vanishing/exploding gradients in ReLU networks
- **Forward pass**: Z = W·A_prev + b → activation, computed layer by layer
- **Binary cross-entropy loss**: L = -1/m · Σ[Y·ln(A) + (1-Y)·ln(1-A)]
- **Backpropagation**: full chain rule gradient computation
  - Output layer: dA from BCE derivative, dZ via sigmoid derivative
  - Hidden layers: dA propagated through transposed weights, dZ via ReLU derivative
  - Weight gradients: dW = 1/m · dZ · A_prev^T
  - Bias gradients: db = mean(dZ, axis=1)
- **Gradient descent**: vanilla SGD weight updates
- **Dead neuron monitoring**: tracks percentage of ReLU neurons outputting zero per layer

## Activation functions

Implemented as plug-and-play dictionaries with forward and derivative:

| Function | Forward | Derivative |
|----------|---------|------------|
| Sigmoid | 1/(1+e^(-z)) | a(1-a) |
| ReLU | max(0, z) | 1 if z > 0, else 0 |

## Key dimensions (convention: neurons × examples)

| Tensor | Shape |
|--------|-------|
| X (input) | (m, 12288) — transposed to (12288, m) in forward pass |
| Y (labels) | (1, m) |
| W[l] | (n_l, n_{l-1}) |
| b[l] | (n_l, 1) — broadcast across m examples |
| Z[l], A[l] | (n_l, m) |

## Bugs I found and fixed along the way

This project was a debugging exercise as much as an implementation one. Major issues encountered:

1. **dtype=object contamination** — using `np.empty(L, dtype=object)` for storing arrays caused `np.exp` to fail silently. Fixed by using Python lists.
2. **Element-wise vs matrix multiply confusion** — `dZ = dA * activation'(A)` is element-wise (`*`), not `np.dot`. Took a while to internalize when to use which.
3. **Broadcasting shape mismatch** — Y as shape `(15000,)` with A as `(1, 15000)` caused NumPy to broadcast into a `(15000, 15000)` matrix in the loss computation, inflating the cost to ~8.0 instead of ~0.693. Fixed with `Y.reshape(1, -1)`.
4. **Double 1/m normalization** — dividing by m in both dA and dW effectively divided gradients by m² = 225,000,000. Gradients were microscopic and the network barely learned.
5. **Dying ReLU in deep networks** — original 9-layer architecture had cascading dead neurons. Reduced to 3 layers to get gradient flow working.

## How to run

```bash
pip install datasets pillow numpy matplotlib
```

```python
# Load and preprocess
from datasets import load_dataset
ds = load_dataset("microsoft/cats_vs_dogs")
# (see notebook for full preprocessing pipeline)

# Train
model = NN(L=3, nx=np.array([10,9,1]), learning_rate=0.001,
           activation_funcs=[ReLinearUnit, ReLinearUnit, logistic_regression])
model.random_weights(n0=12288)

for i in range(100):
    model.forward_pass(X, Y)
    model.backward_pass(X, Y)
```
## Result I got with L = 3, corresponding hidden layers size = [10,9,1], m = 15,000, iterations = 40, learning rate = .001, activation functions = [Relu, Relu, logistic regression] 
<img width="567" height="435" alt="image" src="https://github.com/user-attachments/assets/5a5a6495-001b-4e39-a348-63017f8420be" />


## What's next

- Train/test split and accuracy metrics
- Learning rate tuning and more iterations
- Deeper architectures once the base network converges
- Transition to PyTorch to compare with manual implementation
- Apply similar architecture to financial time series data


## Stack

Python 3 · NumPy · Pillow · HuggingFace Datasets · Matplotlib
