# MLP From Scratch & PyTorch — Cats vs Dogs Binary Classification

A fully connected deep neural network implemented **twice**: first entirely in NumPy (no autograd), then translated to PyTorch. Every component of the from-scratch version (forward pass, backpropagation, weight updates) is written by hand to understand what's actually happening under the hood.

## What this is

A modular multi-layer perceptron trained on 15,000 images from the [microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs) dataset. The network takes a flattened 64×64 RGB image (12,288 features) and outputs a binary prediction: cat (0) or dog (1).

---

## Part 1: NumPy From Scratch

### Architecture

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

### What's implemented from scratch

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

### Activation functions

Implemented as plug-and-play dictionaries with forward and derivative:

| Function | Forward | Derivative |
|----------|---------|------------|
| Sigmoid | 1/(1+e^(-z)) | a(1-a) |
| ReLU | max(0, z) | 1 if z > 0, else 0 |

### Key dimensions (convention: neurons × examples)

| Tensor | Shape |
|--------|-------|
| X (input) | (m, 12288) — transposed to (12288, m) in forward pass |
| Y (labels) | (1, m) |
| W[l] | (n_l, n_{l-1}) |
| b[l] | (n_l, 1) — broadcast across m examples |
| Z[l], A[l] | (n_l, m) |

### How to run (NumPy version)

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

### Result (NumPy)
L = 3, hidden layers = [10, 9, 1], m = 15,000, iterations = 100, lr = 0.001

<img width="567" height="435" alt="image" src="https://github.com/user-attachments/assets/5a5a6495-001b-4e39-a348-63017f8420be" />

---

## Part 2: PyTorch Translation

### Architecture

Same network, but using PyTorch's `nn.Sequential`:

```python
model = nn.Sequential(
    nn.Linear(12288, 10),
    nn.ReLU(),
    nn.Linear(10, 9),
    nn.ReLU(),
    nn.Linear(9, 1),
    nn.Sigmoid()
)
```

### Key differences from NumPy version

| Aspect | NumPy (from scratch) | PyTorch |
|--------|---------------------|---------|
| Gradient computation | Manual backprop formulas | `loss.backward()` (autograd) |
| Weight updates | `W -= lr * dW` | `optimizer.step()` |
| Memory management | Manual | Computation graph stored automatically |
| Data convention | (features, examples) | (examples, features) |

### How to run (PyTorch version)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import numpy as np

# Load and preprocess
ds = load_dataset("microsoft/cats_vs_dogs", split="train")

def preprocessing(example):
    size = (64, 64)
    img = example['image'].convert('RGB')
    img_resized = img.resize(size)
    img_processed = np.array(img_resized, dtype=np.float32) / 255
    example['image'] = img_processed.flatten()
    return example

images = ds.map(preprocessing, num_proc=4)['image']
labels = ds['labels']

# Prepare tensors
m = 15000
vector_images = np.array(images[:m]).squeeze()
tensor_images = torch.tensor(vector_images, dtype=torch.float32)
tensor_labels = torch.tensor(labels[:m], dtype=torch.float32).unsqueeze(1)  # Shape: (m, 1)

# Define model
model = nn.Sequential(
    nn.Linear(12288, 10),
    nn.ReLU(),
    nn.Linear(10, 9),
    nn.ReLU(),
    nn.Linear(9, 1),
    nn.Sigmoid()
)

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
loss_evolution = []
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(tensor_images)
    loss = loss_function(outputs, tensor_labels)
    loss.backward()
    optimizer.step()
    loss_evolution.append(loss.item())  # .item() to avoid memory leak
```

### Critical lesson learned: `loss.item()`

When tracking loss history, **never** do:
```python
loss_evolution.append(loss)  # BAD: stores tensor + entire computation graph
```

Always extract the scalar value:
```python
loss_evolution.append(loss.item())  # GOOD: stores just the float
```

Without `.item()`, each iteration stores the full computation graph in memory. After 100 epochs on 15,000 images, this crashes the kernel (and sometimes the whole machine).

---

## Bugs I found and fixed along the way

### NumPy version
1. **dtype=object contamination** — using `np.empty(L, dtype=object)` caused `np.exp` to fail silently
2. **Element-wise vs matrix multiply confusion** — `dZ = dA * activation'(A)` is element-wise (`*`), not `np.dot`
3. **Broadcasting shape mismatch** — Y as `(15000,)` with A as `(1, 15000)` broadcast into `(15000, 15000)`, inflating cost
4. **Double 1/m normalization** — divided by m² = 225,000,000 instead of m
5. **Dying ReLU** — 9-layer architecture had cascading dead neurons

### PyTorch version
1. **Shape mismatch warning** — output `(m, 1)` vs labels `(m,)` causes broadcasting issues. Fix: `tensor_labels.unsqueeze(1)`
2. **Memory leak from storing tensors** — appending `loss` instead of `loss.item()` crashed the kernel
3. **Confusing nn.ReLU syntax** — `nn.ReLU(12288, 10)` is wrong; need `nn.Linear(12288, 10)` then `nn.ReLU()`

### Result (PyTorch)
L = 3, hidden layers = [10, 9, 1], m = 15,000, iterations = 100, lr = 0.001
<img width="565" height="413" alt="image" src="https://github.com/user-attachments/assets/3baa4cf3-3ccb-44fa-9be6-ff4b43f4532f" />


---

## What's next

- Train/test split and accuracy metrics
- Mini-batch training with DataLoaders
- Learning rate scheduling
- Deeper architectures with batch normalization
- Apply to financial time series data (the real goal)

---

## Stack

Python 3 · NumPy · PyTorch · Pillow · HuggingFace Datasets · Matplotlib
