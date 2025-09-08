<!-- # PyTorch High Order AutoDifferentiator (thoad) -->

<!-- # $\text{{Py}{\color{#EE4C2C}T}{orch }{\color{#EE4C2C}H}{igh }{\color{#EE4C2C}O}{rder }{\color{#EE4C2C}A}{utomatic }{\color{#EE4C2C}D}{ifferentiation}}$ -->
<img src="title.svg" alt="Logo" width="100%"/>

<!--
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://img.shields.io/pypi/v/thoad.svg?label=PyPi&color=blue)](https://pypi.org/project/thoad/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-%23C2A000.svg)](https://github.com/python/cpython)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-%23EE4C2C.svg?)](https://github.com/pytorch/pytorch)
[![black](https://img.shields.io/badge/code%20style-black-202020.svg?style=flat)](https://github.com/psf/black)
-->
<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://img.shields.io/pypi/v/thoad.svg?label=PyPi&color=blue)](https://pypi.org/project/thoad/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-%23C2A000.svg)](https://github.com/python/cpython)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-%23EE4C2C.svg?)](https://github.com/pytorch/pytorch)
[![black](https://img.shields.io/badge/code%20style-black-202020.svg?style=flat)](https://github.com/psf/black)

</div>

<br>

> \[!NOTE]
> This package is still in an experimental stage. It may exhibit unstable behavior or produce unexpected results, and is subject to possible future structural modifications.

<br>

# About

**thoad** is a lightweight reverse-mode automatic differentiation engine written entirely in Python that works over PyTorch’s computational graph to compute **high order partial derivatives**. Unlike PyTorch’s native autograd - which is limited to first-order native partial derivatives - **thoad** is able to performantly propagate arbitray-order derivatives throughout the graph, enabling more advanced derivative-based computations.

## Core Features

- **Python 3.12+**: thoad is implemented in Python 3.12, and its compatible with any higher Python version.
- **Built on PyTorch**: thoad uses PyTorch as its only dependency. It is **compatible with [+70](https://github.com/mntsx/thoad/blob/master/src/thoad/differentiation/initialization/mapping.py) PyTorch operator backwards**.
- **Arbitrary-Order Differentiation**: thoad can compute arbitrary-order partial derivatives - including **cross node derivatives**.
- **Adoption of the PyTorch Computational Graph**: thoad integrates with PyTorch tensors by adopting their internally traced subgraphs.
- **High Performance**: thoad hessian comp time scales asymptotically better than [torch.autograd](https://docs.pytorch.org/docs/stable/autograd.html)'s, remaining closer to [jax.jet](https://docs.jax.dev/en/latest/jax.experimental.jet.html) performance.
- **Non-Sequential Graph Support**: Unlike jax.jet, thoad supports differentiation on arbitrary graph topologies, not only sequentials.
- **Non-Scalar Differentiation**: Unlike `torch.Tensor.backward`, thoad allows launching differentiation from non-scalar tensors.
- **Support for Backward Hooks**: thoad allows registering backward hooks for dynamic tuning of propagated high-order derivatives.
- **Diagonal Optimization**: thoad detects and avoids duplication of cross diagonal dimensions during back-propagation.
- **Symmetry Optimization**: Leveraging Schwarz’s theorem, thoad removes redundant derivative block computations.

## Installation

**thoad** can be installed either from PyPI or directly from the GitHub repository.

- **From PyPI**

  ```bash
  pip install thoad
  ```

- **From GitHub**
  Install directly with `pip` (fetches the latest from the `main` branch):

  ```bash
  pip install git+https://github.com/mntsx/thoad.git
  ```

  Or, if you prefer to clone and install in editable mode:

  ```bash
  git clone https://github.com/mntsx/thoad.git
  cd thoad
  pip install -e .
  ```

<br>

---

# Using the Package

**thoad** exposes two primary interfaces for computing high-order derivatives:

1. **`thoad.backward`**: a function-based interface that closely resembles `torch.Tensor.backward`. It provides a quick way to compute high-order pertial derivatives without needing to manage an explicit controller object, but it offers only the core functionality (derivative computation and storage).
2. **`thoad.Controller`**: a class-based interface that wraps the output tensor’s subgraph in a controller object. In addition to performing the same high-order backward pass, it gives access to advanced features such as fetching specific mixed partials, inspecting batch-dimension optimizations, overriding backward-function implementations, retaining intermediate partials, and registering custom hooks.

<br>

## **`thoad.backward`**

The `thoad.backward` function computes high-order partial derivatives of a given output tensor and stores them in each leaf tensor’s `.hgrad` attribute.

**Arguments**:

- **`tensor`**: A PyTorch tensor from which to start the backward pass. This tensor must have `require_grad=True `and be part of a differentiable graph.

- **`order`**: A positive integer specifying the maximum order of derivatives to compute.

- **`gradient`**: A tensor with the same shape as `tensor` to seed the vector-Jacobian product (i.e., custom upstream gradient). If omitted, the default is used.

- **`crossings`**: A boolean flag (default=`False`). If set to `True`, mixed partial derivatives (i.e., derivatives that involve more than one distinct leaf tensor) will be computed.

- **`groups`**: An iterable of disjoint groups of leaf tensors. When `crossings=False`, only those mixed partials whose participating leaf tensors all lie within a single group will be calculated. If `crossings=True` and `groups` is provided, a *ValueError* will be raised (they are mutually exclusive).

- **`keep_batch`**: A boolean flag (default=`False`) that controls how output dimensions are organized in the computed derivatives.

  - **When `keep_batch=False`:**
    The derivative preserves one first flattened "primal" axis, followed by each original partial shape, sorted in differentiation order. Concretelly:

    - A single "primal" axis that contains every element of the graph output tensor (flattened into one dimension).
    - A group of axes per derivative order, each matching the shape of the respective differentially targeted tensor.

    For an N-th order derivative of a leaf tensor with `input_numel` elements and an output with `output_numel` elements, the deerivative shape is:

    - **Axis 1:** indexes all `output_numel` outputs
    - **Axes 2…(N+1):** each indexes all `input_numel` inputs

  - **When `keep_batch=True`:**
    The derivative shape follows the same ordering as in the previous case, but includes a series of "independent dimensions" immediately after the "primal" axis.

    - **Axis 1** flattens all elements of the output tensor (size = `output_numel`).
    - **Axes 2...(k+i)** correspond to dimensions shared by multiple input tensors and treated independently throughout the graph. These are dimensions that are only operated on element-wise (e.g. batch dimensions).
    - **Axes (k+i+1)...(k+N+1)** each flatten all `input_numel` elements of the leaf tensor, one axis per derivative order.

- **`keep_schwarz`**: A boolean flag (default=`False`). If `True`, symmetric (Schwarz) permutations are retained explicitly instead of being canonicalized/reduced—useful for debugging or inspecting non-reduced layouts.

**Returns**:

- An instance of `thoad.Controller` wrapping the same tensor and graph.

---

### Executing Autodifferentiation via thoad.backward

```python
import torch
import thoad
from torch.nn import functional as F

### Normal PyTorch workflow
X = torch.rand(size=(10,15), requires_grad=True)
Y = torch.rand(size=(15,20), requires_grad=True)
Z = F.scaled_dot_product_attention(query=X, key=Y.T, value=Y.T)

### Call thoad backward
order = 2
thoad.backward(tensor=Z, order=order)

### Checks
# check derivative shapes
for o in range(1, 1 + order):
    assert X.hgrad[o - 1].shape == (Z.numel(), *(o * tuple(X.shape)))
    assert Y.hgrad[o - 1].shape == (Z.numel(), *(o * tuple(Y.shape)))
# check first derivatives (jacobians)
fn = lambda x, y: F.scaled_dot_product_attention(x, y.T, y.T)
J = torch.autograd.functional.jacobian(fn, (X, Y))
assert torch.allclose(J[0].flatten(), X.hgrad[0].flatten(), atol=1e-6)
assert torch.allclose(J[1].flatten(), Y.hgrad[0].flatten(), atol=1e-6)
# check second derivatives (hessians)
fn = lambda x, y: F.scaled_dot_product_attention(x, y.T, y.T).sum()
H = torch.autograd.functional.hessian(fn, (X, Y))
assert torch.allclose(H[0][0].flatten(), X.hgrad[1].sum(0).flatten(), atol=1e-6)
assert torch.allclose(H[1][1].flatten(), Y.hgrad[1].sum(0).flatten(), atol=1e-6)
```

<br>

## **`thoad.Controller`**

The `Controller` class wraps a tensor’s backward subgraph in a controller object, performing the same core high-order backward pass as `thoad.backward` while exposing advanced customization, inspection, and override capabilities.

### Instantiation

Use the constructor to create a controller for any tensor requiring gradients:

```python
controller = thoad.Controller(tensor=GO)  # takes graph output tensor
```

- **`tensor`**: A PyTorch `Tensor` with `requires_grad=True` and a non-`None` `grad_fn`.

### Properties

- **`.tensor → Tensor`**
  The output tensor underlying this controller.
  **Setter**: Replaces the tensor (after validation), rebuilds the internal computation graph, and invalidates any previously computed derivatives.

- **`.compatible → bool`**
  Indicates whether every backward function in the tensor’s subgraph has a supported high-order implementation. If **`False`**, some derivatives may fall back or be unavailable.

- **`.index → Dict[Type[torch.autograd.Function], Type[ExtendedAutogradFunction]]`**
  A mapping from base PyTorch `autograd.Function` classes to thoad’s `ExtendedAutogradFunction` implementations.
  **Setter**: Validates and injects your custom high-order extensions.

### Core Methods

#### **`.backward(order, gradient=None, crossings=False, groups=None, keep_batch=False, keep_schwarz=False) → None`**

Performs the high-order backward pass up to the specified derivative `order`, storing all computed partials in each leaf tensor’s `.hgrad` attribute.

- **`order`** (`int > 0`): maximum derivative order.
- **`gradient`** (`Optional[Tensor]`): custom upstream gradient with the same shape as `controller.tensor`.
- **`crossings`** (`bool`, default `False`): If `True`, mixed partial derivatives across different leaf tensors will be computed.
- **`groups`** (`Optional[Iterable[Iterable[Tensor]]]`, default `None`): When `crossings=False`, restricts mixed partials to those whose leaf tensors all lie within a single group. If `crossings=True` and `groups` is provided, a *ValueError* is raised.
- **`keep_batch`** (`bool`, default `False`): controls whether independent output axes are kept separate (batched) or merged (flattened) in stored/retrieved derivatives.
- **`keep_schwarz`** (`bool`, default `False`): if `True`, retains symmetric permutations explicitly (no Schwarz reduction).

#### **`.display_graph() → None`**

Prints a tree representation of the tensor’s backward subgraph. Supported nodes are shown normally; unsupported ones are annotated with `(not supported)`.

#### **`.register_backward_hook(variables: Sequence[Tensor], hook: Callable) → None`**

Registers a user-provided `hook` to run during the backward pass whenever derivatives for any of the specified leaf `variables` are computed.

- **`variables`** (`Sequence[Tensor]`): Leaf tensors to monitor.
- **`hook`** (`Callable[[Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]], dict[AutogradFunction, set[Tensor]]], Tuple[Tensor, Tuple[Shape, ...], Tuple[Indep, ...]]]`):
  Receives the current `(Tensor, shapes, indeps)` plus contextual info, and must return the modified triple.

#### **`.require_grad_(variables: Sequence[Tensor]) → None`**

Marks the given leaf `variables` so that all intermediate partials involving them are retained, even if not required for the final requested derivatives. Useful for inspecting or re-using higher-order intermediates.

#### **`.fetch_hgrad(variables: Sequence[Tensor], keep_batch: bool = False, keep_schwarz: bool = False) → Tuple[Tensor, Tuple[Tuple[Shape, ...], Tuple[Indep, ...], VPerm]]`**

Retrieves the precomputed high-order partial corresponding to the ordered sequence of leaf `variables`.

- **`variables`** (`Sequence[Tensor]`): the leaf tensors whose mixed partial you want.
- **`keep_batch`** (`bool`, default `False`): if `True`, each independent output axis remains a separate batch dimension in the returned tensor; if `False`, independent axes are distributed/merged into derivative dimensions.
- **`keep_schwarz`** (`bool`, default `False`): if `True`, returns derivatives retaining symmetric permutations explicitly.

Returns a pair:

1. **Derivative tensor**: the computed partial derivatives, shaped according to output and input dimensions (respecting `keep_batch`/`keep_schwarz`).
2. **Metadata tuple**

   - **Shapes** (`Tuple[Shape, ...]`): the original shape of each leaf tensor.
   - **Indeps** (`Tuple[Indep, ...]`): for each variable, indicates which output axes remained independent (batch) vs. which were merged into derivative axes.
   - **VPerm** (`Tuple[int, ...]`): a permutation that maps the internal derivative layout to the requested `variables` order.

Use the combination of independent-dimension info and shapes to reshape or interpret the returned derivative tensor in your workflow.

---

### Executing Autodifferentiation via thoad.Controller.backward

```python
import torch
import thoad
from torch.nn import functional as F

### Normal PyTorch workflow
X = torch.rand(size=(10,15), requires_grad=True)
Y = torch.rand(size=(15,20), requires_grad=True)
Z = F.scaled_dot_product_attention(query=X, key=Y.T, value=Y.T)

### Instantiate thoad controller and call backward
order = 2
controller = thoad.Controller(tensor=Z)
controller.backward(order=order, crossings=True)

### Fetch Partial Derivatives
# fetch X and Y 2nd order derivatives
partial_XX, _ = controller.fetch_hgrad(variables=(X, X))
partial_YY, _ = controller.fetch_hgrad(variables=(Y, Y))
assert torch.allclose(partial_XX, X.hgrad[1])
assert torch.allclose(partial_YY, Y.hgrad[1])
# fetch cross derivatives
partial_XY, _ = controller.fetch_hgrad(variables=(X, Y))
partial_YX, _ = controller.fetch_hgrad(variables=(Y, X))
```

---

> \[!TIP]
> Consult the user guide notebook for an in-depth overview of features with practical examples:
> [user guide](https://github.com/mntsx/thoad/blob/master/examples/user_guide.ipynb)

<br>

# More About the Package

## Future Plans

The following outlines the planned future developments and improvements for thoad:
  
- **Extend Backward Functionality**  
  Develop further backprop capabilities to improve PyTorch integration supporting a broad subset of the most commonly used operators.

- **Advanced Optimization Framework**  
  Build an optimization module inspired by the design of `torch.optim`, with full support for higher-order derivatives and flexible optimizer composition.

- **PyTorch Integration**  
  It would be exciting to eventually fully-integrate the package into the PyTorch framework, although it's unlikely to happen, since ensuring their coordinated stability would require relevant adjustments to the mentioned library. Specifically:
  - Providing it with a more comprehensive tool for accessing controllers' contextual information.
  - Improving the accessibility to the type signatures of the backward functions.

## License

**This project** is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
See the [LICENSE](LICENSE) file for details.

**PyTorch** is distributed under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-Clause).  
See PyTorch’s own [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) file for its full terms.

## How to Cite

If you use **thoad** in your work, please consider citing it with the following BibTeX entry:

```bibtex
@Misc{thoad2025,
  title        = {thoad: PyTorch High Order Reverse-Mode Automatic Differentiation},
  howpublished = {\url{https://github.com/mntsx/thoad}},
  year         = {2025}
}
```
