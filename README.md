# PyTorch High Order AutoDifferentiator (thoad)

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-yellow.svg)](https://github.com/python)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-%23EE4C2C.svg?)](https://github.com/pytorch)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<br>

> \[!NOTE]
> This package is currently in an experimental release. It may exhibit unstable behavior or produce unexpected results, and is subject to possible minor structural modifications in the future.

<br>

## Introduction

**thoad** is a lightweight autodifferentiation engine written entirely in Python that works over PyTorch’s computational graph to compute **high order partial derivatives**. Unlike PyTorch’s native autograd - which is limited to first-order derivatives - **thoad** is able to performantly propagate arbitray-order derivatives throughout the graph, enabling more advanced gradient-based computations.

<br>

## Future Plans
  
- **Extend Backward Functionality**  
  Develop further backprop capabilities to improve operator integration and broaden support for PyTorch’s full operator set.

- **Advanced Optimization Framework**  
  Build an optimization module inspired by the design of `torch.optim`, with full support for higher-order gradients and flexible optimizer composition.


<br>

## License

**This project** is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
See the [LICENSE](LICENSE) file for details.

**PyTorch** is distributed under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-Clause).  
See PyTorch’s own [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) file for its full terms.
