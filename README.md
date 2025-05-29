# PyTorch High Order AutoDifferentiator (thoad)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://github.com/python)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-%23EE4C2C.svg?)](https://github.com/pytorch)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<br>

> Note: This repository is still under development. Some of its functionalities may exhibit incomplete or incorrect behavior.

## Introduction

**thoad** is a lightweight autodifferentiation engine written entirely in Python that works over PyTorch’s computational graph to compute **high order partial derivatives**. Unlike PyTorch’s native autograd - which is limited to first-order derivatives - **thoad** is able to performantly propagate arbitray-order derivatives throughout the graph, enabling more advanced gradient-based computations.
