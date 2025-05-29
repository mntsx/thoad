# PyTorch High Order AutoDifferentiator (thoad)

[![LICENSE: MIT](https://img.shields.io/badge/LICENSE-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://github.com/python)
[![built with PyTorch](https://img.shields.io/badge/built%20with-PyTorch-%23EE4C2C.svg?)](https://github.com/pytorch)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<br>

> Note: This repository is still under development. Some of its functionalities may exhibit incomplete or incorrect behavior.

## Introduction

**thoad** is a lightweight, pure-Python autodifferentiation engine that operates over PyTorch’s computational graph to compute **arbitrary-order partial derivatives**. Unlike PyTorch’s native autograd - which stops at first-order derivatives - **thoad** propagates higher-order derivatives throughout the graph, enabling more advanced gradient-based computations.
