# PyTorch High Order AutoDifferentiator (thoad)

<br>

> Note: This repository is still under development. Some of its functionalities may exhibit incomplete or incorrect behavior.

## Introduction

**thoad** is a lightweight, pure-Python autodifferentiation engine that operates over PyTorch’s computational graph to compute **arbitrary-order partial derivatives**. Unlike PyTorch’s native autograd - which stops at first-order derivatives - **thoad** propagates higher-order derivatives throughout the graph, enabling more advanced gradient-based computations.
