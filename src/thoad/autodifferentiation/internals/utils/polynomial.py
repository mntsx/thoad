import torch
from torch import Tensor


def poly_eval(coeffs: list[float], x: Tensor) -> Tensor:
    """
    Evaluates the polynomial sum_{k=0}^{m} coeffs[k] * x^k
    for each element of x (tensor).
    """
    # We use an inverted Horner-type approach (fewer multiplications).
    # But with large tensors, sometimes the "naive" poly_eval also performs well.
    # Here, for brevity, we do the direct form:
    y: Tensor = torch.zeros_like(x)
    for k, ak in enumerate(coeffs):
        y += ak * (x**k)
    return y


def poly_add(p: list[float], q: list[float]) -> list[float]:
    """Sum of two polynomials p+q."""
    m: int = max(len(p), len(q))
    r: list[float] = [0.0] * m
    for i in range(m):
        if i < len(p):
            r[i] += p[i]
        if i < len(q):
            r[i] += q[i]
    return r


def poly_scalar_mul(p: list[float], alpha: float) -> list[float]:
    """Multiplication of the polynomial p by a constant alpha."""
    return [alpha * pk for pk in p]


def poly_var_mul(p: list[float], q: list[float]) -> list[float]:
    """
    Multiplication of two polynomials p(s)*q(s).
    Returns the coefficients of the product.
    """
    r: list[float] = [0.0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            r[i + j] += pi * qj
    return r


def poly_derivative(p: list[float]) -> list[float]:
    """d/ds derivative of a polynomial p(s)."""
    if len(p) <= 1:
        return [0.0]
    # d/ds [a0 + a1*s + a2*s^2 + ...] = a1 + 2*a2*s + 3*a3*s^2 + ...
    return [(i + 1) * p[i + 1] for i in range(len(p) - 1)]
