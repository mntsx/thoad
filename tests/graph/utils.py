# Standard Library Dependencies
import gc
from typing import Tuple, Type

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.autodifferentiation.internals.base import (
    ExtendedAutogradFunction,
    ContractiveFunction,
)
from thoad.typing.data import Shape, Indep, IDData, AutogradFunction, Notation


class TestUnivariableXBackward(ContractiveFunction):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        return None

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        self._shape = shape
        self._indep = indep
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        return None

    def _process_context(self) -> None:
        assert self._shape is not None
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        # instantiate differential tensor
        differential_shape: Tuple[int, ...] = self._shape
        differential: Tensor = torch.ones(size=differential_shape)
        # define einstein notation
        einstein_external: list[int] = list(range(len(self._shape)))
        einstein_internal: list[int] = list(range(len(self._shape)))
        einstein_composed: list[list[int]] = list(range(len(self._shape)))
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([self._shape])

        return (differential, einstein_notation)

    def compute_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        if self._processed_context is None:
            self._process_context()
        differential: Tensor
        einstein_notation: Notation
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)


class TestBivariableXBackward(ContractiveFunction):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        return None

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        self._shape = shape
        self._indep = indep
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        return None

    def _process_context(self) -> None:
        assert self._shape is not None
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert len(self._processed_context) == 0
        # instantiate differential tensor
        differential_shape: Tuple[int, ...] = self._shape
        differential: Tensor = torch.ones(size=differential_shape)
        # define einstein notation
        einstein_external: list[int] = list(range(len(self._shape)))
        einstein_internal: list[int] = list(range(len(self._shape)))
        einstein_composed: list[list[int]] = list(range(len(self._shape)))
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([self._shape])
        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> Tuple[Tensor, Notation]:
        assert len(self._processed_context) == 0
        # instantiate differential tensor
        differential_shape: Tuple[int, ...] = self._shape
        differential: Tensor = torch.ones(size=differential_shape)
        # define einstein notation
        einstein_external: list[int] = list(range(len(self._shape)))
        einstein_internal: list[int] = list(range(len(self._shape)))
        einstein_composed: list[list[int]] = list(range(len(self._shape)))
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([self._shape])

        return (differential, einstein_notation)

    def compute_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        if self._processed_context is None:
            self._process_context()
        differential: Tensor
        einstein_notation: Notation
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case (0, (1,)):
                (differential, einstein_notation) = self._compute_internal_0_1()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)


class TestTrivariableXBackward(ContractiveFunction):

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        return None

    def check_shape(self, shape: Shape, indep: Indep) -> Tuple[Shape, Indep]:
        self._shape = shape
        self._indep = indep
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        return None

    def _process_context(self) -> None:
        assert self._shape is not None
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _compute_internal_0_0(self) -> Tuple[Tensor, Notation]:
        assert len(self._processed_context) == 0
        # instantiate differential tensor
        differential_shape: Tuple[int, ...] = self._shape
        differential: Tensor = torch.ones(size=differential_shape)
        # define einstein notation
        einstein_external: list[int] = list(range(len(self._shape)))
        einstein_internal: list[int] = list(range(len(self._shape)))
        einstein_composed: list[list[int]] = list(range(len(self._shape)))
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([self._shape])

        return (differential, einstein_notation)

    def _compute_internal_0_1(self) -> Tuple[Tensor, Notation]:
        assert len(self._processed_context) == 0
        # instantiate differential tensor
        differential_shape: Tuple[int, ...] = self._shape
        differential: Tensor = torch.ones(size=differential_shape)
        # define einstein notation
        einstein_external: list[int] = list(range(len(self._shape)))
        einstein_internal: list[int] = list(range(len(self._shape)))
        einstein_composed: list[list[int]] = list(range(len(self._shape)))
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([self._shape])

        return (differential, einstein_notation)

    def _compute_internal_0_2(self) -> Tuple[Tensor, Notation]:
        assert len(self._processed_context) == 0
        # instantiate differential tensor
        differential_shape: Tuple[int, ...] = self._shape
        differential: Tensor = torch.ones(size=differential_shape)
        # define einstein notation
        einstein_external: list[int] = list(range(len(self._shape)))
        einstein_internal: list[int] = list(range(len(self._shape)))
        einstein_composed: list[list[int]] = list(range(len(self._shape)))
        einstein_notation: Notation = list()
        einstein_notation.append([einstein_external, einstein_internal])
        einstein_notation.append([einstein_composed])
        einstein_notation.append([self._shape])

        return (differential, einstein_notation)

    def compute_internal(
        self,
        out_id: int,
        in_id: Tuple[int, ...],
    ) -> IDData:
        if self._processed_context is None:
            self._process_context()
        differential: Tensor
        einstein_notation: Notation
        match (out_id, in_id):
            case (0, (0,)):
                (differential, einstein_notation) = self._compute_internal_0_0()
            case (0, (1,)):
                (differential, einstein_notation) = self._compute_internal_0_1()
            case (0, (2,)):
                (differential, einstein_notation) = self._compute_internal_0_2()
            case _:
                (differential, einstein_notation) = (None, None)
        return (differential, einstein_notation)


def acquire_test_gfn_map() -> dict[Type, Type]:
    ### Typings & definitions
    aux: Tensor
    xfn_type: Type[ExtendedAutogradFunction]
    mapper: dict[Type, Type] = dict()
    ### Instantiate auxiliary tensors
    TA: Tensor = torch.zeros(size=(1,), requires_grad=True)
    TB: Tensor = torch.zeros(size=(1, 1), requires_grad=True)
    TC: Tensor = torch.zeros(size=(1, 1, 1), requires_grad=True)
    TD: Tensor = torch.zeros(size=(2,), requires_grad=True)
    IDX: Tensor = torch.zeros(size=(1,), dtype=torch.long)

    ### ACCUMULATION
    gfn_type = type(torch.sum(TA).grad_fn.next_functions[0][0])
    xfn_type = TestUnivariableXBackward
    mapper[gfn_type] = xfn_type

    ### CONDITION
    # torch.where, Tensor.where
    aux = torch.where(condition=(TB > 0), input=TB, other=TB)
    xfn_type = TestTrivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### EXPONENTIATION
    # torch.pow, Tensor.pow (scalar exponent)
    aux = torch.pow(input=TB, exponent=2)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.pow, Tensor.pow (tensor exponent)
    aux = torch.pow(input=TB, exponent=TB)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sqrt, Tensor.sqrt
    aux = torch.sqrt(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.exp, Tensor.exp
    aux = torch.exp(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### INDEXATION
    # torch.clone, Tensor.clone
    aux = torch.clone(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # [int], torch.select, Tensor.select
    aux = torch.select(input=TA, dim=0, index=0)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # [:]
    aux = TA[:]
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # [Tensor]
    aux = TA[TA > 0]
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.index_select, Tensor.index_select
    aux = torch.index_select(input=TA, dim=0, index=IDX)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.index_put, Tensor.index_put_
    aux = torch.index_put(input=TA, indices=(IDX,), values=TA)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.masked_select, Tensor.masked_select
    aux = torch.masked_select(input=TA, mask=(TA >= 0.0))
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.masked_scatter, Tensor.masked_scatter
    aux = torch.masked_scatter(input=TA, mask=(TA >= 0.0), source=TA)
    xfn_type = TestTrivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.gather, Tensor.gather
    aux = torch.gather(input=TA, dim=0, index=IDX)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.scatter, Tensor.scatter_
    aux = torch.scatter(input=TA, dim=0, index=TA.long(), src=TA)
    xfn_type = TestTrivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.take, Tensor.take
    aux = torch.take(input=TA, index=TA.long())
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.put, Tensor.put
    aux = torch.put(input=TA, index=TA.long(), source=TA)
    xfn_type = TestTrivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### LOSS
    # torch.nn.MSELoss, torch.nn.functional.mse_loss
    aux = torch.nn.functional.mse_loss(input=TA, target=TA)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.SmoothL1Loss, torch.nn.functional.smooth_l1_loss
    aux = torch.nn.functional.smooth_l1_loss(input=TA, target=TA)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.BCELoss, torch.nn.functional.binary_cross_entropy
    aux = torch.nn.functional.binary_cross_entropy(input=TA, target=TA)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.BCEWithLogitsLoss, torch.nn.functional.binary_cross_entropy_with_logits
    aux = torch.nn.functional.binary_cross_entropy_with_logits(input=TA, target=TA)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### LINEAR UNITS
    # torch.nn.CeLU, torch.nn.functional.celu
    aux = torch.nn.functional.celu(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.ELU, torch.nn.functional.elu, torch.nn.SELU, torch.nn.functional.selu
    aux = torch.nn.functional.elu(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.GeLU, torch.nn.functional.gelu
    aux = torch.nn.functional.gelu(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.GLU, torch.nn.functional.glu
    aux = torch.nn.functional.glu(input=TD)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.LeakyReLU, torch.nn.functional.leaky_relu
    aux = torch.nn.functional.leaky_relu(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.PReLU, torch.nn.functional.prelu
    aux = torch.nn.functional.prelu(input=TA, weight=TA)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.relu, torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.RReLU, torch.nn.functional.rrelu
    aux = torch.nn.functional.rrelu(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.SiLU, torch.nn.functional.silu
    aux = torch.nn.functional.silu(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### MATRIX MULTIPLICATION
    # torch.addmm, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.addmm(input=TB, mat1=TB, mat2=TB)
    xfn_type = TestTrivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.bmm, Tensor.bmm
    aux = torch.bmm(input=TC, mat2=TC)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # @, torch.dot, Tensor.dot
    aux = torch.dot(input=TA, tensor=TA)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # @, torch.mm, torch.matmul, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.mm(input=TB, mat2=TB)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### PRODUCTS
    # /, torch.div, Tensor.div, Tensor.div_
    aux = torch.div(input=TB, other=TB)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # *, torch.mul, torch.multiply, Tensor.mul, Tensor.mul_
    aux = torch.mul(input=TB, other=TB)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.prod, Tensor.prod (all dims)
    aux = torch.prod(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.prod, Tensor.prod (along dim=1)
    aux = torch.prod(input=TB, dim=1)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### RESHAPE
    # torch.expand, Tensor.expand, Tensor.expand_as
    aux = TA.expand(size=(1,))
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.permute, Tensor.permute, Tensor.T
    aux = torch.permute(input=TB, dims=(0, 1))
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.view, Tensor.view, Tensor.view_as
    aux = TA.view(size=(1, 1))
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.repeat, Tensor.repeat
    aux = TA.repeat(repeats=(1,))
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.reshape, Tensor.reshape
    aux = TA.reshape(shape=(1,))
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (no dim)
    aux = TC.squeeze()
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dim=0)
    aux = TC.squeeze(dim=0)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dims=(0,1))
    aux = TC.squeeze(dim=(0, 1))
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.t, Tensor.t
    aux = torch.t(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.transpose, Tensor.transpose
    aux = torch.transpose(input=TB, dim0=0, dim1=1)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.unsqueeze, Tensor.unsqueeze
    aux = TA.unsqueeze(dim=0)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### SOFTENING
    # torch.softmax, torch.nn.Softmax, torch.nn.functional.softmax
    aux = torch.nn.functional.softmax(input=TA, dim=0)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.LogSoftmax, torch.nn.functional.log_softmax
    aux = torch.nn.functional.log_softmax(input=TA, dim=0)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.nn.Softplus, torch.nn.functional.softplus
    aux = torch.nn.functional.softplus(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### SUMMATIONS
    # +, torch.add, Tensor.add
    aux = torch.add(input=TB, other=TB)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sub, Tensor.subtract
    aux = torch.sub(input=TB, other=TB)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sum, Tensor.sum (all dims)
    aux = torch.sum(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sum, Tensor.sum (dim=(1,))
    aux = torch.sum(input=TB, dim=(1,))
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### TRIGONOMETRY
    # torch.sin
    aux = torch.sin(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.cos
    aux = torch.cos(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.tan
    aux = torch.tan(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sinh
    aux = torch.sinh(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.cosh
    aux = torch.cosh(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.tanh
    aux = torch.tanh(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    ### MORE MATH
    # torch.abs
    aux = torch.abs(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.neg
    aux = torch.neg(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.mean (all dims)
    aux = torch.mean(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.mean (dim=1)
    aux = torch.mean(input=TB, dim=1)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.log
    aux = torch.log(input=TB)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.special.xlogy
    aux = torch.special.xlogy(input=TB, other=TB)
    xfn_type = TestBivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type
    # torch.sigmoid, torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(input=TA)
    xfn_type = TestUnivariableXBackward
    mapper[type(aux.grad_fn)] = xfn_type

    gc.collect()

    return mapper
