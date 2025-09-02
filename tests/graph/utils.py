# Standard Library Dependencies
import gc
from typing import Any, Tuple, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
import thoad.config as config
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.differentiation.internals.base import (
    ExtendedAutogradFunction,
    ContractiveFunction,
    DirectFunction,
)
from thoad.typing import (
    Shape,
    Indep,
    IDData,
    AutogradFunction,
    Notation,
    StaticEDData,
)


class TestUnivariableXBackward0(ContractiveFunction):

    schwarz: bool = True

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert out_id == 0 and inp_id == 0
        self._shape0 = shape
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        # instantiate derivative tensor
        derivative_shape: Tuple[int, ...] = self._shape0
        derivative: Tensor = torch.ones(
            size=derivative_shape,
            dtype=self._dtype,
            device=self._device,
        )
        # define einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_internal: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(len(self._shape0))),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(False for _ in self._shape0))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data


class TestBivariableXBackward0(ContractiveFunction):

    schwarz: bool = True

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert out_id == 0 and inp_id in (0, 1)
        self._shape0 = shape
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        # instantiate derivative tensor
        derivative_shape: Tuple[int, ...] = self._shape0
        derivative: Tensor = torch.ones(
            size=derivative_shape,
            dtype=self._dtype,
            device=self._device,
        )
        # define einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_internal: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(len(self._shape0))),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(False for _ in self._shape0))
        )
        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        # instantiate derivative tensor
        derivative_shape: Tuple[int, ...] = self._shape0
        derivative: Tensor = torch.ones(
            size=derivative_shape,
            dtype=self._dtype,
            device=self._device,
        )
        # define einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_internal: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(len(self._shape0))),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(False for _ in self._shape0))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()
            case (0, (1,)):
                ID_data = self._compute_internal_0_1()

        return ID_data


class TestTrivariableXBackward0(ContractiveFunction):

    schwarz: bool = True

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert out_id == 0 and inp_id in (0, 1, 2)
        self._shape0 = shape
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        # instantiate derivative tensor
        derivative_shape: Tuple[int, ...] = self._shape0
        derivative: Tensor = torch.ones(
            size=derivative_shape,
            dtype=self._dtype,
            device=self._device,
        )
        # define einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_internal: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(len(self._shape0))),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(False for _ in self._shape0))
        )

        return (derivative, einstein_notation)

    def _compute_internal_0_1(self) -> IDData:
        assert self._processed_context is not None
        # instantiate derivative tensor
        derivative_shape: Tuple[int, ...] = self._shape0
        derivative: Tensor = torch.ones(
            size=derivative_shape,
            dtype=self._dtype,
            device=self._device,
        )
        # define einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_internal: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(len(self._shape0))),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(False for _ in self._shape0))
        )

        return (derivative, einstein_notation)

    def _compute_internal_0_2(self) -> IDData:
        # instantiate derivative tensor
        derivative_shape: Tuple[int, ...] = self._shape0
        derivative: Tensor = torch.ones(
            size=derivative_shape,
            dtype=self._dtype,
            device=self._device,
        )
        # define einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_internal: Tuple[int, ...] = tuple(range(len(self._shape0)))
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(len(self._shape0))),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(self._shape0), tuple(False for _ in self._shape0))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()
            case (0, (1,)):
                ID_data = self._compute_internal_0_1()
            case (0, (2,)):
                ID_data = self._compute_internal_0_2()

        return ID_data


class TestUnivariableXBackward1(DirectFunction):

    schwarz: bool = True

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        self._indeps: list[Union[None, Indep]] = [None]
        return None

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert out_id == 0 and inp_id == 0
        self._shape0 = shape
        self._indeps[0] = indep
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _transform_0_0(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        return (derivative, shapes, indeps)

    def transform(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        out_id: Tuple[Union[None, int], ...],
        inp_id: Tuple[Union[None, int], ...],
    ) -> StaticEDData:
        if bool(getattr(config, "DEBUG", False)):
            self._check_transform(
                derivative=derivative,
                shapes=shapes,
                indeps=indeps,
                out_id=out_id,
                inp_id=inp_id,
            )
        assert all(oo in (None, 0) for oo in out_id)
        assert all(ii in (None, 0) for ii in inp_id)
        variables: Tuple[int, ...]
        variables = tuple(i for i, ii in enumerate(inp_id) if ii == 0)
        ED_data: StaticEDData = self._transform_0_0(
            derivative=derivative,
            shapes=shapes,
            indeps=indeps,
            variables=variables,
        )
        return ED_data


class TestBivariableXBackward1(DirectFunction):

    schwarz: bool = True

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        self._indeps: list[Union[None, Indep]] = [None, None]
        return None

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert out_id == 0 and inp_id in (0, 1)
        self._shape0 = shape
        self._indeps[0] = indep
        self._indeps[1] = indep
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _transform_0_0(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        return (derivative, shapes, indeps)

    def _transform_0_1(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        return (derivative, shapes, indeps)

    def transform(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        out_id: Tuple[Union[None, int], ...],
        inp_id: Tuple[Union[None, int], ...],
    ) -> StaticEDData:
        if bool(getattr(config, "DEBUG", False)):
            self._check_transform(
                derivative=derivative,
                shapes=shapes,
                indeps=indeps,
                out_id=out_id,
                inp_id=inp_id,
            )
        assert all(oo in (None, 0) for oo in out_id)
        ED_data: StaticEDData = (None, None, None)
        for i in range(len(self._indeps)):
            variables: Tuple[int, ...]
            variables = tuple(j for j, ii in enumerate(inp_id) if ii == i)
            match i:
                case 0:
                    ED_data = self._transform_0_0(
                        derivative=derivative,
                        shapes=shapes,
                        indeps=indeps,
                        variables=variables,
                    )
                case 1:
                    ED_data = self._transform_0_1(
                        derivative=derivative,
                        shapes=shapes,
                        indeps=indeps,
                        variables=variables,
                    )
                case _:
                    assert False
        return ED_data


class TestTrivariableXBackward1(DirectFunction):

    schwarz: bool = True

    def __init__(
        self,
        grad_fn: AutogradFunction,
        order: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, dtype=dtype, device=device)
        self._indeps: list[Union[None, Indep]] = [None, None, None]
        return None

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert out_id == 0 and inp_id in (0, 1, 2)
        self._shape0 = shape
        self._indeps[0] = indep
        self._indeps[1] = indep
        self._indeps[2] = indep
        return (shape, indep)

    def _extract_context(self) -> None:
        self._context = dict()
        self._process_context()
        return None

    def _process_context(self) -> None:
        assert self._context is not None
        self._processed_context = dict()
        return None

    def _transform_0_0(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        return (derivative, shapes, indeps)

    def _transform_0_1(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        return (derivative, shapes, indeps)

    def _transform_0_2(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        return (derivative, shapes, indeps)

    def transform(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        out_id: Tuple[Union[None, int], ...],
        inp_id: Tuple[Union[None, int], ...],
    ) -> StaticEDData:
        if bool(getattr(config, "DEBUG", False)):
            self._check_transform(
                derivative=derivative,
                shapes=shapes,
                indeps=indeps,
                out_id=out_id,
                inp_id=inp_id,
            )
        assert all(oo in (None, 0) for oo in out_id)
        ED_data: StaticEDData = (None, None, None)
        for i in range(len(self._indeps)):
            variables: Tuple[int, ...]
            variables = tuple(j for j, ii in enumerate(inp_id) if ii == i)
            match i:
                case 0:
                    ED_data = self._transform_0_0(
                        derivative=derivative,
                        shapes=shapes,
                        indeps=indeps,
                        variables=variables,
                    )
                case 1:
                    ED_data = self._transform_0_1(
                        derivative=derivative,
                        shapes=shapes,
                        indeps=indeps,
                        variables=variables,
                    )
                case 2:
                    ED_data = self._transform_0_2(
                        derivative=derivative,
                        shapes=shapes,
                        indeps=indeps,
                        variables=variables,
                    )
                case _:
                    assert False
        return ED_data


class AccumulateGradX(ContractiveFunction):

    schwarz: bool = True

    def check_shape(
        self,
        out_id: int,
        inp_id: int,
        shape: Shape,
        indep: Indep,
        crossed: bool,
    ) -> Tuple[Shape, Indep]:
        assert self._processed_context is not None
        assert out_id == 0
        assert inp_id == 0
        output: Tensor = self._processed_context["output"]
        # initialize shape and indep projections
        projected_shape: Shape = tuple(output.shape)
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        variable: Tensor = getattr(self._grad_fn, "variable")
        # ensure proper tensor configuration
        variable = variable.to(dtype=self._dtype, device=self._device)
        getattr(variable, "_fix_weakref")()
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["variable"] = variable
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        variable: Tensor = self._context["variable"]
        # process context
        variable: Tensor = denull_tensor(
            tensor=variable, dtype=self._dtype, device=self._device
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["output"] = variable
        self._processed_context = processed_context

        return None

    def _compute_internal_0_0(self) -> IDData:
        assert self._processed_context is not None
        ### Read context
        # ...

        ### Carry out instrumental operations
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)

        ### Instantiate derivative
        derivative: Tensor = t1.sum()

        ### Create einstein notation
        ndim: int = len(self._shape0)
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple()
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = (tuple(range(ndim)),)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append((tuple(), tuple()))

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        assert self._shape0 is not None
        assert self._processed_context is not None
        ID_data: IDData = (None, None)
        match (out_id, inp_id):
            case (0, (0,)):
                ID_data = self._compute_internal_0_0()

        return ID_data


def acquire_test0_gfn_map() -> (
    dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
):
    ### Typings & definitions
    aux: Tensor
    gfn: Union[None, AutogradFunction]
    next_gfn: Union[None, AutogradFunction]
    xfn_type: Type[ExtendedAutogradFunction]
    mapper: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]] = dict()

    ### Instantiate auxiliary tensors
    TA: Tensor = torch.zeros(size=(1,), requires_grad=True)
    TB: Tensor = torch.zeros(size=(1, 1), requires_grad=True)
    TC: Tensor = torch.zeros(size=(1, 1, 1), requires_grad=True)
    TD: Tensor = torch.zeros(size=(2,), requires_grad=True)
    IDX: Tensor = torch.zeros(size=(1,), dtype=torch.long)

    ### ACCUMULATION
    gfn = torch.sum(TA).grad_fn
    assert gfn is not None
    next_gfn = gfn.next_functions[0][0]
    assert next_gfn is not None
    xfn_type = AccumulateGradX
    mapper[type(next_gfn)] = xfn_type

    ### CONDITION
    # torch.where, Tensor.where
    aux = torch.where(condition=(TB > 0), input=TB, other=TB)
    xfn_type = TestTrivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### EXPONENTIATION
    # torch.pow, Tensor.pow (scalar exponent)
    aux = torch.pow(input=TB, exponent=2)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.pow, Tensor.pow (tensor exponent)
    aux = torch.pow(input=TB, exponent=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sqrt, Tensor.sqrt
    aux = torch.sqrt(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.exp, Tensor.exp
    aux = torch.exp(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### INDEXATION
    # torch.clone, Tensor.clone
    aux = torch.clone(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # [int], torch.select, Tensor.select
    aux = torch.select(input=TA, dim=0, index=0)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # [:]
    aux = TA[:]
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # [Tensor]
    aux = TA[TA > 0]
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.index_select, Tensor.index_select
    aux = torch.index_select(input=TA, dim=0, index=IDX)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.index_put, Tensor.index_put_
    aux = torch.index_put(input=TA, indices=(IDX,), values=TA)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.masked_select, Tensor.masked_select
    aux = torch.masked_select(input=TA, mask=(TA >= 0.0))
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.masked_scatter, Tensor.masked_scatter
    aux = torch.masked_scatter(input=TA, mask=(TA >= 0.0), source=TA)
    xfn_type = TestTrivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.gather, Tensor.gather
    aux = torch.gather(input=TA, dim=0, index=IDX)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.scatter, Tensor.scatter_
    aux = torch.scatter(input=TA, dim=0, index=TA.long(), src=TA)
    xfn_type = TestTrivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.take, Tensor.take
    aux = torch.take(input=TA, index=TA.long())
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.put, Tensor.put
    aux = torch.put(input=TA, index=TA.long(), source=TA)
    xfn_type = TestTrivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### LOSS
    # torch.nn.MSELoss, torch.nn.functional.mse_loss
    aux = torch.nn.functional.mse_loss(input=TA, target=TA)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.SmoothL1Loss, torch.nn.functional.smooth_l1_loss
    aux = torch.nn.functional.smooth_l1_loss(input=TA, target=TA)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.BCELoss, torch.nn.functional.binary_cross_entropy
    aux = torch.nn.functional.binary_cross_entropy(input=TA, target=TA)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.BCEWithLogitsLoss, torch.nn.functional.binary_cross_entropy_with_logits
    aux = torch.nn.functional.binary_cross_entropy_with_logits(input=TA, target=TA)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### LINEAR UNITS
    # torch.nn.CeLU, torch.nn.functional.celu
    aux = torch.nn.functional.celu(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.ELU, torch.nn.functional.elu, torch.nn.SELU, torch.nn.functional.selu
    aux = torch.nn.functional.elu(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.GeLU, torch.nn.functional.gelu
    aux = torch.nn.functional.gelu(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.GLU, torch.nn.functional.glu
    aux = torch.nn.functional.glu(input=TD)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.LeakyReLU, torch.nn.functional.leaky_relu
    aux = torch.nn.functional.leaky_relu(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.PReLU, torch.nn.functional.prelu
    aux = torch.nn.functional.prelu(input=TA, weight=TA)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.relu, torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.RReLU, torch.nn.functional.rrelu
    aux = torch.nn.functional.rrelu(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.SiLU, torch.nn.functional.silu
    aux = torch.nn.functional.silu(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### MATRIX MULTIPLICATION
    # torch.addmm, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.addmm(input=TB, mat1=TB, mat2=TB)
    xfn_type = TestTrivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.bmm, Tensor.bmm
    aux = torch.bmm(input=TC, mat2=TC)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # @, torch.dot, Tensor.dot
    aux = torch.dot(input=TA, tensor=TA)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # @, torch.mm, torch.matmul, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.mm(input=TB, mat2=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### PRODUCTS
    # /, torch.div, Tensor.div, Tensor.div_
    aux = torch.div(input=TB, other=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # *, torch.mul, torch.multiply, Tensor.mul, Tensor.mul_
    aux = torch.mul(input=TB, other=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.prod, Tensor.prod (all dims)
    aux = torch.prod(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.prod, Tensor.prod (along dim=1)
    aux = torch.prod(input=TB, dim=1)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### RESHAPE
    # torch.expand, Tensor.expand, Tensor.expand_as
    aux = TA.expand(size=(1,))
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.permute, Tensor.permute, Tensor.T
    aux = torch.permute(input=TB, dims=(0, 1))
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.view, Tensor.view, Tensor.view_as
    aux = TA.view(size=(1, 1))
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.repeat, Tensor.repeat
    aux = TA.repeat(repeats=(1,))
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.reshape, Tensor.reshape
    aux = TA.reshape(shape=(1,))
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (no dim)
    aux = TC.squeeze()
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dim=0)
    aux = TC.squeeze(dim=0)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dims=(0,1))
    aux = TC.squeeze(dim=(0, 1))
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.t, Tensor.t
    aux = torch.t(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.transpose, Tensor.transpose
    aux = torch.transpose(input=TB, dim0=0, dim1=1)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.unsqueeze, Tensor.unsqueeze
    aux = TA.unsqueeze(dim=0)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### SOFTENING
    # torch.softmax, torch.nn.Softmax, torch.nn.functional.softmax
    aux = torch.nn.functional.softmax(input=TA, dim=0)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.LogSoftmax, torch.nn.functional.log_softmax
    aux = torch.nn.functional.log_softmax(input=TA, dim=0)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.Softplus, torch.nn.functional.softplus
    aux = torch.nn.functional.softplus(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### SUMMATIONS
    # +, torch.add, Tensor.add
    aux = torch.add(input=TB, other=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sub, Tensor.subtract
    aux = torch.sub(input=TB, other=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sum, Tensor.sum (all dims)
    aux = torch.sum(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sum, Tensor.sum (dim=(1,))
    aux = torch.sum(input=TB, dim=(1,))
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### TRIGONOMETRY
    # torch.sin
    aux = torch.sin(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.cos
    aux = torch.cos(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.tan
    aux = torch.tan(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sinh
    aux = torch.sinh(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.cosh
    aux = torch.cosh(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.tanh
    aux = torch.tanh(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### MORE MATH
    # torch.abs
    aux = torch.abs(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.log
    aux = torch.log(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.mean (all dims)
    aux = torch.mean(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.mean (dim=1)
    aux = torch.mean(input=TB, dim=1)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.neg
    aux = torch.neg(input=TB)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sigmoid, torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(input=TA)
    xfn_type = TestUnivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.special.xlogy
    aux = torch.special.xlogy(input=TB, other=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    return mapper


def acquire_test1_gfn_map() -> (
    dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]]
):
    ### Typings & definitions
    aux: Tensor
    gfn: Union[None, AutogradFunction]
    next_gfn: Union[None, AutogradFunction]
    xfn_type: Type[ExtendedAutogradFunction]
    mapper: dict[Type[AutogradFunction], Type[ExtendedAutogradFunction]] = dict()

    ### Instantiate auxiliary tensors
    TA: Tensor = torch.zeros(size=(1,), requires_grad=True)
    TB: Tensor = torch.zeros(size=(1, 1), requires_grad=True)
    TC: Tensor = torch.zeros(size=(1, 1, 1), requires_grad=True)
    TD: Tensor = torch.zeros(size=(2,), requires_grad=True)
    IDX: Tensor = torch.zeros(size=(1,), dtype=torch.long)

    ### ACCUMULATION
    gfn = torch.sum(TA).grad_fn
    assert gfn is not None
    next_gfn = gfn.next_functions[0][0]
    assert next_gfn is not None
    xfn_type = AccumulateGradX
    mapper[type(next_gfn)] = xfn_type

    ### CONDITION
    # torch.where, Tensor.where
    aux = torch.where(condition=(TB > 0), input=TB, other=TB)
    xfn_type = TestTrivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### EXPONENTIATION
    # torch.pow, Tensor.pow (scalar exponent)
    aux = torch.pow(input=TB, exponent=2)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.pow, Tensor.pow (tensor exponent)
    aux = torch.pow(input=TB, exponent=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sqrt, Tensor.sqrt
    aux = torch.sqrt(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.exp, Tensor.exp
    aux = torch.exp(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### INDEXATION
    # torch.clone, Tensor.clone
    aux = torch.clone(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # [int], torch.select, Tensor.select
    aux = torch.select(input=TA, dim=0, index=0)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # [:]
    aux = TA[:]
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # [Tensor]
    aux = TA[TA > 0]
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.index_select, Tensor.index_select
    aux = torch.index_select(input=TA, dim=0, index=IDX)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.index_put, Tensor.index_put_
    aux = torch.index_put(input=TA, indices=(IDX,), values=TA)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.masked_select, Tensor.masked_select
    aux = torch.masked_select(input=TA, mask=(TA >= 0.0))
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.masked_scatter, Tensor.masked_scatter
    aux = torch.masked_scatter(input=TA, mask=(TA >= 0.0), source=TA)
    xfn_type = TestTrivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.gather, Tensor.gather
    aux = torch.gather(input=TA, dim=0, index=IDX)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.scatter, Tensor.scatter_
    aux = torch.scatter(input=TA, dim=0, index=TA.long(), src=TA)
    xfn_type = TestTrivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.take, Tensor.take
    aux = torch.take(input=TA, index=TA.long())
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.put, Tensor.put
    aux = torch.put(input=TA, index=TA.long(), source=TA)
    xfn_type = TestTrivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### LOSS
    # torch.nn.MSELoss, torch.nn.functional.mse_loss
    aux = torch.nn.functional.mse_loss(input=TA, target=TA)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.SmoothL1Loss, torch.nn.functional.smooth_l1_loss
    aux = torch.nn.functional.smooth_l1_loss(input=TA, target=TA)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.BCELoss, torch.nn.functional.binary_cross_entropy
    aux = torch.nn.functional.binary_cross_entropy(input=TA, target=TA)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.BCEWithLogitsLoss, torch.nn.functional.binary_cross_entropy_with_logits
    aux = torch.nn.functional.binary_cross_entropy_with_logits(input=TA, target=TA)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### LINEAR UNITS
    # torch.nn.CeLU, torch.nn.functional.celu
    aux = torch.nn.functional.celu(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.ELU, torch.nn.functional.elu, torch.nn.SELU, torch.nn.functional.selu
    aux = torch.nn.functional.elu(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.GeLU, torch.nn.functional.gelu
    aux = torch.nn.functional.gelu(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.GLU, torch.nn.functional.glu
    aux = torch.nn.functional.glu(input=TD)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.LeakyReLU, torch.nn.functional.leaky_relu
    aux = torch.nn.functional.leaky_relu(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.PReLU, torch.nn.functional.prelu
    aux = torch.nn.functional.prelu(input=TA, weight=TA)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.relu, torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.RReLU, torch.nn.functional.rrelu
    aux = torch.nn.functional.rrelu(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.SiLU, torch.nn.functional.silu
    aux = torch.nn.functional.silu(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### MATRIX MULTIPLICATION
    # torch.addmm, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.addmm(input=TB, mat1=TB, mat2=TB)
    xfn_type = TestTrivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.bmm, Tensor.bmm
    aux = torch.bmm(input=TC, mat2=TC)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # @, torch.dot, Tensor.dot
    aux = torch.dot(input=TA, tensor=TA)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # @, torch.mm, torch.matmul, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.mm(input=TB, mat2=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### PRODUCTS
    # /, torch.div, Tensor.div, Tensor.div_
    aux = torch.div(input=TB, other=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # *, torch.mul, torch.multiply, Tensor.mul, Tensor.mul_
    aux = torch.mul(input=TB, other=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.prod, Tensor.prod (all dims)
    aux = torch.prod(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.prod, Tensor.prod (along dim=1)
    aux = torch.prod(input=TB, dim=1)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### RESHAPE
    # torch.expand, Tensor.expand, Tensor.expand_as
    aux = TA.expand(size=(1,))
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.permute, Tensor.permute, Tensor.T
    aux = torch.permute(input=TB, dims=(0, 1))
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.view, Tensor.view, Tensor.view_as
    aux = TA.view(size=(1, 1))
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.repeat, Tensor.repeat
    aux = TA.repeat(repeats=(1,))
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.reshape, Tensor.reshape
    aux = TA.reshape(shape=(1,))
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (no dim)
    aux = TC.squeeze()
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dim=0)
    aux = TC.squeeze(dim=0)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.squeeze, Tensor.squeeze (dims=(0,1))
    aux = TC.squeeze(dim=(0, 1))
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.t, Tensor.t
    aux = torch.t(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.transpose, Tensor.transpose
    aux = torch.transpose(input=TB, dim0=0, dim1=1)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.unsqueeze, Tensor.unsqueeze
    aux = TA.unsqueeze(dim=0)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### SOFTENING
    # torch.softmax, torch.nn.Softmax, torch.nn.functional.softmax
    aux = torch.nn.functional.softmax(input=TA, dim=0)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.LogSoftmax, torch.nn.functional.log_softmax
    aux = torch.nn.functional.log_softmax(input=TA, dim=0)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.nn.Softplus, torch.nn.functional.softplus
    aux = torch.nn.functional.softplus(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### SUMMATIONS
    # +, torch.add, Tensor.add
    aux = torch.add(input=TB, other=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sub, Tensor.subtract
    aux = torch.sub(input=TB, other=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sum, Tensor.sum (all dims)
    aux = torch.sum(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sum, Tensor.sum (dim=(1,))
    aux = torch.sum(input=TB, dim=(1,))
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### TRIGONOMETRY
    # torch.sin
    aux = torch.sin(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.cos
    aux = torch.cos(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.tan
    aux = torch.tan(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sinh
    aux = torch.sinh(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.cosh
    aux = torch.cosh(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.tanh
    aux = torch.tanh(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### MORE MATH
    # torch.abs
    aux = torch.abs(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.log
    aux = torch.log(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.mean (all dims)
    aux = torch.mean(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.mean (dim=1)
    aux = torch.mean(input=TB, dim=1)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.neg
    aux = torch.neg(input=TB)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.sigmoid, torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type
    # torch.special.xlogy
    aux = torch.special.xlogy(input=TB, other=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    return mapper
