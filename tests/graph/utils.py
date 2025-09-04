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

    ### EXPONENTIATION

    ### INDEXATION

    ### LOSS

    ### LINEAR UNITS
    # torch.relu, torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(input=TA)
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
    # @, torch.mm, torch.matmul, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.mm(input=TB, mat2=TB)
    xfn_type = TestBivariableXBackward0
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### PRODUCTS

    ### RESHAPE

    ### SOFTENING

    ### SUMMATIONS

    ### TRIGONOMETRY

    ### MORE MATH
    # torch.sigmoid, torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(input=TA)
    xfn_type = TestUnivariableXBackward0
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

    ### EXPONENTIATION

    ### INDEXATION

    ### LOSS

    ### LINEAR UNITS
    # torch.relu, torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(input=TA)
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
    # @, torch.mm, torch.matmul, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.mm(input=TB, mat2=TB)
    xfn_type = TestBivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    ### PRODUCTS

    ### RESHAPE

    ### SOFTENING

    ### SUMMATIONS

    ### TRIGONOMETRY

    ### MORE MATH
    # torch.sigmoid, torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(input=TA)
    xfn_type = TestUnivariableXBackward1
    gfn = aux.grad_fn
    assert gfn is not None
    mapper[type(gfn)] = xfn_type

    return mapper
