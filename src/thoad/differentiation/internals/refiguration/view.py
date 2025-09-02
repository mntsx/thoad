# Standard Library Dependencies
import math
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
import thoad.config as config
from thoad.differentiation.internals.base import DirectFunction
from thoad.differentiation.internals.utils.denull import denull_shape
from thoad.typing import AutogradFunction, Shape, Indep, StaticEDData


def track_dimensions(
    source_shape: Shape, target_shape: Shape
) -> Tuple[Union[None, int], ...]:
    d1: int = 0
    accumulated_numel_0: int = 1
    accumulated_numel_1: int = 1
    coincident_init_numel: bool = True
    tracked_dimensions: list[Union[None, int]] = list()
    for d0, size in enumerate(source_shape):
        accumulated_numel_0 *= size
        dim_count: int = 0
        while accumulated_numel_0 > accumulated_numel_1:
            dim_count += 1
            accumulated_numel_1 *= target_shape[d1]
            d1 += 1
        coincident_stop_numel: bool = accumulated_numel_0 == accumulated_numel_1
        if coincident_init_numel and coincident_stop_numel and dim_count == 1:
            tracked_dimensions.append(d0)
        else:
            tracked_dimensions.extend([None for _ in range(dim_count)])
        coincident_init_numel = coincident_stop_numel
    return tuple(tracked_dimensions)


class ViewXBackward0(DirectFunction):

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
        assert self._processed_context is not None
        assert out_id == 0
        assert inp_id == 0
        input_shape: Shape = self._processed_context["input_shape"]
        assert math.prod(input_shape) == math.prod(shape)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # ensure distribution of reshaped dimensions
        tracked_dimensions: Tuple[Union[None, int], ...] = track_dimensions(
            source_shape=shape, target_shape=input_shape
        )
        projected_indep: Indep = tuple(
            d if d in tracked_dimensions else None for d in indep
        )
        # save as class attributes
        self._shape0 = projected_shape
        self._indeps[0] = projected_indep
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # save context
        context: dict[str, Any] = dict()
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input_shape"] = input_shape
        self._processed_context = processed_context

        return None

    def _transform_0_0(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        assert self._processed_context is not None
        # process general info about shapes and indeps
        var_range: Tuple[int, ...] = tuple(range(len(variables)))
        assert all(shapes[v] == self._shape0 for v in var_range if v in variables)
        assert all(indeps[v] == self._indeps[0] for v in var_range if v in variables)
        output_shape: Shape = self._shape0
        output_indep: Indep = indeps[variables[0]]
        input_shape: Shape = self._processed_context["input_shape"]
        # determine which dimensions are kept intact
        tracked_dimensions: Tuple[Union[None, int], ...] = track_dimensions(
            source_shape=output_shape, target_shape=input_shape
        )
        # align output indep to input shape
        expected_indep: list[Union[None, int]] = list()
        for d in output_indep:
            expected_indep.append(None if d is None else tracked_dimensions.index(d))
        # determine the new view of distributed diensions in considered partialites
        variable_view: list[int] = list()
        for d, size in enumerate(input_shape):
            td: Union[None, int] = tracked_dimensions[d]
            if td is None or td not in output_indep:
                variable_view.append(size)
        # gather new shapes, new indeps and view for the new derivative
        new_shapes: list[Shape] = list()
        new_indeps: list[Indep] = list()
        view: list[int] = [derivative.shape[0]]
        view.extend([1 for _ in range(len(output_indep))])
        for v, (shape, indep) in enumerate(zip(shapes, indeps)):
            new_shape: Shape
            new_indep: Indep
            if v in variables:
                view.extend(variable_view)
                new_shape = input_shape
                new_indep = tuple(expected_indep)
            else:
                view.extend(tuple(sz for d, sz in enumerate(shape) if d not in indep))
                new_shape = shape
                new_indep = indep
            for i, d in enumerate(new_indep):
                if d is not None:
                    view[(1 + i)] = max(view[(1 + i)], new_shape[d])
            new_shapes.append(new_shape)
            new_indeps.append(new_indep)
        # reshape distributed dimensions
        new_derivative: Tensor = derivative.view(size=view)

        return (new_derivative, tuple(new_shapes), tuple(new_indeps))

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


class UnsafeViewXBackward0(DirectFunction):
    """
    When you see UnsafeViewBackward0 in your .grad_fn chain, it simply means that
    somewhere under the hood PyTorch emitted an _unsafe_view rather than a “safe” view.
    There’s no behavioral difference unless you try to do an in‑place write on that
    tensor (in which case Autograd won’t know how to track it)
    """

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
        assert self._processed_context is not None
        assert out_id == 0
        assert inp_id == 0
        input_shape: Shape = self._processed_context["input_shape"]
        assert math.prod(input_shape) == math.prod(shape)
        # initialize shape and indep projections
        projected_shape: Shape = shape
        projected_indep: Indep = indep
        # ensure distribution of reshaped dimensions
        tracked_dimensions: Tuple[Union[None, int], ...] = track_dimensions(
            source_shape=shape, target_shape=input_shape
        )
        projected_indep: Indep = tuple(
            d if d in tracked_dimensions else None for d in indep
        )
        # save as class attributes
        self._shape0 = projected_shape
        self._indeps[0] = projected_indep
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # save context
        context: dict[str, Any] = dict()
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_self_sym_sizes: Tuple[int, ...] = self._context["saved_self_sym_sizes"]
        # process context
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input_shape"] = input_shape
        self._processed_context = processed_context

        return None

    def _transform_0_0(
        self,
        derivative: Tensor,
        shapes: Tuple[Shape, ...],
        indeps: Tuple[Indep, ...],
        variables: Tuple[int, ...],
    ) -> StaticEDData:
        assert self._processed_context is not None
        # process general info about shapes and indeps
        var_range: Tuple[int, ...] = tuple(range(len(variables)))
        assert all(shapes[v] == self._shape0 for v in var_range if v in variables)
        assert all(indeps[v] == self._indeps[0] for v in var_range if v in variables)
        output_shape: Shape = self._shape0
        output_indep: Indep = indeps[variables[0]]
        input_shape: Shape = self._processed_context["input_shape"]
        # determine which dimensions are kept intact
        tracked_dimensions: Tuple[Union[None, int], ...] = track_dimensions(
            source_shape=output_shape, target_shape=input_shape
        )
        # align output indep to input shape
        expected_indep: list[Union[None, int]] = list()
        for d in output_indep:
            expected_indep.append(None if d is None else tracked_dimensions.index(d))
        # determine the new view of distributed diensions in considered partialites
        variable_view: list[int] = list()
        for d, size in enumerate(input_shape):
            td: Union[None, int] = tracked_dimensions[d]
            if td is None or td not in output_indep:
                variable_view.append(size)
        # gather new shapes, new indeps and view for the new derivative
        new_shapes: list[Shape] = list()
        new_indeps: list[Indep] = list()
        view: list[int] = [derivative.shape[0]]
        view.extend([1 for _ in range(len(output_indep))])
        for v, (shape, indep) in enumerate(zip(shapes, indeps)):
            new_shape: Shape
            new_indep: Indep
            if v in variables:
                view.extend(variable_view)
                new_shape = input_shape
                new_indep = tuple(expected_indep)
            else:
                view.extend(tuple(sz for d, sz in enumerate(shape) if d not in indep))
                new_shape = shape
                new_indep = indep
            for i, d in enumerate(new_indep):
                if d is not None:
                    view[(1 + i)] = max(view[(1 + i)], new_shape[d])
            new_shapes.append(new_shape)
            new_indeps.append(new_indep)
        # reshape distributed dimensions
        new_derivative: Tensor = derivative.view(size=view)

        return (new_derivative, tuple(new_shapes), tuple(new_indeps))

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
