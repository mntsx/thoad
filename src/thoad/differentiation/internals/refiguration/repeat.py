# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
import thoad.config as config
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import DirectFunction
from thoad.differentiation.internals.utils.denull import denull_shape
from thoad.typing import AutogradFunction, Shape, Indep, StaticEDData


class RepeatXBackward0(DirectFunction):

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
        output_shape: Shape = self._processed_context["output_shape"]
        repetitions: Tuple[int, ...] = self._processed_context["repetitions"]
        # initialize shape and indep projections
        projected_shape: Shape = output_shape
        projected_indep: Indep = indep
        # project indep if necesary
        if shape != projected_shape:
            projected_indep = shape_align_indep(
                shape=shape,
                indep=indep,
                expected_shape=projected_shape,
            )
        # ensure distribution of repeated dimensions
        projected_indep = tuple(
            d if repetitions[i] == 1 else None for i, d in enumerate(projected_indep)
        )
        # save as class attributes
        self._shape0 = projected_shape
        self._indeps[0] = projected_indep
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_repeats: Tuple[int, ...] = getattr(self._grad_fn, "_saved_repeats")
        saved_self_sym_sizes: Tuple[int, ...] = getattr(
            self._grad_fn, "_saved_self_sym_sizes"
        )
        # ensure proper tensor configuration
        # ...
        # save context
        context: dict[str, Any] = dict()
        context["saved_repeats"] = saved_repeats
        context["saved_self_sym_sizes"] = saved_self_sym_sizes
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_repeats = self._context["saved_repeats"]
        saved_self_sym_sizes = self._context["saved_self_sym_sizes"]
        # process context
        input_shape: Shape = denull_shape(shape=saved_self_sym_sizes)
        output_shape: Tuple[int, ...] = tuple(
            size * reps for (size, reps) in zip(input_shape, saved_repeats)
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input_shape"] = input_shape
        processed_context["output_shape"] = output_shape
        processed_context["repetitions"] = saved_repeats
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
        ### process general info about shapes and indeps
        var_range: Tuple[int, ...] = tuple(range(len(variables)))
        assert all(shapes[v] == self._shape0 for v in var_range if v in variables)
        assert len(set(indeps[v] for v in var_range if v in variables)) == 1
        output_shape: Shape = self._shape0
        output_indep: Indep = indeps[variables[0]]
        assert output_indep == self._indeps[0]
        input_shape: Shape = self._processed_context["input_shape"]
        assert len(input_shape) == len(output_shape)
        repetitions: Tuple[int, ...] = self._processed_context["repetitions"]
        ### determine the new view of distributed diensions in considered partialites
        variable_view: list[int] = list()
        for d, (size, reps) in enumerate(zip(output_shape, repetitions)):
            if d not in output_indep:
                variable_view.extend((size // reps, reps))
        ### gather new shapes, new indeps and view for the new derivative
        new_shapes: list[Shape] = list()
        new_indeps: list[Indep] = list()
        view: list[int] = [derivative.shape[0]]
        view.extend([1 for _ in range(len(output_indep))])
        idx0: int = 0
        idx1: int = len(view)
        einsum_source_indices: list[int] = list(range(idx0, idx1, 1))
        einsum_target_indices: list[int] = list(range(idx0, idx1, 1))
        idx0 = idx1
        for v, (shape, indep) in enumerate(zip(shapes, indeps)):
            new_shape: Shape
            new_indep: Indep
            if v in variables:
                view.extend(variable_view)
                new_shape = input_shape
                new_indep = output_indep
                idx1 += len(variable_view)
                einsum_source_indices.extend(range(idx0, idx1, 1))
                einsum_target_indices.extend(range(idx0, idx1, 2))
            else:
                distributed_shape: Shape = tuple(
                    sz for d, sz in enumerate(shape) if d not in indep
                )
                view.extend(distributed_shape)
                new_shape = shape
                new_indep = indep
                idx1 += len(distributed_shape)
                einsum_source_indices.extend(range(idx0, idx1, 1))
                einsum_target_indices.extend(range(idx0, idx1, 1))
            idx0 = idx1
            for d in new_indep:
                if d is not None:
                    view[(1 + d)] = max(view[(1 + d)], shape[d])
            new_shapes.append(new_shape)
            new_indeps.append(new_indep)
        ### sum repetitions in distribute dimensions
        new_derivative: Tensor = derivative.view(size=view)
        new_derivative = torch.einsum(
            new_derivative, einsum_source_indices, einsum_target_indices
        )

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
