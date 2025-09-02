# Standard Library Dependencies
from typing import Any, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.alignment import shape_align_indep
from thoad.differentiation.internals.base import ContractiveFunction
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.differentiation.internals.utils.derivate import reduced_logsoftmax_derivate
from thoad.differentiation.internals.utils.recalibration import recalibrate_dim
from thoad.typing import Shape, Indep, Notation, IDData


class LogSoftmaxXBackward0(ContractiveFunction):

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
        dim: int = self._processed_context["dim"]
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
        # desindep the contracted dimension
        aux: list[Union[None, int]] = list(projected_indep)
        if dim in projected_indep:
            aux[projected_indep.index(dim)] = None
        projected_indep = tuple(aux)
        # save as class attributes
        self._shape0 = projected_shape
        return (projected_shape, projected_indep)

    def _extract_context(self) -> None:
        # extract info
        saved_dim: Tuple[int, ...] = getattr(self._grad_fn, "_saved_dim")
        saved_result: Tensor = getattr(self._grad_fn, "_saved_result")
        # ensure proper tensor configuration
        saved_result = saved_result.to(dtype=self._dtype, device=self._device)
        getattr(saved_result, "_fix_weakref")()
        # save context
        context: dict[str, Any] = dict()
        context["saved_dim"] = saved_dim
        context["saved_result"] = saved_result
        self._context = context
        # process context
        self._process_context()
        return None

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_dim: int = self._context["saved_dim"]
        saved_result: Tensor = self._context["saved_result"]
        # process context
        dim: int = recalibrate_dim(dim=saved_dim, shape=saved_result.shape)
        output: Tensor = denull_tensor(
            tensor=saved_result, dtype=self._dtype, device=self._device
        )
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["dim"] = dim
        processed_context["output"] = output
        self._processed_context = processed_context

        return None

    def _compute_internal_0(self, order: int) -> IDData:
        assert self._processed_context is not None
        assert self._shape0 is not None
        assert self._processed_context is not None
        ### Read context
        dim: int = self._processed_context["dim"]
        output: Tensor = self._processed_context["output"]

        ### Carry out instrumental operations
        ndim: int = output.ndim
        dim_size: int = output.shape[dim]
        # treat batch dims
        batch_range: Tuple[int, ...] = tuple(d for d in range(ndim) if d != dim)
        batch_shape: list[int] = [sz for i, sz in enumerate(output.shape) if i != dim]
        # permute output placing interest dimension at the end
        permutation: Tuple[int, ...] = (*batch_range, dim)
        permuted_output: Tensor = output.permute(permutation)
        # arange einsum indices
        dual_range: list[int] = [d for d in range(ndim, ndim + order)]
        internal_indices: list[int]
        internal_indices = [*permutation, *[d for d in dual_range]]
        composed_indices: list[list[int]] = list()
        pre_batch: Tuple[int, ...] = batch_range[:dim]
        pos_batch: Tuple[int, ...] = batch_range[dim:]
        for d in dual_range:
            composed_indices.append([*pre_batch, d, *pos_batch])
        # declare info about internal derivative shape & values
        internal_shape: list[int] = [*batch_shape, *((dim_size,) * (1 + order))]
        internal_independencies: list[bool] = [True for _ in batch_shape]
        internal_independencies.extend([False for _ in range(order + 1)])

        ### Instantiate derivative
        derivative: Tensor = reduced_logsoftmax_derivate(
            input=permuted_output,
            order=order,
        )

        ### Create einstein notation
        einstein_external: Tuple[int, ...] = tuple(range(ndim))
        einstein_internal: Tuple[int, ...] = tuple(internal_indices)
        einstein_composed: Tuple[Tuple[int, ...], ...]
        einstein_composed = tuple(tuple(indices) for indices in composed_indices)
        einstein_notation: Notation = list()
        einstein_notation.append((einstein_external, einstein_internal))
        einstein_notation.append(tuple(einstein_composed))
        einstein_notation.append(
            (tuple(internal_shape), tuple(internal_independencies))
        )

        return (derivative, einstein_notation)

    def compute_internal(self, out_id: int, inp_id: Tuple[int, ...]) -> IDData:
        ID_data: IDData
        if out_id == 0 and all(i == 0 for i in inp_id):
            order: int = len(inp_id)
            ID_data = self._compute_internal_0(order=order)
        else:
            ID_data = (None, None)
        return ID_data
