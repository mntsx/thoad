# Standard Library Dependencies
from typing import Any

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from thoad.differentiation.engine.broadcasting.figuration import infer_broadcast
from thoad.differentiation.internals.mathematic import MaximumXBackward0
from thoad.differentiation.internals.utils.denull import denull_tensor
from thoad.typing import Shape


class MinimumXBackward0(MaximumXBackward0):

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_other: Tensor = self._context["saved_other"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        raw_input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        raw_other: Tensor = denull_tensor(
            tensor=saved_other, dtype=self._dtype, device=self._device
        )
        tensors_shapes: list[Shape]
        tensors_shapes = [T.shape for T in (raw_input, raw_other) if T is not None]
        broadcasted_shape: Shape = infer_broadcast(shapes=tensors_shapes)
        input: Tensor = raw_input.broadcast_to(size=broadcasted_shape)
        other: Tensor = raw_other.broadcast_to(size=broadcasted_shape)
        input_weight: Tensor
        other_weight: Tensor
        input_weight = (input <= other).to(dtype=self._dtype, device=self._device)
        other_weight = (other <= input).to(dtype=self._dtype, device=self._device)
        weight_sum: Tensor = input_weight + other_weight
        input_weight /= weight_sum
        other_weight /= weight_sum
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["broadcasted_shape"] = broadcasted_shape
        processed_context["input"] = input
        processed_context["raw_input"] = raw_input
        processed_context["other"] = other
        processed_context["raw_other"] = raw_other
        processed_context["input_weight"] = input_weight
        processed_context["other_weight"] = other_weight
        self._processed_context = processed_context

        return None
