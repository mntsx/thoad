# Standard Library Dependencies
from typing import Any

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.differentiation.internals.mathematic import MaxXBackward0, MaxXBackward1
from thoad.differentiation.internals.utils.denull import denull_tensor


class MinXBackward0(MaxXBackward0):

    pass


class MinXBackward1(MaxXBackward1):

    def _process_context(self) -> None:
        # checks
        assert self._context is not None
        # load context
        saved_result: Tensor = self._context["saved_result"]
        saved_self: Tensor = self._context["saved_self"]
        # process context
        input: Tensor = denull_tensor(
            tensor=saved_self, dtype=self._dtype, device=self._device
        )
        output: Tensor = denull_tensor(
            tensor=saved_result, dtype=self._dtype, device=self._device
        )
        condition: Tensor = input <= output
        t0: Tensor = torch.zeros(size=(1,), dtype=self._dtype, device=self._device)
        t1: Tensor = torch.ones(size=(1,), dtype=self._dtype, device=self._device)
        weight: Tensor = torch.where(condition=condition, input=t1, other=t0)
        maximum_count: int = int(torch.sum(condition).item())
        assert maximum_count >= 1
        weight /= maximum_count
        # save processed context
        processed_context: dict[str, Any] = dict()
        processed_context["input"] = input
        processed_context["output"] = output
        processed_context["weight"] = weight
        self._processed_context = processed_context

        return None
