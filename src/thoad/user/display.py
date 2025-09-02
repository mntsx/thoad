# Standard Library Dependencies
import logging
import sys
from logging import Logger, Handler
from typing import Callable

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from thoad.typing import AutogradFunction


class LogScaper:
    _logger: Logger
    _handlers: list[Handler]

    def scape_logger_handlers(self) -> None:
        self._logger = logging.getLogger("graph_logger")
        # save existing handlers
        self._handlers = self._logger.handlers.copy()
        self._logger.setLevel(logging.INFO)
        # replace them with our simple “message only” handler
        self._logger.handlers.clear()
        handler: Handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        return None

    def log(self, message: str) -> None:
        self._logger.info(message)
        return None

    def restore_logger_handlers(self) -> None:
        self._logger.handlers.clear()
        for h in self._handlers:
            self._logger.addHandler(h)
        return None


def display_tensor_subgraph(
    tensor: Tensor,
    supports: Callable[[AutogradFunction], bool],
) -> None:
    # typings
    message: str
    # color
    _RED: str = "\033[31m"
    _GREY: str = "\033[90m"
    _RESET: str = "\033[0m"
    # log scaper
    scaper: LogScaper = LogScaper()
    scaper.scape_logger_handlers()

    ### Display
    if not (r := tensor.grad_fn):
        return scaper.log(message="\u00b7<no grad_fn>")
    seen: set[AutogradFunction] = set()

    def dfs(fn: AutogradFunction, indent: str, is_last: bool) -> None:
        message: str
        p: str = "\u2514" if is_last else "\u251c"
        if fn in seen:
            message = f"{indent}{p}{f'\u2500{_GREY}'}"
            message = f"{message}\u00b7merge: ({fn}){_RESET}"
            return scaper.log(message=message)
        seen.add(fn)
        ok: bool = supports(fn)
        inner: str = f"\u00b7{fn}" + ("" if ok else " (not supported)")
        message = f"{indent}{p}{'\u2500' if ok else f'\u2500{_RED}'}"
        message = f"{message}{inner}{_RESET if not ok else ''}"
        scaper.log(message=message)
        ch: list[AutogradFunction] = [c for c, _ in fn.next_functions if c]
        for i, c in enumerate(ch):
            dfs(c, indent + ("  " if is_last else "| "), i == len(ch) - 1)

    # root
    ok: bool = supports(grad_fn=r)
    root_line: str = f"\u00b7{r}" + ("" if ok else " (not supported)")
    message = f"{'\u252C' if ok else f'\u252C{_RED}'}"
    message = f"{message}{root_line}{_RESET if not ok else ''}"
    scaper.log(message=message)
    # recurse
    roots: list[AutogradFunction] = [c for c, _ in r.next_functions if c]
    for i, c in enumerate(roots):
        dfs(c, "", i == len(roots) - 1)
    scaper.restore_logger_handlers()
    return None
