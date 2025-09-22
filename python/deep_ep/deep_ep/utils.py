import functools
import inspect
import logging
import os
from typing import Optional, Tuple

import torch
import torch_npu
from deep_ep_cpp import Config, EventHandle


class EventOverlap:

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        pass


logger = logging.getLogger()


def log_parameters(func, head):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank_info = "unknown"
        if logger.isEnabledFor(logging.DEBUG):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            self_instance = bound_args.arguments.get('self')
            if self_instance is not None and hasattr(self_instance, 'rank'):
                rank_info = str(self_instance.rank)

            param_str = "\n".join(
                [
                    f"{k}: {v}"
                    for k, v in bound_args.arguments.items()
                    if k not in ("self", "cls")
                ]
            )
            logger.debug("[rank %s]" % rank_info + f"Calling {func.__name__} with parameters:\n{param_str}")

        result = func(*args, **kwargs)
        
        if logger.isEnabledFor(logging.DEBUG):
            result_str = str(result)
            logger.debug("[rank %s]" % rank_info + f"Function {func.__name__} returned:\n{result_str}")

        return result

    return wrapper
