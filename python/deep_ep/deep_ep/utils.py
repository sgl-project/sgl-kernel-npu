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

def log_parameters(input_name_simplify_tensor=None, output_idx_simplify_tensor=None):
    if input_name_simplify_tensor is None:
        input_name_simplify_tensor = []
    if output_idx_simplify_tensor is None:
        output_idx_simplify_tensor = []
    def log_parameters_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rank_info = "unknown"
            if logger.isEnabledFor(logging.DEBUG):
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                self_instance = bound_args.arguments.get("self")
                if self_instance is not None and hasattr(self_instance, "rank"):
                    rank_info = str(self_instance.rank)

                param_str = "\n".join(
                    [
                        f"{k}: {(v.dtype, v.shape) if k in input_name_simplify_tensor else v}"
                        for k, v in bound_args.arguments.items()
                        if k not in ("self", "cls")
                    ]
                )
                logger.debug(
                    "[rank %s]" % rank_info
                    + f"Calling {func.__name__} with parameters:\n{param_str}"
                )

            result = func(*args, **kwargs)

            if logger.isEnabledFor(logging.DEBUG):
                if isinstance(result, tuple):
                    result_str_list = []
                    for idx, v in enumerate(result):
                        if idx in output_idx_simplify_tensor:
                            result_str_list.append(str((v.dtype, v.shape)))
                        else:
                            result_str_list.append(str(value))
                    result_str = '\n'.join(result_str_list)

                logger.debug(
                    "[rank %s]" % rank_info
                    + f"Function {func.__name__} returned:\n{result_str}\n{func.__name__} returned value finish."
                )

            return result

        return wrapper
    return log_parameters_decorator
