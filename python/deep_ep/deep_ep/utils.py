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
torch.set_printoptions(profile="full")


def get_simplify_tensor(arg):
    if type(arg) in (tuple, list):
        return ", ".join([get_simplify_tensor(a) for a in arg])
    elif isinstance(arg, torch.Tensor):
        return str((arg.dtype, arg.shape))
    return str(arg)


def log_parameters(input_name_full_tensor=None, output_idx_full_tensor=None):
    """
    A decorator for printing the input and output of functions.
    By default, tensors print dtype and shape.

    Arguments:
        input_name_full_tensor: input names of tensors that need to be fully printed.
        output_idx_full_tensor: ouput indexs of the tensor that needs to be fully printed.
    """
    if input_name_full_tensor is None:
        input_name_full_tensor = []
    if output_idx_full_tensor is None:
        output_idx_full_tensor = []

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
                        f"{k}: {v if k in input_name_full_tensor else get_simplify_tensor(v)}"
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
                        if idx in output_idx_full_tensor:
                            result_str_list.append(str(v))
                        else:
                            result_str_list.append(get_simplify_tensor(v))
                    result_str = "\n".join(result_str_list)
                else:
                    if 0 in output_idx_full_tensor:
                        result_str = str(result)
                    else:
                        result_str = get_simplify_tensor(result)

                logger.debug(
                    "[rank %s]" % rank_info
                    + f"Function {func.__name__} returned:\n{result_str}\n{func.__name__} returned value finish."
                )

            return result

        return wrapper

    return log_parameters_decorator
