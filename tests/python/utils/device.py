"""Device policy for combined NPU accuracy/performance tests."""

from contextlib import contextmanager


def validate_npu_options(index, workers):
    if index is not None and index < 0:
        raise ValueError("--npu-device must be a nonnegative visible device index")
    if workers not in (None, 0, 1, "0", "1"):
        raise ValueError(
            "Combined accuracy/performance tests require a single worker; "
            "remove -n or use -n 1. Automatic multi-worker NPU allocation is disabled."
        )


@contextmanager
def use_npu_device(torch, index=None):
    """Use current NPU by default and restore device settings after each case."""
    previous_default = torch.get_default_device()
    previous_npu = torch.npu.current_device()
    try:
        if index is not None:
            torch.npu.set_device(index)
        torch.set_default_device("npu")
        yield
    finally:
        try:
            if index is not None:
                torch.npu.set_device(previous_npu)
        finally:
            torch.set_default_device(previous_default)
