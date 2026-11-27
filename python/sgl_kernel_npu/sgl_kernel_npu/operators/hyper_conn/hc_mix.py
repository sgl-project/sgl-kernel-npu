"""Public hyper-connection gated branch mix with Triton default and Ascend C backend."""

import torch
import torch.nn.functional as F

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None

# Unified Buffer sizing caps the per-branch hidden size, the low-rank width and
# the branch count for the explicit Ascend C backend.
MAX_HIDDEN_SIZE = 4096
MAX_LOW_RANK = 4096
MAX_HC_COUNT = 8
WIDTH_ALIGNMENT = 16


def _check_inputs(x, down, up, hc, hs):
    """Reject inputs outside the operator contract before any kernel launch."""
    if any(not isinstance(t, torch.Tensor) for t in (x, down, up)):
        raise ValueError("hc_mix requires BF16 tensor inputs")
    if any(t.dtype != torch.bfloat16 for t in (x, down, up)):
        raise ValueError("hc_mix requires BF16 inputs")
    if x.device.type != "npu" or any(t.device != x.device for t in (down, up)):
        raise ValueError("all inputs must be on the same NPU")
    if x.ndim != 2 or down.ndim != 2 or up.ndim != 2:
        raise ValueError("x, down and up must be 2-D")
    if any(not t.is_contiguous() for t in (x, down, up)):
        raise ValueError("x, down and up must be contiguous")
    if (
        isinstance(hc, bool)
        or not isinstance(hc, int)
        or hc <= 0
        or hc > MAX_HC_COUNT
        or hc & (hc - 1)
    ):
        raise ValueError("hc must be a power of two in [1, 8]")
    if (
        isinstance(hs, bool)
        or not isinstance(hs, int)
        or hs <= 0
        or hs > MAX_HIDDEN_SIZE
        or hs % WIDTH_ALIGNMENT
    ):
        raise ValueError("hs must be a multiple of 16 up to 4096")
    total_width = hc * hs
    if x.shape[1] != total_width:
        raise ValueError("x must have shape [rows, hc * hs]")
    if down.shape[1] != total_width:
        raise ValueError("down must have shape [low_rank, hc * hs]")
    low_rank = down.shape[0]
    if low_rank <= 0 or low_rank > MAX_LOW_RANK or low_rank % WIDTH_ALIGNMENT:
        raise ValueError("low_rank (down.shape[0]) must be a multiple of 16 up to 4096")
    if up.shape != (total_width, low_rank):
        raise ValueError("up must have shape [hc * hs, low_rank]")
    if any(t.requires_grad for t in (x, down, up)):
        raise ValueError("hc_mix is inference only")


@register_impl(
    "hyper_conn.hc_mix",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "hyper_conn.hc_mix",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _triton_impl(x, down, up, hc, hs):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import mix

        _triton_fn = mix
    return _triton_fn(x, down, up, hc, hs)


# Native CANN/ACLNN AIV kernels declare SUPPORTED_ARCHS 2201, so impl="ascendc"
# is registered for arch22 only while auto remains on the Triton path.
@register_impl(
    "hyper_conn.hc_mix",
    "ascendc",
    supported_generations={Generation.ARCH22},
)
def _ascendc_impl(x, down, up, hc, hs):
    if x.shape[0] == 0:
        return x.new_empty((0, hs))
    low = F.linear(x, down)
    activated = torch.ops.npu.hyper_conn_scaled_silu(low, float(hc))
    gate_logits = F.linear(activated, up)
    return torch.ops.npu.hyper_conn_branch_mix(x, gate_logits, hc)


@register_impl(
    "hyper_conn.hc_mix",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _torch_native_impl(x, down, up, hc, hs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import mix

        _torch_native_fn = mix
    return _torch_native_fn(x, down, up, hc, hs)


@register_operator("hyper_conn.hc_mix")
def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    """Compute the low-rank gated average of hyper-connection branches.

    The default ``impl="auto"`` selects the Triton/native FP32-intermediate
    path (``hc=4, hs=2560, low_rank=320``) on both ``arch22`` and ``arch35``.
    Explicit ``impl="ascendc"`` selects the arch22 CANN/ACLNN path using BF16
    ``F.linear`` projections (including pre-converted ``FRACTAL_NZ`` ``up``
    weights) and FP32-internal Ascend C AIV kernels. ``torch_native`` is an
    independent CPU FP64 reference.
    """
    _check_inputs(x, down, up, hc, hs)
    impl_fn = get_impl_by_name("hyper_conn.hc_mix", impl)
    return impl_fn(x, down, up, hc, hs)
