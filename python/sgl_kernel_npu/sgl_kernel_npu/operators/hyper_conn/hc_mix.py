"""Public hyper-connection gated branch mix operator."""

import torch
import torch.nn.functional as F

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None

# Unified Buffer sizing caps the per-branch hidden size, the low-rank width and
# the branch count.
MAX_HIDDEN_SIZE = 4096
MAX_LOW_RANK = 4096
MAX_HC_COUNT = 8
WIDTH_ALIGNMENT = 16


def _check_inputs(normed, down_weight, up_weight, hc_count, hidden_size):
    """Reject inputs outside the operator contract before any kernel launch.

    Both backends share this contract, so it is checked once here instead of
    being duplicated per backend. Only metadata is inspected; no
    device-to-host data read happens.
    """
    if any(t.dtype != torch.bfloat16 for t in (normed, down_weight, up_weight)):
        raise ValueError("hc_mix requires BF16 inputs")
    if normed.device.type != "npu" or any(
        t.device != normed.device for t in (down_weight, up_weight)
    ):
        raise ValueError("all inputs must be on the same NPU")
    if normed.ndim != 2 or down_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("normed, down_weight and up_weight must be 2-D")
    if any(not t.is_contiguous() for t in (normed, down_weight, up_weight)):
        raise ValueError("normed, down_weight and up_weight must be contiguous")
    # The branch count divides both the down-projected activation and the
    # accumulated branch sum, and the kernels multiply by the reciprocal, so
    # only a power of two stays exact.
    if (
        isinstance(hc_count, bool)
        or not isinstance(hc_count, int)
        or hc_count <= 0
        or hc_count > MAX_HC_COUNT
        or hc_count & (hc_count - 1)
    ):
        raise ValueError("hc_count must be a power of two in [1, 8]")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
        or hidden_size > MAX_HIDDEN_SIZE
        or hidden_size % WIDTH_ALIGNMENT
    ):
        raise ValueError("hidden_size must be a multiple of 16 up to 4096")
    total_width = hc_count * hidden_size
    if normed.shape[1] != total_width:
        raise ValueError("normed must have shape [rows, hc_count * hidden_size]")
    if down_weight.shape[1] != total_width:
        raise ValueError(
            "down_weight must have shape [low_rank, hc_count * hidden_size]"
        )
    low_rank = down_weight.shape[0]
    if low_rank <= 0 or low_rank > MAX_LOW_RANK or low_rank % WIDTH_ALIGNMENT:
        raise ValueError(
            "low_rank (down_weight.shape[0]) must be a multiple of 16 up to 4096"
        )
    if up_weight.shape != (total_width, low_rank):
        raise ValueError("up_weight must have shape [hc_count * hidden_size, low_rank]")
    if any(t.requires_grad for t in (normed, down_weight, up_weight)):
        raise ValueError("hc_mix is inference only")


# The native entry points are compiled by the A3-only operator set of an arch22
# build, so this implementation is not declared for arch35.
@register_impl("hyper_conn.hc_mix", "auto", supported_generations={Generation.ARCH22})
@register_impl(
    "hyper_conn.hc_mix", "ascendc", supported_generations={Generation.ARCH22}
)
def _ascendc_impl(normed, down_weight, up_weight, hc_count, hidden_size):
    if normed.shape[0] == 0:
        return normed.new_empty((0, hidden_size))
    low = F.linear(normed, down_weight)
    activated = torch.ops.npu.hyper_conn_scaled_silu(low, float(hc_count))
    gate_logits = F.linear(activated, up_weight)
    return torch.ops.npu.hyper_conn_branch_mix(normed, gate_logits, hc_count)


@register_impl("hyper_conn.hc_mix", "torch_native")
def _torch_native_impl(normed, down_weight, up_weight, hc_count, hidden_size):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.hyper_conn.hc_mix import hc_mix

        _torch_native_fn = hc_mix
    return _torch_native_fn(normed, down_weight, up_weight, hc_count, hidden_size)


@register_operator("hyper_conn.hc_mix")
def hc_mix(normed, down_weight, up_weight, hc_count, hidden_size, *, impl="auto"):
    """Compute the low-rank gated average of hyper-connection branches.

    The default ``impl="auto"`` selects Ascend C. Given a normalized residual
    ``normed`` of shape ``[rows, hc_count * hidden_size]``, this operator runs
    the down projection, divides by ``hc_count`` and applies SiLU, runs the up
    projection to produce per-branch gate logits of shape
    ``[rows, hc_count, hidden_size]``, and averages the sigmoid-gated branches
    into ``[rows, hidden_size]``:

    ``t = linear(normed, down_weight) / hc_count``

    ``g = linear(bfloat16(t * sigmoid(t)), up_weight)``

    ``out[m, j] = bfloat16(sum_c sigmoid(g[m, c, j]) * normed[m, c, j]
    / hc_count)``

    Args:
        normed (torch.Tensor): BF16 contiguous ``[rows, hc_count * hidden_size]``
            normalized residual on an NPU.
        down_weight (torch.Tensor): BF16 contiguous
            ``[low_rank, hc_count * hidden_size]`` down-projection weight on
            normed's device. ``low_rank`` must be a multiple of 16 up to 4096.
        up_weight (torch.Tensor): BF16 contiguous
            ``[hc_count * hidden_size, low_rank]`` up-projection weight on
            normed's device. Both ND and pre-converted ``FRACTAL_NZ`` weights
            are accepted.
        hc_count (int): Hyper-connection branch count; a power of two in [1, 8].
        hidden_size (int): Per-branch hidden size; a multiple of 16 up to 4096.
        impl (str): Implementation name: "auto", "ascendc" or "torch_native".
            Defaults to "auto". Keyword-only.

    Returns:
        torch.Tensor: Newly allocated BF16 result of shape
        ``[rows, hidden_size]`` on normed's device. Inputs are not modified.

    The two projections run through ``torch.nn.functional.linear``, while the
    scaled SiLU and branch reduction accumulate in ``float32`` and round to
    ``bfloat16`` only when writing their outputs (``activated`` before the up
    projection, and ``out``). Both implementations evaluate sigmoid as
    ``1 / (1 + exp(-x))`` with backend-specific transcendentals, so outputs may
    differ by up to a BF16 ulp rather than being bitwise equal.
    """
    _check_inputs(normed, down_weight, up_weight, hc_count, hidden_size)
    impl_fn = get_impl_by_name("hyper_conn.hc_mix", impl)
    return impl_fn(normed, down_weight, up_weight, hc_count, hidden_size)
