"""Ascend 950 ZeRCMoE with explicit, process-lifetime communication state."""

from importlib import import_module
from threading import RLock
from types import MappingProxyType

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_BACKEND = None
_GROUPS = {}
_GROUP_LOCK = RLock()


def _load_backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = import_module(
            "..._third_party.zerc_moe.ops.mega_moe", package=__package__
        )
    return _BACKEND


def _buffer_metadata(name, *, writable=False):
    """Expose backend metadata; permit mutation only for informational fields."""

    def get(self):
        return getattr(self._buffer, name)

    def set_value(self, value):
        with self._lock:
            if self._closed:
                raise ValueError("buffer has been destroyed")
            setattr(self._buffer, name, value)
            if name in (
                "num_max_tokens_per_rank",
                "num_topk",
                "hidden",
                "intermediate_hidden",
            ):
                self._config[name] = value

    return property(get, set_value if writable else None)


class ZercMoeBuffer:
    """Own a single ZeRCMoE context until the group and process are torn down.

    Use a distinct process group from other communication backends. A context
    cannot be recreated for the same group after destruction or a failed create.
    All ranks must configure the same parameters and call in the same order.
    Each call must use the same token count, hidden dimension and top-k across
    ranks. Calls sharing a buffer must not overlap, including on different streams.
    """

    rank_id = _buffer_metadata("rank_id", writable=True)
    group_name = _buffer_metadata("group_name", writable=True)
    ep_world_size = _buffer_metadata("ep_world_size")
    ccl_buffer_size = _buffer_metadata("ccl_buffer_size")
    num_experts = _buffer_metadata("num_experts")
    max_recv_token_num = _buffer_metadata("max_recv_token_num")
    num_max_tokens_per_rank = _buffer_metadata("num_max_tokens_per_rank", writable=True)
    num_topk = _buffer_metadata("num_topk", writable=True)
    hidden = _buffer_metadata("hidden", writable=True)
    intermediate_hidden = _buffer_metadata("intermediate_hidden", writable=True)
    dispatch_quant_mode = _buffer_metadata("dispatch_quant_mode")
    dispatch_quant_out_dtype = _buffer_metadata("dispatch_quant_out_dtype")
    combine_quant_mode = _buffer_metadata("combine_quant_mode")
    comm_alg = _buffer_metadata("comm_alg")
    global_bs = _buffer_metadata("global_bs")

    def __init__(self, group, config, buffer):
        self.group = group
        self._config = config
        self._allocated_max_tokens = config["num_max_tokens_per_rank"]
        self.config = MappingProxyType(config)
        self._buffer = buffer
        self._closed = False
        self._lock = RLock()

    @property
    def context(self):
        """Return the live backend context; do not retain it after destruction."""
        with self._lock:
            if self._closed:
                raise ValueError("buffer has been destroyed")
            return self._buffer.context

    def destroy(self):
        """Release the device buffer after all operations have completed."""
        with self._lock:
            if self._closed:
                return
            # Even if destruction fails, reuse is unsafe: the HCCL context may
            # still exist and cannot be recreated for this process group.
            self._closed = True
            self._buffer.destroy()


def create_zerc_moe_buffer(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    max_recv_token_num=0,
    dispatch_quant_mode=0,
    dispatch_quant_out_dtype=28,
    combine_quant_mode=0,
    comm_alg="",
):
    """Create one group-owned buffer for the Ascend 950 source implementation.

    ``dispatch_quant_mode`` is 0 for BF16 or 4 for MXFP8; MXFP8 also requires
    ``dispatch_quant_out_dtype`` to be the CANN FP8 enum 23 or 24. The buffer
    capacity must cover every call and must be consistent across all ranks.
    A nonzero max_recv_token_num is safe only if it covers all routed tokens.
    """
    if group is None:
        raise ValueError("an explicit process group is required")
    world_size = torch.distributed.get_world_size(group)
    if not 2 <= world_size <= 768:
        raise ValueError("ZeRCMoE requires a group of 2 to 768 ranks")
    if not 1 <= num_experts <= 1024 or num_experts % world_size:
        raise ValueError("num_experts must be <= 1024 and divisible by group size")
    if num_max_tokens_per_rank < 1 or hidden < 1:
        raise ValueError("buffer capacity and hidden dimension must be positive")
    if not 1 <= num_topk <= num_experts or max_recv_token_num < 0:
        raise ValueError("invalid top-k or max_recv_token_num")
    if dispatch_quant_mode not in (0, 4):
        raise ValueError("only BF16 (0) and MXFP8 (4) are supported")
    if dispatch_quant_mode == 0 and hidden % 16:
        raise ValueError("BF16 hidden dimension must be divisible by 16")
    if dispatch_quant_mode == 4 and dispatch_quant_out_dtype not in (23, 24):
        raise ValueError("MXFP8 dispatch dtype must be CANN enum 23 or 24")
    if combine_quant_mode != 0 or comm_alg != "":
        raise ValueError("combine_quant_mode must be 0 and comm_alg must be empty")
    get_impl_by_name("moe.zerc_moe", "auto")
    config = {
        "num_experts": num_experts,
        "world_size": world_size,
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "num_topk": num_topk,
        "hidden": hidden,
        "intermediate_hidden": intermediate_hidden,
        "max_recv_token_num": max_recv_token_num,
        "dispatch_quant_mode": dispatch_quant_mode,
        "dispatch_quant_out_dtype": dispatch_quant_out_dtype,
    }
    # Keep a strong reference for the process lifetime. HCCL Engine contexts
    # survive buffer.destroy(); letting an object ID be reused is not safe.
    backend = _load_backend()
    with _GROUP_LOCK:
        group_id = id(group)
        if group_id in _GROUPS:
            raise RuntimeError("ZeRCMoE context already created for this group")
        _GROUPS[group_id] = group
    buffer = backend.get_symm_buffer_for_mega_moe(
        group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        max_recv_token_num=max_recv_token_num,
        dispatch_quant_mode=dispatch_quant_mode,
        dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        combine_quant_mode=combine_quant_mode,
        comm_alg=comm_alg,
    )
    return ZercMoeBuffer(group, config, buffer)


@register_impl("moe.zerc_moe", "auto", supported_generations={Generation.ARCH35})
@register_impl("moe.zerc_moe", "ascendc", supported_generations={Generation.ARCH35})
def _ascendc_impl(*args, **kwargs):
    return _load_backend().mega_moe(*args, **kwargs)


@register_operator("moe.zerc_moe")
def zerc_moe(
    x,
    topk_ids,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    scales=None,
    l1_weights_sf=None,
    l2_weights_sf=None,
    x_active_mask=None,
    weight1_type=28,
    weight2_type=28,
    timer=None,
    l1_weights_shared=None,
    l2_weights_shared=None,
    l1_weights_shared_sf=None,
    l2_weights_shared_sf=None,
    *,
    impl="auto",
):
    """Run the Ascend 950 eager kernel; auto selects its Ascend C backend.

    Input x and routing weights are BF16; ids are int32. Weight lists contain
    one stacked tensor each. A buffer created by create_zerc_moe_buffer
    must be shared across calls. Each token's expert IDs must be in range and
    distinct. All ranks must use the same token count, hidden dimension and top-k
    for each call. Calls using the same buffer must not overlap across streams. Shared experts are
    supported only in eager mode. All ranks must finish and synchronize device
    work before destroying the buffer.
    """
    if not isinstance(sym_buffer, ZercMoeBuffer) or sym_buffer._closed:
        raise ValueError("sym_buffer must be a live ZercMoeBuffer")
    if scales is not None or x_active_mask is not None:
        raise ValueError("scales and x_active_mask must be None")
    if weight1_type != 28 or weight2_type != 28:
        raise ValueError("weight1_type and weight2_type must be 28")
    config = sym_buffer.config
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError("x must be a BF16 [tokens, hidden] tensor")
    if not 1 <= x.shape[0] <= min(sym_buffer._allocated_max_tokens, 262144):
        raise ValueError("token count exceeds buffer capacity or is zero")
    if x.shape[1] < 1:
        raise ValueError("hidden dimension must be positive")
    if config["dispatch_quant_mode"] == 0 and x.shape[1] % 16:
        raise ValueError("BF16 hidden dimension must be divisible by 16")
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.bfloat16:
        raise ValueError("topk_ids must be int32 and topk_weights BF16")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != x.shape[0]
        or not 1 <= topk_ids.shape[1] <= config["num_experts"]
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("routing tensors must have shape [tokens, topk]")
    if config["max_recv_token_num"] > (
        x.shape[0] * config["world_size"] * topk_ids.shape[1]
    ):
        raise ValueError("max_recv_token_num exceeds the routed token count")
    # As in the original implementation, the host tiling check compares the
    # actual shape's CCL requirement with the allocated buffer size. Created
    # hidden/top-k values size the allocation, not hard per-call limits.
    weight_lists = (l1_weights, l2_weights)
    optional_lists = (
        l1_weights_sf,
        l2_weights_sf,
        l1_weights_shared,
        l2_weights_shared,
        l1_weights_shared_sf,
        l2_weights_shared_sf,
    )
    if any(not isinstance(value, (list, tuple)) for value in weight_lists):
        raise ValueError("expert weights must be lists of tensors")
    if any(
        value is not None and not isinstance(value, (list, tuple))
        for value in optional_lists
    ):
        raise ValueError("optional weights and scales must be lists of tensors")
    # The original C++ binding treats empty optional lists as absent.
    (
        l1_weights_sf,
        l2_weights_sf,
        l1_weights_shared,
        l2_weights_shared,
        l1_weights_shared_sf,
        l2_weights_shared_sf,
    ) = (value if value else None for value in optional_lists)
    optional_lists = (
        l1_weights_sf,
        l2_weights_sf,
        l1_weights_shared,
        l2_weights_shared,
        l1_weights_shared_sf,
        l2_weights_shared_sf,
    )
    if len(l1_weights) != 1 or len(l2_weights) != 1:
        raise ValueError("each weight list must contain one stacked tensor")
    local_experts = config["num_experts"] // config["world_size"]
    if l1_weights[0].ndim != 3 or l2_weights[0].ndim != 3:
        raise ValueError("expert weights must be three-dimensional")
    # The buffer's intermediate_hidden is informational in the original API;
    # each call derives its intermediate size from the actual weight tensors.
    intermediate = l2_weights[0].shape[2]
    if intermediate < 1:
        raise ValueError("expert intermediate dimension must be positive")
    if tuple(l1_weights[0].shape) != (local_experts, 2 * intermediate, x.shape[1]):
        raise ValueError("invalid first expert weight shape")
    if tuple(l2_weights[0].shape) != (local_experts, x.shape[1], intermediate):
        raise ValueError("invalid second expert weight shape")
    if config["dispatch_quant_mode"] == 0:
        if (
            l1_weights[0].dtype != torch.bfloat16
            or l2_weights[0].dtype != torch.bfloat16
        ):
            raise ValueError("BF16 mode requires BF16 weights")
        if l1_weights_sf is not None or l2_weights_sf is not None:
            raise ValueError("BF16 mode does not use expert weight scales")
    else:
        expected_weight_dtype = getattr(
            torch,
            (
                "float8_e5m2"
                if config["dispatch_quant_out_dtype"] == 23
                else "float8_e4m3fn"
            ),
            None,
        )
        if expected_weight_dtype is None or any(
            t.dtype != expected_weight_dtype for t in (l1_weights[0], l2_weights[0])
        ):
            raise ValueError("MXFP8 weights must match the configured FP8 dtype")
        if (
            l1_weights_sf is None
            or l2_weights_sf is None
            or len(l1_weights_sf) != 1
            or len(l2_weights_sf) != 1
        ):
            raise ValueError("MXFP8 mode requires one scale tensor per weight list")
        expected_l1 = (local_experts, 2 * intermediate, (x.shape[1] + 63) // 64, 2)
        expected_l2 = (local_experts, x.shape[1], (intermediate + 63) // 64, 2)
        if (
            tuple(l1_weights_sf[0].shape) != expected_l1
            or tuple(l2_weights_sf[0].shape) != expected_l2
        ):
            raise ValueError("invalid MXFP8 expert scale shape")
        scale_dtype = getattr(torch, "float8_e8m0fnu", None)
        if scale_dtype is None or any(
            t.dtype != scale_dtype for t in (l1_weights_sf[0], l2_weights_sf[0])
        ):
            raise ValueError("MXFP8 scales must have E8M0 dtype")
    if (l1_weights_shared is None) != (l2_weights_shared is None):
        raise ValueError("shared expert weights must be supplied together")
    if (l1_weights_shared_sf is None) != (l2_weights_shared_sf is None):
        raise ValueError("shared expert scales must be supplied together")
    if l1_weights_shared is None and l1_weights_shared_sf is not None:
        raise ValueError("shared scales require shared expert weights")
    if l1_weights_shared is not None:
        if len(l1_weights_shared) != 1 or len(l2_weights_shared) != 1:
            raise ValueError("each shared weight list must contain one stacked tensor")
        w1, w2 = l1_weights_shared[0], l2_weights_shared[0]
        if w1.ndim != 3 or w2.ndim != 3 or not 1 <= w1.shape[0] <= 4:
            raise ValueError("shared weights must have 1 to 4 stacked experts")
        if w1.shape[1] < 2 or w1.shape[1] % 2 or w1.shape[2] != x.shape[1]:
            raise ValueError("invalid first shared expert weight shape")
        if tuple(w2.shape) != (w1.shape[0], x.shape[1], w1.shape[1] // 2):
            raise ValueError("invalid second shared expert weight shape")
        if w1.dtype != l1_weights[0].dtype or w2.dtype != l2_weights[0].dtype:
            raise ValueError("shared and routed expert weight dtypes must match")
        shared_intermediate = w1.shape[1] // 2
        alignment = 16 if config["dispatch_quant_mode"] == 0 else 64
        if shared_intermediate % alignment:
            raise ValueError("shared intermediate size has invalid alignment")
    if config["dispatch_quant_mode"] == 0 and l1_weights_shared_sf is not None:
        raise ValueError("BF16 mode does not use shared expert scales")
    if (
        config["dispatch_quant_mode"] == 4
        and l1_weights_shared is not None
        and l1_weights_shared_sf is None
    ):
        raise ValueError("MXFP8 shared experts require both scales")
    if l1_weights_shared_sf is not None:
        if len(l1_weights_shared_sf) != 1 or len(l2_weights_shared_sf) != 1:
            raise ValueError("each shared scale list must contain one tensor")
        shared_experts, shared_n, _ = l1_weights_shared[0].shape
        expected_shared_l1 = (shared_experts, shared_n, (x.shape[1] + 63) // 64, 2)
        expected_shared_l2 = (
            shared_experts,
            x.shape[1],
            (shared_n // 2 + 63) // 64,
            2,
        )
        if (
            tuple(l1_weights_shared_sf[0].shape) != expected_shared_l1
            or tuple(l2_weights_shared_sf[0].shape) != expected_shared_l2
        ):
            raise ValueError("invalid MXFP8 shared expert scale shape")
        scale_dtype = getattr(torch, "float8_e8m0fnu", None)
        if scale_dtype is None or any(
            t.dtype != scale_dtype
            for t in (l1_weights_shared_sf[0], l2_weights_shared_sf[0])
        ):
            raise ValueError("MXFP8 shared scales must have E8M0 dtype")
    tensors = [x, topk_ids, topk_weights, *l1_weights, *l2_weights]
    if timer is not None:
        tensors.append(timer)
    for optional in optional_lists:
        if optional is not None:
            tensors.extend(optional)
    if x.device.type not in ("npu", "privateuseone"):
        raise ValueError("all tensors must be on an NPU device")
    if any(t.device != x.device for t in tensors):
        raise ValueError("all tensors must be on the same NPU device")
    if sym_buffer._buffer.context.device != x.device:
        raise ValueError("input device must match the buffer context device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must use contiguous storage")
    # Contiguous PyTorch strides alone do not rule out NPU-specific layouts.
    import torch_npu

    # ACL_FORMAT_ND is 2. No implicit format conversion is safe for these
    # pointer-based kernels, especially for optional shared expert weights.
    if any(
        int(getattr(fmt, "value", fmt)) != 2
        for fmt in (torch_npu.get_npu_format(t) for t in tensors)
    ):
        raise ValueError("all tensors must use NPU ND storage format")
    fn = get_impl_by_name("moe.zerc_moe", impl)
    with sym_buffer._lock:
        if sym_buffer._closed:
            raise ValueError("sym_buffer must be a live ZercMoeBuffer")
        return fn(
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            sym_buffer._buffer,
            scales=scales,
            l1_weights_sf=l1_weights_sf,
            l2_weights_sf=l2_weights_sf,
            x_active_mask=x_active_mask,
            weight1_type=weight1_type,
            weight2_type=weight2_type,
            timer=timer,
            l1_weights_shared=l1_weights_shared,
            l2_weights_shared=l2_weights_shared,
            l1_weights_shared_sf=l1_weights_shared_sf,
            l2_weights_shared_sf=l2_weights_shared_sf,
        )
