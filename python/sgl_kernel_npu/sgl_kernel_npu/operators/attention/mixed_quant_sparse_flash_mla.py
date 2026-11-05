"""Public mixed_quant_sparse_flash_mla operator with registered implementations."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    require_op,
)
from .attention_utils import logical_to_physical


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "auto",
    supported_generations={Generation.ARCH35},
)
def _auto_mixed_quant_sparse_flash_mla(q, **kwargs):
    backend = cann_backend()
    if backend == "cann_9_1":
        backend = (
            "ascendc"
            if _uses_native_sparse_inputs(kwargs)
            else "ops_transformer_cann910"
        )
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla", backend)(
        q, **kwargs
    )


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "cann_ops_transformer_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla(q, **kwargs):
    op = require_op(
        "cann_ops_transformer",
        "mixed_quant_sparse_flash_mla",
        backend="cann_ops_transformer_cann920",
    )
    if kwargs.get("quant_mode", 1) != 1:
        raise OperatorUnavailableError(
            "The migrated paired KV reader/writer contract supports quant_mode=1"
        )
    width = 608
    for name in ("ori_kv", "cmp_kv"):
        cache = kwargs.get(name)
        if cache is not None and cache.shape[-1] != width:
            raise ValueError(f"{name} must use the selected {width}B KV encoding")
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    if args.get("ori_mask_mode", 4) == 0:
        args.update(ori_win_left=-1, ori_win_right=-1)
    out, lse = op(q, **args)
    if not kwargs.get("return_softmax_lse", False):
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse


def _uses_native_sparse_inputs(kwargs):
    return kwargs.get("ori_sparse_indices") is not None


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "ops_transformer_cann910",
    supported_generations={Generation.ARCH35},
)
def _ops_transformer_cann910(q, **kwargs):
    if _uses_native_sparse_inputs(kwargs):
        raise OperatorUnavailableError(
            "Explicit sparse inputs require impl='ascendc' on CANN 9.1"
        )
    return _cann_9_1_mixed_quant_sparse_flash_mla(q, **kwargs)


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "ascendc",
    supported_generations={Generation.ARCH35},
)
def _ascendc(q, **kwargs):
    if not _uses_native_sparse_inputs(kwargs):
        raise OperatorUnavailableError(
            "Window-only inputs require impl='ops_transformer_cann910' on CANN 9.1"
        )
    return _cann_9_1_mixed_quant_sparse_flash_mla(q, **kwargs)


def _cann_9_1_mixed_quant_sparse_flash_mla(q, **kwargs):
    indices = kwargs.get("ori_sparse_indices")
    namespace, binding = (
        ("sgl_native_dsv4", "kv_quant_sparse_attn_sharedkv")
        if indices is not None
        else ("custom", "npu_kv_quant_sparse_attn_sharedkv")
    )
    op = require_op(namespace, binding, backend="cann_9_1")
    if kwargs.get("quant_mode", 1) != 1:
        raise OperatorUnavailableError(
            "The migrated paired KV reader/writer contract supports quant_mode=1"
        )
    width = 640
    for name in ("ori_kv", "cmp_kv"):
        cache = kwargs.get(name)
        if cache is not None and cache.shape[-1] != width:
            raise ValueError(f"{name} must use the selected {width}B KV encoding")
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    args.pop("seqused_cmp_kv", None)
    args.pop("cmp_residual_kv", None)
    args["seqused_kv"] = args.pop("seqused_ori_kv", None)
    layout = args.get("layout_kv", "BSND")
    if layout == "PA_BBND":
        args["layout_kv"] = "PA_ND"
    elif layout not in ("BSND", "TND"):
        raise ValueError(f"Unsupported shared-KV layout: {layout}")
    if args.pop("topk_value_mode", 1) != 1:
        raise OperatorUnavailableError(
            "cann_9_1 attention only supports logical compressed top-k indices"
        )
    if args.pop("cmp_topk_length", None) is not None:
        raise OperatorUnavailableError(
            "cann_9_1 attention does not accept cmp_topk_length"
        )
    args.pop("ori_topk_length", None)
    args.pop("ori_topk", None)
    if args.get("ori_mask_mode", 4) == 0:
        # Native DSpark selects explicit indices by tensor presence, with the
        # cann_9_1 SWA/window metadata ABI.
        args["ori_mask_mode"] = 4
    if not args.get("has_cmp_kv", True) or args.get("cmp_ratio", 1) == 1:
        args["cmp_mask_mode"] = 3
    if args.pop("quant_mode", 1) != 1:
        raise OperatorUnavailableError("cann_9_1 packed KV supports quant_mode=1 only")
    args.update(kv_quant_mode=1, tile_size=64)
    if args.get("rope_head_dim", 64) != 64:
        raise OperatorUnavailableError("cann_9_1 packed KV requires rope_head_dim=64")
    if kwargs.get("ori_mask_mode", 4) == 0 and indices is None:
        raise ValueError(
            "cann_9_1 explicit sparse attention requires ori_sparse_indices"
        )
    if indices is not None:
        if kwargs.get("cmp_kv") is not None:
            raise OperatorUnavailableError(
                "cann_9_1 DSpark explicit original KV cannot be combined with compressed KV"
            )
        args["ori_sparse_indices"] = logical_to_physical(
            indices,
            kwargs.get("ori_topk_length"),
            q,
            kwargs.get("ori_block_table"),
            kwargs["ori_kv"],
            kwargs.get("cu_seqlens_q"),
            kwargs.get("layout_q", "BSND"),
        )
    args.pop("has_ori_kv", None)
    args.pop("has_cmp_kv", None)
    out, lse = op(q, **args)
    if not kwargs.get("return_softmax_lse", False):
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.attention.attention import (
        mixed_quant_sparse_flash_mla,
    )

    return mixed_quant_sparse_flash_mla(q, **kwargs)


@register_operator("attention.mixed_quant_sparse_flash_mla")
def mixed_quant_sparse_flash_mla(
    q,
    *,
    quant_mode=1,
    rope_head_dim=64,
    cmp_ratio=1,
    softmax_scale=1.0,
    impl="auto",
    **kwargs,
):
    """Compute A5 mixed-quant sparse attention; return output and LSE.

    impl: auto (SDK-selected), ops_transformer_cann910, ascendc, cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla", impl)(
        q,
        quant_mode=quant_mode,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        softmax_scale=softmax_scale,
        **kwargs,
    )
