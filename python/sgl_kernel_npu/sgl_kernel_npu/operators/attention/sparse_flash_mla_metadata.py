"""Public sparse_flash_mla_metadata operator with registered implementations."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    require_op,
)


@register_impl(
    "attention.sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH22},
)
def _auto_sparse_flash_mla_metadata(**kwargs):
    backend = cann_backend()
    if backend == "cann_9_1":
        backend = (
            "host" if _uses_native_sparse_inputs(kwargs) else "ops_transformer_cann910"
        )
    return get_impl_by_name("attention.sparse_flash_mla_metadata", backend)(**kwargs)


@register_impl(
    "attention.sparse_flash_mla_metadata",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
@register_impl(
    "attention.sparse_flash_mla_metadata",
    "cann_ops_transformer_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_sparse_flash_mla_metadata(**kwargs):
    op = require_op(
        "cann_ops_transformer",
        "sparse_flash_mla_metadata",
        backend="cann_ops_transformer_cann920",
    )
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    return op(**args)


def _uses_native_sparse_inputs(kwargs):
    return (
        kwargs.get("ori_mask_mode", 4) == 0
        or kwargs.get("ori_topk_length") is not None
        or kwargs.get("ori_topk", 0) > 0
    )


@register_impl("attention.sparse_flash_mla_metadata", "ops_transformer_cann910")
def _ops_transformer_cann910(**kwargs):
    if _uses_native_sparse_inputs(kwargs):
        raise OperatorUnavailableError(
            "Explicit sparse inputs require impl='host' on CANN 9.1"
        )
    return _cann_9_1_sparse_flash_mla_metadata(**kwargs)


@register_impl("attention.sparse_flash_mla_metadata", "host")
def _host(**kwargs):
    if not _uses_native_sparse_inputs(kwargs):
        raise OperatorUnavailableError(
            "Window-only inputs require impl='ops_transformer_cann910' on CANN 9.1"
        )
    return _cann_9_1_sparse_flash_mla_metadata(**kwargs)


def _cann_9_1_sparse_flash_mla_metadata(**kwargs):
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    sparse = _uses_native_sparse_inputs(kwargs)
    if sparse and kwargs.get("has_cmp_kv", False):
        raise OperatorUnavailableError(
            "cann_9_1 DSpark explicit original KV cannot be combined with compressed KV"
        )
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
    # Device metadata has no seqused_q scheduling-bound argument.
    args.pop("seqused_q", None)
    if sparse:
        for public, private in (
            ("cu_seqlens_q_cpu", "cu_seqlens_q"),
            ("seqused_ori_kv_cpu", "seqused_kv"),
        ):
            value = kwargs.get(public)
            if (
                value is None
                or value.device.type != "cpu"
                or value.dtype != torch.int32
            ):
                raise ValueError(
                    f"cann_9_1 host metadata requires an existing CPU int32 {public}; no device copy is performed during capture"
                )
            count = kwargs["batch_size"] + (private == "cu_seqlens_q")
            args[private] = value[:count].contiguous()
        args.pop("max_seqlen_q", None)
        args.pop("max_seqlen_kv", None)
    namespace, binding = (
        ("npu", "sparse_attn_sharedkv_metadata_host")
        if sparse
        else ("custom", "npu_sparse_attn_sharedkv_metadata")
    )
    op = require_op(namespace, binding, backend="cann_9_1", native=namespace == "npu")
    return op(**args)


@register_impl("attention.sparse_flash_mla_metadata", "torch_native")
def _torch_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.torch_native.attention.metadata import (
        sparse_flash_mla_metadata,
    )

    return sparse_flash_mla_metadata(
        **{
            key: value
            for key, value in kwargs.items()
            if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
        }
    )


@register_operator("attention.sparse_flash_mla_metadata")
def sparse_flash_mla_metadata(
    num_heads_q, num_heads_kv, head_dim, *, impl="auto", **kwargs
):
    """Build BF16 attention metadata.

    impl: auto (SDK-selected), ops_transformer_cann910, host, cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.sparse_flash_mla_metadata", impl)(
        num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, head_dim=head_dim, **kwargs
    )
