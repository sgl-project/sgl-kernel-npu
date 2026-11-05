"""Public sparse_flash_mla operator with registered implementations."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    require_op,
)
from .attention_utils import logical_to_physical


@register_impl("attention.sparse_flash_mla", "auto")
def _auto_sparse_flash_mla(q, **kwargs):
    return get_impl_by_name("attention.sparse_flash_mla", cann_backend())(q, **kwargs)


@register_impl("attention.sparse_flash_mla", "cann_ops_transformer")
def _cann_sparse_flash_mla(q, **kwargs):
    op = require_op(
        "cann_ops_transformer", "sparse_flash_mla", backend="cann_ops_transformer"
    )
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    out, lse = op(q, **args)
    if not kwargs.get("return_softmax_lse", False):
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse


@register_impl("attention.sparse_flash_mla", "cann_9_1")
def _cann_9_1_sparse_flash_mla(q, **kwargs):
    indices = kwargs.get("ori_sparse_indices")
    namespace, binding = (
        ("npu", "sparse_attn_sharedkv")
        if indices is not None
        else ("custom", "npu_sparse_attn_sharedkv")
    )
    op = require_op(namespace, binding, backend="cann_9_1")
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    if args.get("layout_q", "BSND") == "TND":
        # cann_9_1 TND uses cu_seqlens_q and rejects a non-null seqused_q.
        # Its tiling diagnostic misleadingly reports that seqused_q is null.
        args.pop("seqused_q", None)
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


@register_impl("attention.sparse_flash_mla", "torch_native")
def _torch_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.attention.attention import sparse_flash_mla

    return sparse_flash_mla(q, **kwargs)


@register_operator("attention.sparse_flash_mla")
def sparse_flash_mla(q, *, impl="auto", **kwargs):
    """Compute BF16 shared-KV sparse attention; return output and LSE.

    impl: auto (SDK-selected), cann_9_1, cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.sparse_flash_mla", impl)(q, **kwargs)
