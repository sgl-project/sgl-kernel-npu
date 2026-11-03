"""Public sparse_flash_mla operator with registered implementations."""

import torch

from ..utils._transformer_compat import OperatorUnavailableError
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._backend import attention_family, sparse_host_family
from ._sparse_indices import logical_to_physical


@register_impl("attention.sparse_flash_mla", "auto")
@register_impl("attention.sparse_flash_mla", "cann_ops_transformer")
def _cann_sparse_flash_mla(q, **kwargs):
    family = attention_family(False)
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    if family.legacy:
        if args.get("layout_q", "BSND") == "TND":
            # Legacy TND uses cu_seqlens_q and rejects a non-null seqused_q.
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
                "Legacy attention only supports logical compressed top-k indices"
            )
        if args.pop("cmp_topk_length", None) is not None:
            raise OperatorUnavailableError(
                "Legacy attention does not accept cmp_topk_length"
            )
        args.pop("ori_topk_length", None)
        args.pop("ori_topk", None)
        if args.get("ori_mask_mode", 4) == 0:
            # Native DSpark selects explicit indices by tensor presence, with the
            # legacy SWA/window metadata ABI.
            args["ori_mask_mode"] = 4
        if not args.get("has_cmp_kv", True) or args.get("cmp_ratio", 1) == 1:
            args["cmp_mask_mode"] = 3
        indices = kwargs.get("ori_sparse_indices")
        if kwargs.get("ori_mask_mode", 4) == 0 and indices is None:
            raise ValueError(
                "Legacy explicit sparse attention requires ori_sparse_indices"
            )
        if indices is not None:
            if kwargs.get("cmp_kv") is not None:
                raise OperatorUnavailableError(
                    "Legacy DSpark explicit original KV cannot be combined with compressed KV"
                )
            family = sparse_host_family(False)
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
    # The paired family supplies the released or legacy compute binding.
    out, lse = family.ops[1](q, **args)
    if not kwargs.get("return_softmax_lse", False):
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse


@register_impl("attention.sparse_flash_mla", "torch_native")
def _torch_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.attention.sparse_flash_mla import (
        sparse_flash_mla,
    )

    return sparse_flash_mla(q, **kwargs)


@register_operator("attention.sparse_flash_mla")
def sparse_flash_mla(q, *, impl="auto", **kwargs):
    """Compute BF16 shared-KV sparse attention; return output and LSE.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.sparse_flash_mla", impl)(q, **kwargs)
