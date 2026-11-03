"""Public mixed_quant_sparse_flash_mla operator with registered implementations."""

import torch

from ..utils._transformer_compat import OperatorUnavailableError
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._backend import attention_family, sparse_host_family
from ._sparse_indices import logical_to_physical


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla(q, **kwargs):
    family = attention_family(True)
    if kwargs.get("quant_mode", 1) != 1:
        raise OperatorUnavailableError(
            "The migrated paired KV reader/writer contract supports quant_mode=1"
        )
    width = 640 if family.legacy else 608
    for name in ("ori_kv", "cmp_kv"):
        cache = kwargs.get(name)
        if cache is not None and cache.shape[-1] != width:
            raise ValueError(f"{name} must use the selected {width}B KV encoding")
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    if family.legacy:
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
        if args.pop("quant_mode", 1) != 1:
            raise OperatorUnavailableError(
                "Legacy packed KV supports quant_mode=1 only"
            )
        args.update(kv_quant_mode=1, tile_size=64)
        if args.get("rope_head_dim", 64) != 64:
            raise OperatorUnavailableError("Legacy packed KV requires rope_head_dim=64")
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
            family = sparse_host_family(True)
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
    # Keyword quant_mode works with both released 9.1 and 9.2 schemas.
    out, lse = family.ops[1](q, **args)
    if not kwargs.get("return_softmax_lse", False):
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse


@register_impl(
    "attention.mixed_quant_sparse_flash_mla",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla(q, **kwargs):
    from ...implementation.torch_native.attention.mixed_quant_sparse_flash_mla import (
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

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
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
