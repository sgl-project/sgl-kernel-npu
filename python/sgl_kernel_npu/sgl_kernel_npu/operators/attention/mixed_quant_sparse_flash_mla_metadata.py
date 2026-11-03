"""Public mixed_quant_sparse_flash_mla_metadata operator with registered implementations."""

import torch

from ..utils._transformer_compat import OperatorUnavailableError
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._backend import attention_family, sparse_host_family


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    family = attention_family(True)
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    if not family.legacy:
        return family.ops[0](**args)
    sparse = (
        kwargs.get("ori_mask_mode", 4) == 0
        or kwargs.get("ori_topk_length") is not None
        or kwargs.get("ori_topk", 0) > 0
    )
    if sparse and kwargs.get("has_cmp_kv", False):
        raise OperatorUnavailableError(
            "Legacy DSpark explicit original KV cannot be combined with compressed KV"
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
        raise OperatorUnavailableError("Legacy packed KV supports quant_mode=1 only")
    args.update(kv_quant_mode=1, tile_size=64)
    if args.get("rope_head_dim", 64) != 64:
        raise OperatorUnavailableError("Legacy packed KV requires rope_head_dim=64")
    # Device metadata has no seqused_q scheduling-bound argument.
    args.pop("seqused_q", None)
    if sparse:
        family = sparse_host_family(True)
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
                    f"Legacy host metadata requires an existing CPU int32 {public}; no device copy is performed during capture"
                )
            count = kwargs["batch_size"] + (private == "cu_seqlens_q")
            args[private] = value[:count].contiguous()
    return family.ops[0](**args)


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.torch_native.attention.mixed_quant_sparse_flash_mla_metadata import (
        mixed_quant_sparse_flash_mla_metadata,
    )

    return mixed_quant_sparse_flash_mla_metadata(
        **{
            key: value
            for key, value in kwargs.items()
            if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
        }
    )


@register_operator("attention.mixed_quant_sparse_flash_mla_metadata")
def mixed_quant_sparse_flash_mla_metadata(*, impl="auto", **kwargs):
    """Build A5 mixed-quant attention metadata.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla_metadata", impl)(
        **kwargs
    )
