"""Public mixed_quant_sparse_flash_mla_metadata operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

# Fixed metadata ABI: CANN 9.1's SAS_META_SIZE and the bundled CANN 9.2
# mixed-quant metadata producer use 1024 int32 entries (4096 bytes).
# Graph setup queries this count to allocate buffers without running the producer.
_MIXED_QUANT_SPARSE_FLASH_MLA_METADATA_INT32_COUNT = 1024


def get_mixed_quant_sparse_flash_mla_metadata_size():
    """Return metadata length in int32 elements without launching a kernel."""
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        return _MIXED_QUANT_SPARSE_FLASH_MLA_METADATA_INT32_COUNT
    if cann_version.startswith("9.2."):
        return _MIXED_QUANT_SPARSE_FLASH_MLA_METADATA_INT32_COUNT
    raise RuntimeError(
        f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
    )


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "auto",
    supported_generations={Generation.ARCH35},
)
def _auto_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        if _uses_native_sparse_inputs(kwargs):
            return _cann_9_1_mixed_quant_sparse_flash_mla_metadata(**kwargs)
        backend = "ops_transformers_cann910"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla_metadata", backend)(
        **kwargs
    )


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH35},
)
def _cann_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer_dsv4 import ops

    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    if args.get("ori_mask_mode", 4) == 0:
        args.update(ori_win_left=-1, ori_win_right=-1)
    return ops.mixed_quant_sparse_flash_mla_metadata(**args)


def _uses_native_sparse_inputs(kwargs):
    return (
        kwargs.get("ori_mask_mode", 4) == 0
        or kwargs.get("ori_topk_length") is not None
        or kwargs.get("ori_topk", 0) > 0
    )


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "ops_transformers_cann910",
    supported_generations={Generation.ARCH35},
)
def _ops_transformers_cann910(**kwargs):
    if _uses_native_sparse_inputs(kwargs):
        raise RuntimeError("Explicit sparse metadata requires impl='auto' on CANN 9.1")
    return _cann_9_1_mixed_quant_sparse_flash_mla_metadata(**kwargs)


def _cann_9_1_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    sparse = _uses_native_sparse_inputs(kwargs)
    if sparse and kwargs.get("has_cmp_kv", False):
        raise RuntimeError(
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
        raise RuntimeError(
            "cann_9_1 attention only supports logical compressed top-k indices"
        )
    if args.pop("cmp_topk_length", None) is not None:
        raise RuntimeError("cann_9_1 attention does not accept cmp_topk_length")
    args.pop("ori_topk_length", None)
    args.pop("ori_topk", None)
    if args.get("ori_mask_mode", 4) == 0:
        # Native DSpark selects explicit indices by tensor presence, with the
        # cann_9_1 SWA/window metadata ABI.
        args["ori_mask_mode"] = 4
    if not args.get("has_cmp_kv", True) or args.get("cmp_ratio", 1) == 1:
        args["cmp_mask_mode"] = 3
    if args.pop("quant_mode", 1) != 1:
        raise RuntimeError("cann_9_1 packed KV supports quant_mode=1 only")
    args.update(kv_quant_mode=1, tile_size=64)
    if args.get("rope_head_dim", 64) != 64:
        raise RuntimeError("cann_9_1 packed KV requires rope_head_dim=64")
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
    namespace, binding = (
        ("npu", "kv_quant_sparse_attn_sharedkv_metadata_host")
        if sparse
        else ("custom", "npu_kv_quant_sparse_attn_sharedkv_metadata")
    )
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    if namespace == "custom":
        from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

        namespace = "sgl_cann_recipes_infer"
    op = getattr(getattr(torch.ops, namespace), binding, None)
    if op is None:
        raise RuntimeError(
            f"Missing operator {namespace}.{binding} in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    return op(**args)


@register_impl(
    "attention.mixed_quant_sparse_flash_mla_metadata",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_mixed_quant_sparse_flash_mla_metadata(**kwargs):
    from ...implementation.torch_native.attention.metadata import (
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

    impl: auto (SDK-selected), ops_transformers_cann910,
        cann_ops_transformers_cann920, or torch_native.
    CANN 9.1 explicit sparse inputs use CPU scheduling internally through auto.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.mixed_quant_sparse_flash_mla_metadata", impl)(
        **kwargs
    )
