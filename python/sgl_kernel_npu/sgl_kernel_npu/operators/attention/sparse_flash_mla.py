"""Public sparse_flash_mla operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from .attention_utils import cann_9_1_dspark_kv_indices_to_physical_slots
from .sparse_flash_mla_metadata import sparse_flash_mla_metadata  # noqa: F401


@register_impl(
    "attention.sparse_flash_mla", "auto", supported_generations={Generation.ARCH22}
)
def _auto_sparse_flash_mla(q, **kwargs):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = (
            "ascendc"
            if _uses_native_sparse_inputs(kwargs)
            else "ops_transformers_cann910"
        )
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("attention.sparse_flash_mla", backend)(q, **kwargs)


@register_impl(
    "attention.sparse_flash_mla",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_sparse_flash_mla(q, **kwargs):
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    from ..._third_party.sgl_ops_transformer import ops

    args = {
        key: value
        for key, value in kwargs.items()
        if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
    }
    out, lse = ops.sparse_flash_mla(q, **args)
    if not kwargs.get("return_softmax_lse", False):
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return out, lse


def _uses_native_sparse_inputs(kwargs):
    return kwargs.get("ori_sparse_indices") is not None


@register_impl("attention.sparse_flash_mla", "ops_transformers_cann910")
def _ops_transformers_cann910(q, **kwargs):
    if _uses_native_sparse_inputs(kwargs):
        raise RuntimeError("Explicit sparse inputs require impl='ascendc' on CANN 9.1")
    return _cann_9_1_sparse_flash_mla(q, **kwargs)


@register_impl("attention.sparse_flash_mla", "ascendc")
def _ascendc(q, **kwargs):
    if not _uses_native_sparse_inputs(kwargs):
        raise RuntimeError(
            "Window-only inputs require impl='ops_transformers_cann910' on CANN 9.1"
        )
    return _cann_9_1_sparse_flash_mla(q, **kwargs)


def _cann_9_1_sparse_flash_mla(q, **kwargs):
    indices = kwargs.get("ori_sparse_indices")
    namespace, binding = (
        ("npu", "sparse_attn_sharedkv")
        if indices is not None
        else ("custom", "npu_sparse_attn_sharedkv")
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
    if kwargs.get("ori_mask_mode", 4) == 0 and indices is None:
        raise ValueError(
            "cann_9_1 explicit sparse attention requires ori_sparse_indices"
        )
    if indices is not None:
        if kwargs.get("cmp_kv") is not None:
            raise RuntimeError(
                "cann_9_1 DSpark explicit original KV cannot be combined with compressed KV"
            )
        args["ori_sparse_indices"] = cann_9_1_dspark_kv_indices_to_physical_slots(
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
    from ...implementation.torch_native.attention.metadata import ReferenceMetadata

    # DSpark and migrated tests carry independently built reference descriptors.
    # The upstream reference additionally covers BSND/SWA/CSA/HCA inputs with
    # opaque production metadata and the CANN 9.2 LSE layout.
    if (
        isinstance(kwargs.get("metadata"), ReferenceMetadata)
        or kwargs.get("ori_sparse_indices") is not None
    ):
        from ...implementation.torch_native.attention.attention import sparse_flash_mla
    else:
        from ...implementation.torch_native.attention.sparse_flash_mla import (
            sparse_flash_mla,
        )
    return sparse_flash_mla(
        q,
        **{
            key: value
            for key, value in kwargs.items()
            if key not in ("cu_seqlens_q_cpu", "seqused_ori_kv_cpu")
        },
    )


@register_operator("attention.sparse_flash_mla")
def sparse_flash_mla(q, *, impl="auto", **kwargs):
    """Compute BF16 shared-KV sparse attention; return output and LSE.

    impl: auto (SDK-selected), ops_transformers_cann910, ascendc, cann_ops_transformers_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.sparse_flash_mla", impl)(q, **kwargs)
