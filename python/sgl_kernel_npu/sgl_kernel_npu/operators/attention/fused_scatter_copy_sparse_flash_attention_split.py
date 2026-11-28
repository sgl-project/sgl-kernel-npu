"""Public fused KV copy and sparse attention for split CKV/KPE caches."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP = "attention.fused_scatter_copy_sparse_flash_attention_split"
_SUPPORTED_GENERATIONS = (Generation.ARCH22,)

# Geometry fixed by the native split Copy-SFA tiling and launch ABI.
_CKV_DIM = 512
_KPE_DIM = 64
_PAGE_SIZE = 128


def validate_pd_split_copy_sfa_layout(ckv_dim, kpe_dim, page_size, dtype) -> None:
    """Validate the BF16 KV pool allocated by SGLang disaggregated decode workers.

    The standalone Copy-SFA operator also accepts FP16. It checks its tensor
    layouts in the native binding at launch.
    """
    if (ckv_dim, kpe_dim, page_size, dtype) != (
        _CKV_DIM,
        _KPE_DIM,
        _PAGE_SIZE,
        torch.bfloat16,
    ):
        raise ValueError(
            "Ascend PD split KV requires row-major BF16 MLA with "
            f"{_CKV_DIM}/{_KPE_DIM} features and page size {_PAGE_SIZE}."
        )


@register_impl(_OP, "auto", supported_generations=_SUPPORTED_GENERATIONS)
@register_impl(_OP, "ascendc", supported_generations=_SUPPORTED_GENERATIONS)
def _ascendc_impl(*args, **kwargs) -> None:
    torch.ops.npu.fused_scatter_copy_sparse_flash_attention_split.default(
        *args, **kwargs
    )


@register_impl(_OP, "torch_native", supported_generations=_SUPPORTED_GENERATIONS)
def _torch_native_impl(*args, **kwargs) -> None:
    from ...implementation.torch_native.attention.fused_scatter_copy_sparse_flash_attention_split import (
        fused_scatter_copy_sparse_flash_attention_split,
    )

    fused_scatter_copy_sparse_flash_attention_split(*args, **kwargs)


@register_operator(_OP)
def fused_scatter_copy_sparse_flash_attention_split(
    query,
    query_rope,
    hbm_ckv,
    hbm_rope,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    num_cache_tokens,
    topk_src_ids,
    topk_dst_slots,
    topk_miss_counts,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    hbm_block_table,
    dram_block_table,
    dram_kv_dev_ptr,
    attention_out,
    scale_value,
    *,
    full_visible=False,
    resident_refresh_tail=False,
    current=None,
    positions=None,
    valid=None,
    request_slots=None,
    req_to_token=None,
    resident_end=None,
    host_token_capacity=0,
    dram_rope_dev_ptr=0,
    dram_ckv=None,
    dram_rope=None,
    impl="auto",
) -> None:
    """Consume a LIM plan and write resident payload and attention in place.

    Auto selects the wheel's Ascend C implementation. Query and output use
    FP16 or BF16 TND shape [T, H, 512]; query_rope has shape [T, H, 64].
    The split HBM caches have shape [P, 128, 1, 512|64] and the registered
    host pointers address CKV and KPE allocations. Int32 plan metadata
    includes [B] lengths, capacities and counts, [T, 2048] TopK source and
    destination IDs, [T] TopK miss counts, [B, M] miss IDs and destinations,
    and two [B, pages] page tables. scale_value scales attention scores.
    The optional full-visible staging group takes current KV, positions and
    validity per query, request slots and global token mappings. It updates
    resident_end in place and bounds host indices by host_token_capacity.
    resident_refresh_tail enables overlap for the refresh suffix in a
    full-visible LIM plan.

    The eager `torch_native` reference takes host tensors in `dram_ckv` and
    `dram_rope`. The integer device addresses are only used by Ascend C.
    """
    reference_host = {}
    if impl == "torch_native":
        reference_host = {"dram_ckv": dram_ckv, "dram_rope": dram_rope}
    elif dram_ckv is not None or dram_rope is not None:
        raise ValueError("dram_ckv and dram_rope are only used by torch_native")
    get_impl_by_name(_OP, impl)(
        query,
        query_rope,
        hbm_ckv,
        hbm_rope,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        num_cache_tokens,
        topk_src_ids,
        topk_dst_slots,
        topk_miss_counts,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        hbm_block_table,
        dram_block_table,
        dram_kv_dev_ptr,
        attention_out,
        scale_value,
        full_visible=full_visible,
        resident_refresh_tail=resident_refresh_tail,
        current=current,
        positions=positions,
        valid=valid,
        request_slots=request_slots,
        req_to_token=req_to_token,
        resident_end=resident_end,
        host_token_capacity=host_token_capacity,
        dram_rope_dev_ptr=dram_rope_dev_ptr,
        **reference_host,
    )
