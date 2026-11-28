"""Public Ascend950 HiSparse operator."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP = "attention.fused_quant_scatter_copy_sparse_flash_attention_v2"
_SUPPORTED_GENERATIONS = (Generation.ARCH35,)


@register_impl(_OP, "auto", supported_generations=_SUPPORTED_GENERATIONS)
@register_impl(_OP, "ascendc", supported_generations=_SUPPORTED_GENERATIONS)
def _ascendc_impl(*args, **kwargs) -> None:
    torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2.default(
        *args, **kwargs
    )


@register_impl(_OP, "torch_native", supported_generations=_SUPPORTED_GENERATIONS)
def _torch_native_impl(*args, **kwargs) -> None:
    from ...implementation.torch_native.attention.fused_quant_scatter_copy_sparse_flash_attention_v2 import (
        fused_quant_scatter_copy_sparse_flash_attention_v2 as reference,
    )

    reference(*args, **kwargs)


@register_operator(_OP)
def fused_quant_scatter_copy_sparse_flash_attention_v2(
    query_nope,
    query_rope,
    hbm_kv_bytes,
    dram_kv_bytes,
    dram_kv_dev_ptr,
    hbm_block_table,
    dram_block_table,
    topk_dst_slots,
    topk_src_ids,
    topk_miss_counts,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    num_cache_tokens,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    scale_value,
    attention_out,
    *,
    current=None,
    positions=None,
    current_valid=None,
    request_slots=None,
    req_to_token=None,
    resident_end=None,
    native_lim_plan=False,
    impl="auto",
) -> None:
    """Update packed C8 resident KV and run sparse attention in place.

    Both auto and ascendc select the native 3510 operator. The torch_native
    implementation evaluates the copy and attention plan on CPU using dram_kv_bytes.
    See the operator README for the complete tensor contract.
    """
    get_impl_by_name(_OP, impl)(
        query_nope,
        query_rope,
        hbm_kv_bytes,
        dram_kv_bytes,
        dram_kv_dev_ptr,
        hbm_block_table,
        dram_block_table,
        topk_dst_slots,
        topk_src_ids,
        topk_miss_counts,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        num_cache_tokens,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        scale_value,
        attention_out,
        current=current,
        positions=positions,
        current_valid=current_valid,
        request_slots=request_slots,
        req_to_token=req_to_token,
        resident_end=resident_end,
        native_lim_plan=native_lim_plan,
    )
