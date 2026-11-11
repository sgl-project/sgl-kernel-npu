"""Public RainFusionAttention dispatch for the CANN built-in ACLNN operator."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl(
    "attention.rain_fusion_attention", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.rain_fusion_attention",
    "cann_ops_transformer",
    supported_generations={Generation.ARCH22},
)
def _cann_ops_transformer_impl(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    blockshape,
    attn_mask,
    actual_seq_qlen,
    actual_seq_kvlen,
    block_table,
    q_input_layout,
    kv_input_layout,
    head_num,
    mask_type,
    scale,
    inner_precise,
    block_size,
):
    import torch

    return torch.ops.npu.rainfusionattention(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        blockshape,
        attn_mask,
        actual_seq_qlen,
        actual_seq_kvlen,
        block_table,
        q_input_layout,
        kv_input_layout,
        head_num,
        mask_type,
        scale,
        inner_precise,
        block_size,
    )


@register_impl(
    "attention.rain_fusion_attention",
    "torch_native",
    supported_generations={Generation.ARCH22},
)
def _torch_native_impl(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    blockshape,
    attn_mask,
    actual_seq_qlen,
    actual_seq_kvlen,
    block_table,
    q_input_layout,
    kv_input_layout,
    head_num,
    mask_type,
    scale,
    inner_precise,
    block_size,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.rain_fusion_attention import (
            rain_fusion_attention,
        )

        _torch_native_fn = rain_fusion_attention
    return _torch_native_fn(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        blockshape,
        attn_mask=attn_mask,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        block_table=block_table,
        q_input_layout=q_input_layout,
        kv_input_layout=kv_input_layout,
        head_num=head_num,
        mask_type=mask_type,
        scale=scale,
        inner_precise=inner_precise,
        block_size=block_size,
    )


@register_operator("attention.rain_fusion_attention")
def rain_fusion_attention(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    blockshape,
    attn_mask=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    block_table=None,
    q_input_layout="TND",
    kv_input_layout="TND",
    head_num=1,
    mask_type=0,
    scale=1.0,
    inner_precise=1,
    block_size=0,
    impl="auto",
):
    """Return (attention_out, softmax_lse) from block-sparse fused attention.

    auto selects cann_ops_transformer on arch22 without fallback. select_idx is
    int64 [q_blocks, head_num, kv_blocks] with valid KV block ids first in
    ascending order and -1 padding; select_num_idx is int64
    [q_blocks, head_num]. q_input_layout and kv_input_layout accept TND or BNSD
    only.

    attention_out keeps the query shape/dtype/NPU format. softmax_lse uses the
    first three query dimensions and the query dtype/NPU format; the CANN
    operator may leave its content untouched.
    """
    return get_impl_by_name("attention.rain_fusion_attention", impl)(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        blockshape,
        attn_mask,
        actual_seq_qlen,
        actual_seq_kvlen,
        block_table,
        q_input_layout,
        kv_input_layout,
        head_num,
        mask_type,
        scale,
        inner_precise,
        block_size,
    )


# Legacy name; the call contract is unchanged.
rainfusionattention = rain_fusion_attention
