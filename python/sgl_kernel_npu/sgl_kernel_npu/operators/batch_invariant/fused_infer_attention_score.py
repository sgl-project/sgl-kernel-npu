"""Public batch-invariant fused infer attention operators.

The main operator returns the external OpOverloadPacket so callers keep its
schema and .out overload for NPU graph capture; the helpers expose the
workspace and output buffers needed to capture that call.
"""

import torch
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_batch_invariant_ops = None


def _get_batch_invariant_ops():
    global _batch_invariant_ops
    if _batch_invariant_ops is None:
        import batch_invariant_ops  # noqa: F401

        _batch_invariant_ops = torch.ops.batch_invariant_ops
    return _batch_invariant_ops


# The external extension is imported lazily on the first call.
@register_impl("batch_invariant.fused_infer_attention_score", "auto")
@register_impl("batch_invariant.fused_infer_attention_score", "batch_invariant_ops")
def _batch_invariant_ops_impl():
    return _get_batch_invariant_ops().npu_fused_infer_attention_score_batch_invariant


@register_operator("batch_invariant.fused_infer_attention_score")
def fused_infer_attention_score_batch_invariant(*, impl="auto"):
    """Return the external FIA OpOverloadPacket instead of calling it.

    Keeping the packet (not a forwarding function) preserves its schema and
    .out overload for NPU graph capture.
    """
    return get_impl_by_name("batch_invariant.fused_infer_attention_score", impl)()


@register_impl("batch_invariant.fused_infer_attention_score_get_max_workspace", "auto")
@register_impl(
    "batch_invariant.fused_infer_attention_score_get_max_workspace",
    "batch_invariant_ops",
)
def _get_max_workspace_batch_invariant_ops_impl(
    query,
    key,
    value,
    *,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
    softmax_lse_flag=False,
):
    ops = _get_batch_invariant_ops()
    return ops._npu_fused_infer_attention_score_batch_invariant_get_max_workspace(
        query,
        key,
        value,
        pse_shift=pse_shift,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        dequant_scale1=dequant_scale1,
        quant_scale1=quant_scale1,
        dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2,
        quant_offset2=quant_offset2,
        antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset,
        block_table=block_table,
        query_padding_size=query_padding_size,
        kv_padding_size=kv_padding_size,
        key_antiquant_scale=key_antiquant_scale,
        key_antiquant_offset=key_antiquant_offset,
        value_antiquant_scale=value_antiquant_scale,
        value_antiquant_offset=value_antiquant_offset,
        key_shared_prefix=key_shared_prefix,
        value_shared_prefix=value_shared_prefix,
        actual_shared_prefix_len=actual_shared_prefix_len,
        query_rope=query_rope,
        key_rope=key_rope,
        key_rope_antiquant_scale=key_rope_antiquant_scale,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        antiquant_mode=antiquant_mode,
        key_antiquant_mode=key_antiquant_mode,
        value_antiquant_mode=value_antiquant_mode,
        softmax_lse_flag=softmax_lse_flag,
    )


@register_operator("batch_invariant.fused_infer_attention_score_get_max_workspace")
def fused_infer_attention_score_batch_invariant_get_max_workspace(
    query,
    key,
    value,
    *,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
    softmax_lse_flag=False,
    impl="auto",
):
    """Return the workspace size needed for the same FIA arguments."""
    impl_fn = get_impl_by_name(
        "batch_invariant.fused_infer_attention_score_get_max_workspace", impl
    )
    return impl_fn(
        query,
        key,
        value,
        pse_shift=pse_shift,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        dequant_scale1=dequant_scale1,
        quant_scale1=quant_scale1,
        dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2,
        quant_offset2=quant_offset2,
        antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset,
        block_table=block_table,
        query_padding_size=query_padding_size,
        kv_padding_size=kv_padding_size,
        key_antiquant_scale=key_antiquant_scale,
        key_antiquant_offset=key_antiquant_offset,
        value_antiquant_scale=value_antiquant_scale,
        value_antiquant_offset=value_antiquant_offset,
        key_shared_prefix=key_shared_prefix,
        value_shared_prefix=value_shared_prefix,
        actual_shared_prefix_len=actual_shared_prefix_len,
        query_rope=query_rope,
        key_rope=key_rope,
        key_rope_antiquant_scale=key_rope_antiquant_scale,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        antiquant_mode=antiquant_mode,
        key_antiquant_mode=key_antiquant_mode,
        value_antiquant_mode=value_antiquant_mode,
        softmax_lse_flag=softmax_lse_flag,
    )


@register_impl("batch_invariant.fused_infer_attention_score_infer_output", "auto")
@register_impl(
    "batch_invariant.fused_infer_attention_score_infer_output",
    "batch_invariant_ops",
)
def _infer_output_batch_invariant_ops_impl(
    query,
    value,
    *,
    input_layout="BSH",
    quant_scale2=None,
    block_table=None,
    num_heads=1,
    num_key_value_heads=0,
    softmax_lse_flag=False,
    query_rope=None,
):
    ops = _get_batch_invariant_ops()
    return ops._npu_fused_infer_attention_score_batch_invariant_infer_output(
        query,
        value,
        input_layout=input_layout,
        quant_scale2=quant_scale2,
        block_table=block_table,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        softmax_lse_flag=softmax_lse_flag,
        query_rope=query_rope,
    )


@register_operator("batch_invariant.fused_infer_attention_score_infer_output")
def fused_infer_attention_score_batch_invariant_infer_output(
    query,
    value,
    *,
    input_layout="BSH",
    quant_scale2=None,
    block_table=None,
    num_heads=1,
    num_key_value_heads=0,
    softmax_lse_flag=False,
    query_rope=None,
    impl="auto",
):
    """Allocate the attention output and log-sum-exp buffers for .out."""
    impl_fn = get_impl_by_name(
        "batch_invariant.fused_infer_attention_score_infer_output", impl
    )
    return impl_fn(
        query,
        value,
        input_layout=input_layout,
        quant_scale2=quant_scale2,
        block_table=block_table,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        softmax_lse_flag=softmax_lse_flag,
        query_rope=query_rope,
    )


@register_impl("batch_invariant.fused_infer_attention_score_out", "auto")
@register_impl("batch_invariant.fused_infer_attention_score_out", "batch_invariant_ops")
def _out_batch_invariant_ops_impl(
    query,
    key,
    value,
    *,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
    softmax_lse_flag=False,
    workspace=None,
    attention_out=None,
    softmax_lse=None,
):
    ops = _get_batch_invariant_ops()
    return ops.npu_fused_infer_attention_score_batch_invariant_out(
        query,
        key,
        value,
        pse_shift=pse_shift,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        dequant_scale1=dequant_scale1,
        quant_scale1=quant_scale1,
        dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2,
        quant_offset2=quant_offset2,
        antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset,
        block_table=block_table,
        query_padding_size=query_padding_size,
        kv_padding_size=kv_padding_size,
        key_antiquant_scale=key_antiquant_scale,
        key_antiquant_offset=key_antiquant_offset,
        value_antiquant_scale=value_antiquant_scale,
        value_antiquant_offset=value_antiquant_offset,
        key_shared_prefix=key_shared_prefix,
        value_shared_prefix=value_shared_prefix,
        actual_shared_prefix_len=actual_shared_prefix_len,
        query_rope=query_rope,
        key_rope=key_rope,
        key_rope_antiquant_scale=key_rope_antiquant_scale,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        antiquant_mode=antiquant_mode,
        key_antiquant_mode=key_antiquant_mode,
        value_antiquant_mode=value_antiquant_mode,
        softmax_lse_flag=softmax_lse_flag,
        workspace=workspace,
        attention_out=attention_out,
        softmax_lse=softmax_lse,
    )


@register_operator("batch_invariant.fused_infer_attention_score_out")
def fused_infer_attention_score_batch_invariant_out(
    query,
    key,
    value,
    *,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    query_padding_size=None,
    kv_padding_size=None,
    key_antiquant_scale=None,
    key_antiquant_offset=None,
    value_antiquant_scale=None,
    value_antiquant_offset=None,
    key_shared_prefix=None,
    value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None,
    key_rope=None,
    key_rope_antiquant_scale=None,
    num_heads=1,
    scale=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=0,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    antiquant_mode=0,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
    softmax_lse_flag=False,
    workspace=None,
    attention_out=None,
    softmax_lse=None,
    impl="auto",
):
    """Run FIA with the same semantics, writing into caller-provided buffers."""
    impl_fn = get_impl_by_name("batch_invariant.fused_infer_attention_score_out", impl)
    return impl_fn(
        query,
        key,
        value,
        pse_shift=pse_shift,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        dequant_scale1=dequant_scale1,
        quant_scale1=quant_scale1,
        dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2,
        quant_offset2=quant_offset2,
        antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset,
        block_table=block_table,
        query_padding_size=query_padding_size,
        kv_padding_size=kv_padding_size,
        key_antiquant_scale=key_antiquant_scale,
        key_antiquant_offset=key_antiquant_offset,
        value_antiquant_scale=value_antiquant_scale,
        value_antiquant_offset=value_antiquant_offset,
        key_shared_prefix=key_shared_prefix,
        value_shared_prefix=value_shared_prefix,
        actual_shared_prefix_len=actual_shared_prefix_len,
        query_rope=query_rope,
        key_rope=key_rope,
        key_rope_antiquant_scale=key_rope_antiquant_scale,
        num_heads=num_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        input_layout=input_layout,
        num_key_value_heads=num_key_value_heads,
        sparse_mode=sparse_mode,
        inner_precise=inner_precise,
        block_size=block_size,
        antiquant_mode=antiquant_mode,
        key_antiquant_mode=key_antiquant_mode,
        value_antiquant_mode=value_antiquant_mode,
        softmax_lse_flag=softmax_lse_flag,
        workspace=workspace,
        attention_out=attention_out,
        softmax_lse=softmax_lse,
    )
