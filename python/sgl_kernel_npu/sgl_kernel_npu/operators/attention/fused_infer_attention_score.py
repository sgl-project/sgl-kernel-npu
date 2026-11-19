"""TorchNPU fused inference attention and its graph-capture helpers."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_npu = None


def _get_torch_npu():
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


@register_impl("attention.fused_infer_attention_score", "auto")
@register_impl("attention.fused_infer_attention_score", "torch_npu")
def _fia_impl(query, key, value, **kwargs):
    return _get_torch_npu().npu_fused_infer_attention_score(query, key, value, **kwargs)


@register_operator("attention.fused_infer_attention_score")
def fused_infer_attention_score(query, key, value, *, impl="auto", **kwargs):
    """Return attention and the SDK's softmax LSE tensor.

    Query, key and value shapes follow input_layout, e.g. [B, N, S, D] for
    BNSD. Optional TorchNPU keywords include block_table, sequence lengths,
    RoPE, mask and quantization settings. auto selects torch_npu.
    """
    return get_impl_by_name("attention.fused_infer_attention_score", impl)(
        query, key, value, **kwargs
    )


@register_impl("attention.fused_infer_attention_score_out", "auto")
@register_impl("attention.fused_infer_attention_score_out", "torch_npu")
def _fia_out_impl(query, key, value, **kwargs):
    return _get_torch_npu().npu_fused_infer_attention_score.out(
        query, key, value, **kwargs
    )


@register_operator("attention.fused_infer_attention_score_out")
def fused_infer_attention_score_out(query, key, value, *, out, impl="auto", **kwargs):
    """Run FIA into out=[attention_out, softmax_lse] using the native .out overload.

    The other keywords match FIA and may include a preallocated workspace.
    auto selects torch_npu.
    """
    return get_impl_by_name("attention.fused_infer_attention_score_out", impl)(
        query, key, value, out=out, **kwargs
    )


@register_impl("attention.fused_infer_attention_score_get_max_workspace", "auto")
@register_impl("attention.fused_infer_attention_score_get_max_workspace", "torch_npu")
def _fia_workspace_impl(query, key, value, **kwargs):
    return _get_torch_npu()._npu_fused_infer_attention_score_get_max_workspace(
        query, key, value, **kwargs
    )


@register_operator("attention.fused_infer_attention_score_get_max_workspace")
def fused_infer_attention_score_get_max_workspace(
    query, key, value, *, impl="auto", **kwargs
):
    """Return the TorchNPU workspace for the matching FIA arguments."""
    return get_impl_by_name(
        "attention.fused_infer_attention_score_get_max_workspace", impl
    )(query, key, value, **kwargs)


@register_impl("attention.fused_infer_attention_score_v2", "auto")
@register_impl("attention.fused_infer_attention_score_v2", "torch_npu")
def _fia_v2_impl(query, key, value, **kwargs):
    return _get_torch_npu().npu_fused_infer_attention_score_v2(
        query, key, value, **kwargs
    )


@register_operator("attention.fused_infer_attention_score_v2")
def fused_infer_attention_score_v2(query, key, value, *, impl="auto", **kwargs):
    """Return attention and the SDK's softmax LSE tensor using the V2 schema.

    Query, key and value shapes follow input_layout, e.g. [B, N, S, D] for
    BNSD. Optional SDK keywords include actual_seq_qlen, actual_seq_kvlen,
    softmax_scale, num_query_heads and learnable_sink. auto selects torch_npu.
    """
    return get_impl_by_name("attention.fused_infer_attention_score_v2", impl)(
        query, key, value, **kwargs
    )


@register_impl("attention.fused_infer_attention_score_v2_out", "auto")
@register_impl("attention.fused_infer_attention_score_v2_out", "torch_npu")
def _fia_v2_out_impl(query, key, value, **kwargs):
    return _get_torch_npu().npu_fused_infer_attention_score_v2.out(
        query, key, value, **kwargs
    )


@register_operator("attention.fused_infer_attention_score_v2_out")
def fused_infer_attention_score_v2_out(
    query, key, value, *, out, impl="auto", **kwargs
):
    """Run FIA V2 into out=[attention_out, softmax_lse] via its native .out overload."""
    return get_impl_by_name("attention.fused_infer_attention_score_v2_out", impl)(
        query, key, value, out=out, **kwargs
    )


@register_impl("attention.fused_infer_attention_score_v2_get_max_workspace", "auto")
@register_impl(
    "attention.fused_infer_attention_score_v2_get_max_workspace", "torch_npu"
)
def _fia_v2_workspace_impl(query, key, value, **kwargs):
    return _get_torch_npu()._npu_fused_infer_attention_score_v2_get_max_workspace(
        query, key, value, **kwargs
    )


@register_operator("attention.fused_infer_attention_score_v2_get_max_workspace")
def fused_infer_attention_score_v2_get_max_workspace(
    query, key, value, *, impl="auto", **kwargs
):
    """Return the TorchNPU workspace for the matching FIA V2 arguments."""
    return get_impl_by_name(
        "attention.fused_infer_attention_score_v2_get_max_workspace", impl
    )(query, key, value, **kwargs)
