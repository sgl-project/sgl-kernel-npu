"""Public split qkv rmsnorm mrope."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("norm.split_qkv_rmsnorm_mrope", "auto")
@register_impl("norm.split_qkv_rmsnorm_mrope", "triton")
def _triton_impl(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    num_q_heads,
    num_kv_heads,
    head_size,
    eps,
    mrope_section,
    is_interleaved,
    rope_dim=None,
    q_bias=None,
    k_bias=None,
    has_gate=False,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.split_qkv_rmsnorm_mrope import (
            split_qkv_rmsnorm_mrope,
        )

        _triton_fn = split_qkv_rmsnorm_mrope
    return _triton_fn(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
        q_bias,
        k_bias,
        has_gate,
    )


@register_impl("norm.split_qkv_rmsnorm_mrope", "torch_native")
def _torch_native_impl(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    num_q_heads,
    num_kv_heads,
    head_size,
    eps,
    mrope_section,
    is_interleaved,
    rope_dim=None,
    q_bias=None,
    k_bias=None,
    has_gate=False,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.split_qkv_rmsnorm_mrope import (
            split_qkv_rmsnorm_mrope,
        )

        _torch_native_fn = split_qkv_rmsnorm_mrope
    return _torch_native_fn(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
        q_bias,
        k_bias,
        has_gate,
    )


@register_operator("norm.split_qkv_rmsnorm_mrope")
def split_qkv_rmsnorm_mrope(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    num_q_heads,
    num_kv_heads,
    head_size,
    eps,
    mrope_section,
    is_interleaved,
    rope_dim=None,
    q_bias=None,
    k_bias=None,
    has_gate=False,
    *,
    impl="auto"
):
    """Splits packed Q/K/V and optional per-head gate channels, normalizes Q and K with RMSNorm weights and optional biases, then applies temporal/height/width rotary embeddings. Full and partial rotary dimensions and interleaved or contiguous MRoPE sections are supported. V and gate are copied without normalization.

    Args:
        qkv (Tensor): BF16/FP16 contiguous [tokens,QH*H*(2 if has_gate else 1)+2*KVH*H]. Default: Required.
        q_weight (Tensor): Floating [head_size] query RMSNorm weight. Default: Required.
        k_weight (Tensor): Floating [head_size] key RMSNorm weight. Default: Required.
        cos_sin (Tensor): Floating [3,tokens,rope_dim]; cosine then sine halves per axis. Default: Required.
        num_q_heads (int): Local query head count after tensor parallel partitioning. Default: Required.
        num_kv_heads (int): Local KV head count. Default: Required.
        head_size (int): Channels per Q/K/V head. Default: Required.
        eps (float): RMSNorm stabilization constant. Default: Required.
        mrope_section (list[int]): Three temporal/height/width rotary sections. Default: Required.
        is_interleaved (bool): Interleave rotary axes instead of using contiguous sections. Default: Required.
        rope_dim (int or None): Even rotated width <=head_size; None rotates the complete head. Default: None.
        q_bias (Tensor or None): Optional floating [head_size] query bias. Default: None.
        k_bias (Tensor or None): Key bias paired with q_bias, same shape. Default: None.
        has_gate (bool): Input contains per-query-head gate channels. Default: False.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        Q [T,QH*H], K/V [T,KVH*H], gate [T,QH*H] or [T,0]; all match qkv dtype.
    """
    return get_impl_by_name("norm.split_qkv_rmsnorm_mrope", impl)(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        num_q_heads,
        num_kv_heads,
        head_size,
        eps,
        mrope_section,
        is_interleaved,
        rope_dim,
        q_bias,
        k_bias,
        has_gate,
    )
