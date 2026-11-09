"""PyTorch QKV normalization/RoPE and per-head Q/gate deinterleaving."""

from .norm_utils import rms_norm, rotary, validate_split_qkv, validate_split_qkvgate


def split_qkv_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps=None,
    q_weight=None,
    k_weight=None,
    q_bias=None,
    k_bias=None,
    is_neox_style=True,
):
    validate_split_qkv(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_bias,
        k_bias,
        is_neox_style,
    )
    tokens = input.shape[0]
    rope_dim = sin.shape[-1]
    q, k, v = input.split((q_hidden_size, kv_hidden_size, kv_hidden_size), dim=-1)
    outputs = []
    for values, width, weight, bias in (
        (q, q_hidden_size, q_weight, q_bias),
        (k, kv_hidden_size, k_weight, k_bias),
    ):
        values = values.float().reshape(tokens, width // head_dim, head_dim)
        if eps is not None:
            values = rms_norm(values, weight, eps, bias)
        result = rotary(values, sin, cos, rope_dim, is_neox_style)
        outputs.append(result.to(input.dtype).reshape(tokens, width))
    return outputs[0], outputs[1], v.clone().contiguous()


def split_qkvgate_gemma_rmsnorm_rope(
    input,
    sin,
    cos,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    rope_dim,
    eps,
    q_weight,
    k_weight,
):
    validate_split_qkvgate(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        q_weight,
        k_weight,
    )
    tokens = input.shape[0]
    q_gate, k, v = input.split(
        (q_hidden_size * 2, kv_hidden_size, kv_hidden_size), dim=-1
    )
    q, gate = q_gate.reshape(tokens, q_hidden_size // head_dim, 2, head_dim).unbind(2)
    k = k.reshape(tokens, kv_hidden_size // head_dim, head_dim)
    q = rms_norm(q, q_weight, eps, gemma=True)
    k = rms_norm(k, k_weight, eps, gemma=True)
    q = rotary(q, sin, cos, rope_dim).to(input.dtype).reshape(tokens, q_hidden_size)
    k = rotary(k, sin, cos, rope_dim).to(input.dtype).reshape(tokens, kv_hidden_size)
    return (
        q,
        k,
        v.clone().contiguous(),
        gate.reshape(tokens, q_hidden_size).clone().contiguous(),
    )
