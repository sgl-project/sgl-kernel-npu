from ._qkv_compute import flatten_heads, normalize, rotate


def split_qkv_rmsnorm_rope(
    q, k, v, sin, cos, eps, q_weight, k_weight, q_bias, k_bias, is_neox_style, dtype
):
    q = rotate(normalize(q, eps, q_weight, q_bias), sin, cos, neox=is_neox_style)
    k = rotate(normalize(k, eps, k_weight, k_bias), sin, cos, neox=is_neox_style)
    return flatten_heads(q, dtype), flatten_heads(k, dtype), v.clone().contiguous()
