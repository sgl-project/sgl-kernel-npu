from ._qkv_compute import flatten_heads, normalize, rotate


def split_qkvgate_gemma_rmsnorm_rope(
    q, k, v, gate, sin, cos, eps, q_weight, k_weight, dtype
):
    q = rotate(normalize(q, eps, q_weight, gemma=True), sin, cos)
    k = rotate(normalize(k, eps, k_weight, gemma=True), sin, cos)
    return (
        flatten_heads(q, dtype),
        flatten_heads(k, dtype),
        v.clone().contiguous(),
        flatten_heads(gate.clone(), dtype),
    )
