"""Released CANN calls behind the registered DSV4 APIs."""

import torch

from ._ops import get_op


def sparse_flash_mla_metadata(**kwargs):
    return get_op("sparse_flash_mla_metadata")(**kwargs)


def sparse_flash_mla(q, **kwargs):
    return get_op("sparse_flash_mla")(q, **kwargs)


def mixed_quant_sparse_flash_mla_metadata(**kwargs):
    return get_op("mixed_quant_sparse_flash_mla_metadata")(**kwargs)


def mixed_quant_sparse_flash_mla(q, **kwargs):
    return get_op("mixed_quant_sparse_flash_mla")(q, **kwargs)


def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    return get_op("quant_lightning_indexer_metadata")(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    return get_op("quant_lightning_indexer")(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


def indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    get_op("indexer_quant_cache")(cache, scale_cache, x, slot_mapping, **kwargs)
    return cache, scale_cache


def inplace_partial_rotary_mul(x, cos, sin, **kwargs):
    get_op("inplace_partial_rotary_mul")(x, cos, sin, **kwargs)
    return x


def mhc_pre(x, fn, scale, base, hc_mult, sinkhorn_iters, hc_eps, rms_norm_eps):
    shape = x.shape
    h_in, h_post, h_res, *_ = get_op("mhc_pre_sinkhorn")(
        x.reshape(-1, 1, hc_mult, shape[-1]),
        fn,
        scale,
        base,
        hc_mult,
        sinkhorn_iters,
        hc_eps,
        rms_norm_eps,
        False,
    )
    return (
        h_in.reshape(-1, shape[-1]).to(x.dtype),
        h_post.reshape(-1, hc_mult),
        h_res.reshape(-1, hc_mult, hc_mult),
    )


def mhc_post(x, residual, post, comb):
    return get_op("mhc_post")(residual, comb, x, post)


def kv_compress_epilog(cache, x, slot_mapping, **kwargs):
    get_op("kv_compress_epilog")(cache.view(torch.uint8), x, slot_mapping, **kwargs)
    return cache


def compressor(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs):
    # Both A3 and A5 bind the released/native compressor through the npu namespace.
    # Do not redirect this to the similarly named experimental custom op.
    return torch.ops.npu.compressor(
        x, wkv, wgate, state_cache, ape, norm_weight, **kwargs
    )
