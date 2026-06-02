import numpy as np
import torch
import torch_npu
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (
    split_qkv_rmsnorm_rope,
    split_qkvgate_gemma_rmsnorm_rope,
)


def custom_rope(q, k, sin, cos, half_rope_dim):
    sin = sin.to(torch.float32).cpu().numpy()
    cos = cos.to(torch.float32).cpu().numpy()
    x1 = q[..., :half_rope_dim]
    x2 = q[..., half_rope_dim:]
    cat_x = np.concatenate((-x2, x1), axis=-1)
    mul1 = cat_x * sin
    mul2 = q * cos
    res1 = mul1 + mul2

    x1 = k[..., :half_rope_dim]
    x2 = k[..., half_rope_dim:]
    cat_x = np.concatenate((-x2, x1), axis=-1)
    mul1 = cat_x * sin
    mul2 = k * cos
    res2 = mul1 + mul2
    return res1, res2


def rms_norm(
    input,
    norm_weight,
    norm_bias,
    eps,
):
    input = input.to(torch.float32).cpu().numpy()
    norm_weight = norm_weight.to(torch.float32).cpu().numpy()
    norm_bias = norm_bias.to(torch.float32).cpu().numpy()
    reciprocal_std = 1 / np.sqrt(np.mean(input**2, axis=-1, keepdims=True) + eps)
    out = input * reciprocal_std * norm_weight + norm_bias
    return out


def test_split_qkv_rmsnorm_rope():
    q_hidden_size = 6144
    kv_hidden_size = 1024
    head_dim = 128
    bsz = 12
    eps = 1e-6
    qkv = torch.randn(bsz, q_hidden_size + kv_hidden_size * 2).to(torch.bfloat16).npu()
    q_weight = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    k_weight = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    q_bias = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    k_bias = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    sin = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    cos = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    sin = torch.from_numpy(sin).to(torch.bfloat16).npu()
    cos = torch.from_numpy(cos).to(torch.bfloat16).npu()
    # fused kernel
    q, k, v = split_qkv_rmsnorm_rope(
        qkv,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        q_bias=q_bias,
        k_bias=k_bias,
    )

    # split
    _q, _k, _v = qkv.split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    # norm
    _q = rms_norm(_q.reshape(-1, head_dim), q_weight, q_bias, eps)
    _k = rms_norm(_k.reshape(-1, head_dim), k_weight, k_bias, eps)
    _q = _q.reshape(bsz, 1, -1, head_dim)
    _k = _k.reshape(bsz, 1, -1, head_dim)

    # rope
    cus_q, cus_k = custom_rope(_q, _k, sin, cos, half_rope_dim=64)
    cus_q = cus_q.reshape(bsz, -1)
    cus_k = cus_k.reshape(bsz, -1)

    assert (
        np.testing.assert_allclose(
            q.to(torch.float32).cpu().numpy(),
            cus_q,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            k.to(torch.float32).cpu().numpy(),
            cus_k,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            v.to(torch.float32).cpu().numpy(),
            _v.to(torch.float32).cpu().numpy(),
            rtol=5e-3,
        )
        is None
    )


def test_split_qkv_rope():
    q_hidden_size = 6144
    kv_hidden_size = 1024
    head_dim = 128
    bsz = 12
    eps = 1e-6
    qkv = torch.randn(bsz, q_hidden_size + kv_hidden_size * 2).to(torch.bfloat16).npu()
    sin = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    cos = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    sin = torch.from_numpy(sin).to(torch.bfloat16).npu()
    cos = torch.from_numpy(cos).to(torch.bfloat16).npu()
    # fused kernel
    q, k, v = split_qkv_rmsnorm_rope(
        qkv,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
    )

    # split
    _q, _k, _v = qkv.split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)

    # rope
    _q = _q.reshape(bsz, 1, -1, head_dim).to(torch.float32).cpu().numpy()
    _k = _k.reshape(bsz, 1, -1, head_dim).to(torch.float32).cpu().numpy()
    cus_q, cus_k = custom_rope(_q, _k, sin, cos)
    cus_q = cus_q.reshape(bsz, -1)
    cus_k = cus_k.reshape(bsz, -1)

    assert (
        np.testing.assert_allclose(
            q.to(torch.float32).cpu().numpy(),
            cus_q,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            k.to(torch.float32).cpu().numpy(),
            cus_k,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            v.to(torch.float32).cpu().numpy(),
            _v.to(torch.float32).cpu().numpy(),
            rtol=5e-3,
        )
        is None
    )


def gemma_rms_norm(
    input,
    norm_weight,
    eps,
):
    input = input.to(torch.float32).cpu().numpy()
    norm_weight = norm_weight.to(torch.float32).cpu().numpy() + 1
    reciprocal_std = 1 / np.sqrt(np.mean(input**2, axis=-1, keepdims=True) + eps)
    out = input * reciprocal_std * norm_weight
    return out


def golden_ref_split_qkvgate_gemma_rmsnorm_rope(
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
    bs = input.shape[0]
    half_rope_dim = rope_dim // 2
    pass_dim = head_dim - rope_dim

    q_gate, _k, v = input.split(
        [q_hidden_size * 2, kv_hidden_size, kv_hidden_size], dim=-1
    )
    orig_shape = torch.Size([bs])

    q_head = q_hidden_size // head_dim
    kv_head = kv_hidden_size // head_dim

    q_gate = q_gate.reshape(*orig_shape, q_head, -1)
    _q, _gate = torch.chunk(q_gate, 2, dim=-1)
    _q = _q.reshape(*orig_shape, -1)
    _gate = _gate.reshape(*orig_shape, -1)

    _q = gemma_rms_norm(_q.reshape(-1, head_dim), q_weight, eps)
    _k = gemma_rms_norm(_k.reshape(-1, head_dim), k_weight, eps)
    _q = _q.reshape(bs, 1, -1, head_dim)
    _k = _k.reshape(bs, 1, -1, head_dim)

    cus_q, cus_k = custom_rope(
        _q[..., :rope_dim], _k[..., :rope_dim], sin, cos, rope_dim // 2
    )

    rop_q = np.concatenate((cus_q, _q[..., rope_dim:]), axis=-1).reshape(
        bs, q_hidden_size
    )
    rop_k = np.concatenate((cus_k, _k[..., rope_dim:]), axis=-1).reshape(
        bs, kv_hidden_size
    )

    return rop_q, rop_k, v, _gate


def test_split_qkvgate_gemma_rmsnorm_rope():
    q_hidden_size = 512
    kv_hidden_size = 256
    head_dim = 256
    bsz = 12
    eps = 1e-6
    partial_rotary_factor = 0.25
    rope_dim = int(head_dim * partial_rotary_factor)

    qkvgate = (
        torch.randn(bsz, q_hidden_size * 2 + kv_hidden_size * 2)
        .to(torch.bfloat16)
        .npu()
    )
    q_weight = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    k_weight = (
        torch.randn(
            head_dim,
        )
        .to(torch.bfloat16)
        .npu()
    )
    sin = np.random.uniform(0, 1, [bsz, 1, 1, rope_dim])
    cos = np.random.uniform(0, 1, [bsz, 1, 1, rope_dim])
    sin = torch.from_numpy(sin).to(torch.bfloat16).npu()
    cos = torch.from_numpy(cos).to(torch.bfloat16).npu()
    # fused kernel
    q, k, v, gate = split_qkvgate_gemma_rmsnorm_rope(
        qkvgate,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
    )
    q1, k1, v1, gate1 = golden_ref_split_qkvgate_gemma_rmsnorm_rope(
        qkvgate,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
    )

    assert (
        np.testing.assert_allclose(
            q.to(torch.float32).cpu().numpy(),
            q1,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            k.to(torch.float32).cpu().numpy(),
            k1,
            atol=5e-2,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            v.to(torch.float32).cpu().numpy(),
            v1.to(torch.float32).cpu().numpy(),
            rtol=5e-3,
        )
        is None
    )

    assert (
        np.testing.assert_allclose(
            gate.to(torch.float32).cpu().numpy(),
            gate1.to(torch.float32).cpu().numpy(),
            rtol=5e-3,
        )
        is None
    )


if __name__ == "__main__":
    test_split_qkv_rmsnorm_rope()
    test_split_qkv_rope()
    test_split_qkvgate_gemma_rmsnorm_rope()
