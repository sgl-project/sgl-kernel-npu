import numpy as np
import torch
import torch_npu
from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope


def custom_rope(q, k, sin, cos):
    sin = sin.to(torch.float32).cpu().numpy()
    cos = cos.to(torch.float32).cpu().numpy()
    x1 = q[..., :64]
    x2 = q[..., 64:]
    cat_x = np.concatenate((-x2, x1), axis=-1)
    mul1 = cat_x * sin
    mul2 = q * cos
    res1 = mul1 + mul2

    x1 = k[..., :64]
    x2 = k[..., 64:]
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


if __name__ == "__main__":
    test_split_qkv_rmsnorm_rope()
    test_split_qkv_rope()
