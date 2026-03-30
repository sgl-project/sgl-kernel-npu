import numpy as np
import torch
import torch_npu
from sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope import split_qkv_tp_rmsnorm_rope


def custom_rope(q, k, sin, cos):
    sin = sin.to(torch.float32).cpu().numpy()
    cos = cos.to(torch.float32).cpu().numpy()
    half = sin.shape[-1] // 2
    sin_half = sin[..., :half]
    cos_half = cos[..., :half]

    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)

    x1 = q[..., :half]
    x2 = q[..., half : 2 * half]
    qn1 = x1 * cos_half - x2 * sin_half
    qn2 = x2 * cos_half + x1 * sin_half
    res1 = np.concatenate((qn1, qn2), axis=-1)

    x1 = k[..., :half]
    x2 = k[..., half : 2 * half]
    kn1 = x1 * cos_half - x2 * sin_half
    kn2 = x2 * cos_half + x1 * sin_half
    res2 = np.concatenate((kn1, kn2), axis=-1)

    res1 = torch.from_numpy(res1).to(torch.bfloat16).to(torch.float32).numpy()
    res2 = torch.from_numpy(res2).to(torch.bfloat16).to(torch.float32).numpy()
    return res1, res2


def rms_norm_tp(input, norm_weight, eps, tp_world=1):
    input_f32 = input.to(torch.float32)
    norm_weight_f32 = norm_weight.to(torch.float32)
    variance = torch.mean(input_f32**2, dim=-1, keepdim=True)
    global_variance = variance * (1.0 / tp_world)
    reciprocal_std = 1.0 / torch.sqrt(global_variance + eps)
    out = input_f32 * reciprocal_std * norm_weight_f32
    out_bf16 = out.to(torch.bfloat16).to(torch.float32)
    return out_bf16.cpu().numpy()


def test_split_qkv_tp_rmsnorm_rope():
    q_hidden_size = 6144
    kv_hidden_size = 1024
    head_dim = 128
    bsz = 12
    eps = 1e-6
    tp_world = 1
    qkv = torch.randn(bsz, q_hidden_size + kv_hidden_size * 2).to(torch.bfloat16).npu()
    q_weight = torch.randn(q_hidden_size).to(torch.bfloat16).npu()
    k_weight = torch.randn(kv_hidden_size).to(torch.bfloat16).npu()
    sin = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    cos = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    sin = torch.from_numpy(sin).to(torch.bfloat16).npu()
    cos = torch.from_numpy(cos).to(torch.bfloat16).npu()

    q, k, v = split_qkv_tp_rmsnorm_rope(
        qkv,
        cos.squeeze(1).squeeze(1),
        sin.squeeze(1).squeeze(1),
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_weight,
        k_weight,
        head_dim,
        tp_world,
        None,
    )

    _q, _k, _v = qkv.split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    _q = rms_norm_tp(_q, q_weight, eps, tp_world)
    _k = rms_norm_tp(_k, k_weight, eps, tp_world)
    _q = _q.reshape(bsz, 1, -1, head_dim)
    _k = _k.reshape(bsz, 1, -1, head_dim)

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
    test_split_qkv_tp_rmsnorm_rope()
    print("test_split_qkv_tp_rmsnorm_rope passed")
