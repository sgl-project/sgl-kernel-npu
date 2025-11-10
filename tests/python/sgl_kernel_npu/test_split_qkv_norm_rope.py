import numpy as np
import torch
import torch_npu
from sgl_kernel_npu.norm.split_qkv_norm_rope import split_qkv_norm_rope


def custom_rope(q, k, sin, cos):
    x1 = q[..., :64]
    x2 = q[..., 64:]
    cat_x = torch.concat((-x2, x1), dim=-1)
    mul1 = cat_x * sin
    mul2 = q * cos
    res1 = mul1 + mul2

    x1 = k[..., :64]
    x2 = k[..., 64:]
    cat_x = torch.concat((-x2, x1), dim=-1)
    mul1 = cat_x * sin
    mul2 = k * cos
    res2 = mul1 + mul2
    return res1, res2


def test_split_qkv_norm_rope():
    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    q_hidden_size = 6144
    kv_hidden_size = 1024
    head_dim = 128
    bsz = 12
    eps = 1e-6
    qkv = torch.randn(bsz, q_hidden_size + kv_hidden_size * 2).to(torch.bfloat16).npu()
    q_weight = torch.randn(head_dim, ).to(torch.bfloat16).npu()
    k_weight = torch.randn(head_dim, ).to(torch.bfloat16).npu()
    q_bias = torch.randn(head_dim, ).to(torch.bfloat16).npu()
    k_bias = torch.randn(head_dim, ).to(torch.bfloat16).npu()
    sin = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    cos = np.random.uniform(0, 1, [bsz, 1, 1, head_dim])
    sin = torch.from_numpy(sin).to(torch.bfloat16).npu()
    cos = torch.from_numpy(cos).to(torch.bfloat16).npu()
    # fused kernel
    q, k, v = split_qkv_norm_rope(
        qkv,
        sin,
        cos,
        q_weight,
        k_weight,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps,
        q_bias,
        k_bias
    )

    # split
    _q, _k, _v = qkv.split([q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    # norm
    _q = torch_npu.npu_rms_norm(_q.reshape(-1, head_dim), q_weight, eps)[0] + q_bias
    _k = torch_npu.npu_rms_norm(_k.reshape(-1, head_dim), k_weight, eps)[0] + k_bias
    _q = _q.view(bsz, -1)
    _k = _k.view(bsz, -1)
    # print(k - _k)
    # rope
    _q = _q.contiguous().view(_q.shape[0], 1, -1, head_dim)
    _k = _k.contiguous().view(_k.shape[0], 1, -1, head_dim)
    cus_q, cus_k = custom_rope(_q, _k, sin, cos)
    torch_npu.npu_apply_rotary_pos_emb(
        _q, _k, cos, sin,
    )

    _q = _q.view(bsz, -1)
    _k = _k.view(bsz, -1)
    cus_q = cus_q.view(bsz, -1)
    cus_k = cus_k.view(bsz, -1)

    assert torch.allclose(q, cus_q, rtol=5e-3)

    assert torch.allclose(k, cus_k, rtol=5e-3)

    assert torch.allclose(v, _v, rtol=5e-3)


if __name__ == "__main__":
    test_split_qkv_norm_rope()