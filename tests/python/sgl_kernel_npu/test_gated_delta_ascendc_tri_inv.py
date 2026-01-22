import os

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_native, fast_inv_tril

device = "npu"


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-4):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def print_diff(name, ref, tri, atol=0.005):
    abs_diff = torch.abs(ref - tri)
    max_abs_diff = abs_diff.max().item()
    print(f"[{name}] Max absolute difference: {max_abs_diff:.6f}")
    if max_abs_diff > atol:
        print(f"Exceeds tolerance ({atol})!")


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (8, 128, 0, [0, 6], torch.float16),
            (8, 128, 0, [0, 31], torch.float16),
            (8, 128, 0, [0, 64], torch.float16),
            (8, 128, 0, [0, 100], torch.float16),
            (8, 128, 0, [0, 127], torch.float16),
            (8, 128, 0, [0, 3584, 7168], torch.float16),
            (8, 128, 0.5, [0, 3584, 7168], torch.float16),
            (8, 128, 0, [0, 256, 500, 1000], torch.float16),
            (8, 128, 0.5, [0, 256, 500, 1000], torch.float16),
            (8, 128, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
            (8, 128, 0, [0, 64, 100, 300, 1200, 2000], torch.float16),
            (8, 128, 0, [0, 64, 300, 1200, 2000], torch.float16),
            (8, 128, 0, [0, 100, 300, 1200, 2000], torch.float16),
            (8, 128, 0, [0, 128, 300, 1200, 2000], torch.float16),
            (8, 128, 0, [0, 256, 300, 1200, 2000], torch.float16),
            (4, 128, 0, [0, 6], torch.float16),
            (4, 128, 0, [0, 31], torch.float16),
            (4, 128, 0, [0, 64], torch.float16),
            (4, 128, 0, [0, 100], torch.float16),
            (4, 128, 0, [0, 127], torch.float16),
            (4, 128, 0, [0, 3584, 7168], torch.float16),
            (4, 128, 0.5, [0, 3584, 7168], torch.float16),
            (4, 128, 0, [0, 256, 500, 1000], torch.float16),
            (4, 128, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 128, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
            (4, 128, 0, [0, 64, 100, 300, 1200, 2000], torch.float16),
            (4, 128, 0, [0, 64, 300, 1200, 2000], torch.float16),
            (4, 128, 0, [0, 100, 300, 1200, 2000], torch.float16),
            (4, 128, 0, [0, 128, 300, 1200, 2000], torch.float16),
            (4, 128, 0, [0, 256, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv("SKIP_TEST_CHUNK_VARLEN") == "1",
    reason="Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set",
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if D != 128:
        pytest.skip(
            reason="chunk_gated_delta_rule is not supported on alchemist for D!=128"
        )
    torch.manual_seed(42)
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype)
    g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, g, h0 = map(
        lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0)
    )

    ref = []
    ref_ht = []
    for i in range(N):
        idx = slice(cu_seqlens[i], cu_seqlens[i + 1])
        ref_i, ref_ht_i = chunk_gated_delta_rule_native(
            query=q[:, idx],
            key=k[:, idx],
            value=v[:, idx],
            beta=beta[:, idx],
            g=g[:, idx],
            initial_state=h0[i],
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    torch.npu.synchronize()

    actual = []
    actual_ht = []
    for i in range(N):
        idx = slice(cu_seqlens[i], cu_seqlens[i + 1])
        actual_i, actual_ht_i = chunk_gated_delta_rule_native(
            query=q[:, idx],
            key=k[:, idx],
            value=v[:, idx],
            beta=beta[:, idx],
            g=g[:, idx],
            initial_state=h0[i],
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            tri_inv_fn=fast_inv_tril,
        )
        actual.append(actual_i)
        actual_ht.append(actual_ht_i)
    actual = torch.cat(actual, 1)
    actual_ht = torch.cat(actual_ht, 0)

    torch.npu.synchronize()

    print_diff("o", ref, actual, 0.01)
    print_diff("ht", ref_ht, actual_ht, 0.01)

    assert_close("o", ref, actual, 0.01)
    assert_close("ht", ref_ht, actual_ht, 0.01)
