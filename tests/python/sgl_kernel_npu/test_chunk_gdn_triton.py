import os
import time
from typing import Optional

import pytest
import sgl_kernel_npu  # noqa: F401  registers npu ops before pytestmark
import sgl_kernel_npu.fla.chunk as chunk_module
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  makes torch.ops.npu namespace available
from sgl_kernel_npu.fla.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_native,
    chunk_gated_delta_rule_npu,
)
from utils import require_npu_op

pytestmark = require_npu_op("mega_chunk_gdn")

LAUNCH_MIN = 2
LAUNCH_CNT = max(2, LAUNCH_MIN)  # specify your run cnt for profiling
device = "npu"


@pytest.fixture(autouse=True)
def force_triton_backend(monkeypatch):
    monkeypatch.setenv("GDN_ATTN_BACKEND_TRITON", "1")


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
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
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
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

    begin_time = 0
    for i in range(LAUNCH_CNT):
        if i == 1 or LAUNCH_CNT == 1:
            torch.npu.synchronize()
            begin_time = time.time()
        _, tri, _, tri_ht, _, _, _ = chunk_gated_delta_rule_fwd(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            beta=beta.clone(),
            g=g.clone(),
            scale=None,
            initial_state=h0.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

    torch.npu.synchronize()
    use_time = time.time() - begin_time
    print(f"[DEBUG] triton using time is {use_time * 1000 / (LAUNCH_CNT-1)}")

    begin_time = 0
    for i in range(LAUNCH_CNT):
        if i == 1 or LAUNCH_CNT == 1:
            torch.npu.synchronize()
        ref = []
        ref_ht = []
        for i in range(N):
            ref_i, ref_ht_i = chunk_gated_delta_rule_native(
                query=q[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                key=k[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                value=v[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                beta=beta[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                g=g[:, cu_seqlens[i] : cu_seqlens[i + 1]],
                initial_state=h0[i],
                output_final_state=True,
            )
            ref.append(ref_i)
            ref_ht.append(ref_ht_i)
        ref = torch.cat(ref, 1)
        ref_ht = torch.cat(ref_ht, 0)

    torch.npu.synchronize()
    use_time = time.time() - begin_time
    print(f"[DEBUG] native using time is {use_time * 1000 / (LAUNCH_CNT-1)}")

    print_diff("o", ref, tri, 0.005)
    print_diff("ht", ref_ht, tri_ht, 0.005)

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


def make_indexed_gdn_case(seed, sequence_lengths=(64, 64), pool_size=6):
    """Build one small varlen case shared by the indexed-state tests."""
    torch.manual_seed(seed)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    num_value_heads, num_key_heads, head_dim = 4, 2, 128
    total_tokens = sum(sequence_lengths)
    offsets = [0]
    for length in sequence_lengths:
        offsets.append(offsets[-1] + length)

    q = torch.randn(
        1, total_tokens, num_key_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    inputs = {
        "q": q,
        "k": torch.randn_like(q),
        "v": torch.randn(
            1,
            total_tokens,
            num_value_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        "g": F.logsigmoid(
            torch.randn(
                1, total_tokens, num_value_heads, dtype=torch.float32, device=device
            )
        ),
        "beta": torch.sigmoid(
            torch.randn(
                1,
                total_tokens,
                num_value_heads,
                dtype=torch.bfloat16,
                device=device,
            )
        ),
        "output_final_state": True,
        "cu_seqlens": torch.tensor(offsets, dtype=torch.long, device=device),
    }
    state_pool = (
        torch.randn(
            pool_size,
            num_value_heads,
            head_dim,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.01
    )
    return inputs, state_pool


def test_chunk_varlen_indexed_state_pool_inplace():
    """Indexed state I/O matches the legacy gather + scatter path exactly."""
    inputs, initial_pool = make_indexed_gdn_case(1234, pool_size=17)
    pool_size = initial_pool.shape[0]
    state_indices = torch.tensor([3, 13], dtype=torch.long, device=device)

    legacy_pool = initial_pool.clone()
    legacy_out, legacy_final, _ = chunk_gated_delta_rule_npu(
        **inputs,
        initial_state=legacy_pool[state_indices],
        use_qk_l2norm_in_kernel=True,
    )
    legacy_pool[state_indices] = legacy_final.to(legacy_pool.dtype)

    direct_pool = initial_pool.clone()
    direct_out, direct_final, _ = chunk_gated_delta_rule_npu(
        **inputs,
        initial_state=direct_pool,
        initial_state_indices=state_indices,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.npu.synchronize()

    assert direct_final is None
    torch.testing.assert_close(direct_out, legacy_out, rtol=0, atol=0)
    torch.testing.assert_close(
        direct_pool[state_indices], legacy_pool[state_indices], rtol=0, atol=0
    )

    untouched = torch.ones(pool_size, dtype=torch.bool, device=device)
    untouched[state_indices] = False
    torch.testing.assert_close(
        direct_pool[untouched], initial_pool[untouched], rtol=0, atol=0
    )


def test_chunk_varlen_indexed_state_pool_returns_compact_final_state():
    """Indexed reads with non-inplace output must write compact final state."""
    inputs, initial_pool = make_indexed_gdn_case(5678, pool_size=17)
    state_indices = torch.tensor([3, 13], dtype=torch.long, device=device)
    initial_pool_before = initial_pool.clone()

    expected_out, expected_final, _ = chunk_gated_delta_rule_npu(
        **inputs,
        initial_state=initial_pool[state_indices],
        use_qk_l2norm_in_kernel=True,
    )
    actual_out, actual_final, _ = chunk_gated_delta_rule_npu(
        **inputs,
        initial_state=initial_pool,
        initial_state_indices=state_indices,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.npu.synchronize()

    assert actual_final.shape == (
        len(state_indices),
        *initial_pool.shape[1:],
    )
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(actual_final, expected_final, rtol=0, atol=0)
    torch.testing.assert_close(initial_pool, initial_pool_before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (torch.tensor([0], dtype=torch.int64), "at least one entry per sequence"),
        (torch.tensor([0, 1], dtype=torch.float32), "int32 or int64"),
    ],
)
def test_chunk_varlen_indexed_state_pool_validates_indices(indices, message):
    inputs, state_pool = make_indexed_gdn_case(9012)

    with pytest.raises(ValueError, match=message):
        chunk_gated_delta_rule_npu(
            **inputs,
            initial_state=state_pool,
            initial_state_indices=indices.to(device),
        )


@pytest.mark.parametrize("invalid_slot", [-1, 6])
def test_chunk_varlen_indexed_state_pool_invalid_slots_are_safe(invalid_slot):
    inputs, state_pool = make_indexed_gdn_case(9012)
    pool_size = state_pool.shape[0]
    state_indices = torch.tensor([invalid_slot, 3], dtype=torch.long, device=device)
    reference_initial = torch.stack([torch.zeros_like(state_pool[0]), state_pool[3]])
    expected_out, expected_final, _ = chunk_gated_delta_rule_npu(
        **inputs,
        initial_state=reference_initial,
    )

    before = state_pool.clone()
    actual_out, actual_final, _ = chunk_gated_delta_rule_npu(
        **inputs,
        initial_state=state_pool,
        initial_state_indices=state_indices,
        inplace_final_state=False,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_final[0], torch.zeros_like(actual_final[0]), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_final[1], expected_final[1], rtol=0, atol=0)
    torch.testing.assert_close(state_pool, before, rtol=0, atol=0)

    inplace_pool = before.clone()
    actual_out, actual_final, _ = chunk_gated_delta_rule_npu(
        **inputs,
        initial_state=inplace_pool,
        initial_state_indices=state_indices,
        inplace_final_state=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    assert actual_final is None
    torch.testing.assert_close(
        inplace_pool[3], expected_final[1].to(inplace_pool.dtype), rtol=0, atol=0
    )
    untouched = torch.ones(pool_size, dtype=torch.bool, device=device)
    untouched[3] = False
    torch.testing.assert_close(
        inplace_pool[untouched], before[untouched], rtol=0, atol=0
    )


def test_mega_gdn_indexed_state_pool_invalid_slots_are_safe(monkeypatch):
    monkeypatch.setenv("GDN_USE_MEGA_GDN", "1")
    state_pool = torch.arange(6 * 1 * 2 * 2, device=device, dtype=torch.bfloat16).view(
        6, 1, 2, 2
    )
    state_indices = torch.tensor([-1, 3], device=device)
    cu_seqlens = torch.tensor([0, 0, 1], dtype=torch.long, device=device)
    q = torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16, device=device)
    v = torch.zeros_like(q)
    g = torch.zeros(1, 1, 1, dtype=torch.float32, device=device)
    beta = torch.ones(1, 1, 1, dtype=torch.bfloat16, device=device)
    produced_final = torch.stack(
        [torch.full_like(state_pool[0], 11), torch.full_like(state_pool[0], 22)]
    ).to(torch.float32)
    gathered_states = []

    def fake_mega(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
    ):
        gathered_states.append(initial_state.clone())
        return g, v, None, produced_final.clone(), None, None, None

    monkeypatch.setattr(chunk_module, "run_mega_chunk_gdn", fake_mega)

    before = state_pool.clone()
    _, compact_final, _ = chunk_gated_delta_rule_npu(
        q,
        q,
        v,
        g,
        beta,
        initial_state=state_pool,
        initial_state_indices=state_indices,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        gathered_states[-1][0], torch.zeros_like(state_pool[0]), rtol=0, atol=0
    )
    torch.testing.assert_close(gathered_states[-1][1], state_pool[3], rtol=0, atol=0)
    torch.testing.assert_close(
        compact_final[0], torch.zeros_like(compact_final[0]), rtol=0, atol=0
    )
    torch.testing.assert_close(compact_final[1], produced_final[1], rtol=0, atol=0)
    torch.testing.assert_close(state_pool, before, rtol=0, atol=0)

    inplace_pool = before.clone()
    _, returned_final, _ = chunk_gated_delta_rule_npu(
        q,
        q,
        v,
        g,
        beta,
        initial_state=inplace_pool,
        initial_state_indices=state_indices,
        output_final_state=True,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    assert returned_final is None
    torch.testing.assert_close(
        inplace_pool[3], produced_final[1].to(inplace_pool.dtype), rtol=0, atol=0
    )
    untouched = torch.ones(inplace_pool.shape[0], dtype=torch.bool, device=device)
    untouched[3] = False
    torch.testing.assert_close(
        inplace_pool[untouched], before[untouched], rtol=0, atol=0
    )


def test_chunk_varlen_indexed_inplace_rejects_noncontiguous_state_pool():
    inputs, state_pool = make_indexed_gdn_case(7890, pool_size=4)
    # Match the speculative NPU pool layout produced by transpose(-1, -2).
    state_pool = state_pool.transpose(-1, -2)
    assert not state_pool.is_contiguous()

    with pytest.raises(ValueError, match="requires a contiguous initial_state"):
        chunk_gated_delta_rule_npu(
            **inputs,
            initial_state=state_pool,
            initial_state_indices=torch.tensor([0, 1], device=device),
            inplace_final_state=True,
        )
