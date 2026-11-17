"""Sparse attention with shared KV through the public attention interface.

The first cases reproduce the legacy test_sparse_attn_sharedkv coverage
(BSND query, paged KV, SWA). Additional cases cover unaligned sequence lengths,
sliding windows, multiple batches and the TND query layout. The native kernel
requires a single KV head (``N2 == 1``), so grouped KV heads are not exercised.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# FP16/BF16 attention over up to 384 visible tokens accumulates more error than
# the elementwise default, so the legacy test budget is kept explicitly.
_STANDARD = {
    "torch.float16": AccuracyStandard(atol=2e-2, rtol=2e-2),
    "torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=2e-2),
}

perf_shape = [
    (torch.float16, "BSND", 640, 1),
    (torch.bfloat16, "BSND", 640, 1),
]


def _single_core_metadata(device, bn2):
    """Assign the full BN2 range to AIC core 0 (enable, starts, ends)."""
    metadata = torch.zeros(1024, dtype=torch.int32, device=device)
    metadata[0] = 1  # core enable
    metadata[1] = 0  # BN2 start
    metadata[2] = 0  # M start
    metadata[3] = 0  # S2 start
    metadata[4] = bn2  # BN2 end
    metadata[5] = 0  # M end
    metadata[6] = 0  # S2 end
    return metadata


def generate_data(
    dtype,
    *,
    layout_q="BSND",
    batch=1,
    q_len=1,
    kv_len=640,
    heads=64,
    head_dim=512,
    block_size=128,
    win_left=None,
    win_right=0,
):
    """Create paged KV, a nonconsecutive block table and the metadata fixture."""
    generator = torch.Generator(device="cpu").manual_seed(20260812)
    device = "npu"
    # The native kernel requires a single KV head.
    kv_heads = 1

    def randn(shape, scale=1.0, offset=0.0):
        value = torch.randn(shape, generator=generator, device="cpu") * scale + offset
        return value.to(dtype).npu()

    if layout_q == "BSND":
        q = randn((batch, q_len, heads, head_dim), 0.1, 0.05)
        cu_seqlens_q = None
    else:
        q = randn((batch * q_len, heads, head_dim), 0.1, 0.05)
        cu_seqlens_q = torch.tensor(
            [i * q_len for i in range(batch + 1)], dtype=torch.int32, device=device
        )

    blocks = (kv_len + block_size - 1) // block_size
    physical_blocks = blocks + 1
    ori_kv = randn((physical_blocks, block_size, kv_heads, head_dim), 0.5)
    # Distort two pages so a wrong block-table lookup fails clearly.
    if physical_blocks > 5:
        ori_kv[2] = ori_kv[2] + 2.0
        ori_kv[5] = ori_kv[5] - 2.0
    block_table = (
        torch.stack(
            [
                torch.randperm(physical_blocks, generator=generator, device="cpu")[
                    :blocks
                ]
                for _ in range(batch)
            ]
        )
        .to(torch.int32)
        .npu()
    )

    data = {
        "q": q,
        "ori_kv": ori_kv,
        "ori_block_table": block_table,
        "seqused_kv": torch.tensor([kv_len] * batch, dtype=torch.int32, device=device),
        "sinks": torch.full((heads,), -1.0e4, dtype=torch.float32, device=device),
        "metadata": _single_core_metadata(device, batch * kv_heads),
        "softmax_scale": 1.0 / math.sqrt(head_dim),
        "cmp_ratio": 1,
        "ori_win_left": kv_len - 1 if win_left is None else win_left,
        "ori_win_right": win_right,
        "layout_q": layout_q,
        "layout_kv": "PA_ND",
    }
    if cu_seqlens_q is not None:
        data["cu_seqlens_q"] = cu_seqlens_q
    return data


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "kv_len,q_len",
    [(128, 1), (256, 1), (384, 3), (640, 1), (640, 2), (640, 4)],
    ids=lambda value: str(value),
)
@pytest.mark.parametrize("window", ["full", "sliding"], ids=["full", "sliding"])
def test_sparse_attn_sharedkv_swa_bsnd(dtype, kv_len, q_len, window):
    """Legacy SWA/PA_ND path plus window and sequence-length variations."""
    from sgl_kernel_npu.operators.attention import sparse_attn_sharedkv

    data = generate_data(
        dtype,
        kv_len=kv_len,
        q_len=q_len,
        win_left=None if window == "full" else 2,
        win_right=0 if window == "full" else 1,
    )
    executor = check_accuracy(sparse_attn_sharedkv, kwargs=data, standard=_STANDARD)
    check_performance(
        executor,
        enable_perf=window == "full" and (dtype, "BSND", kv_len, q_len) in perf_shape,
    )


@pytest.mark.parametrize(
    "batch,layout_q",
    [
        (2, "BSND"),
        (4, "BSND"),
        (2, "TND"),
        (4, "TND"),
    ],
    ids=["b2-bsnd", "b4-bsnd", "b2-tnd", "b4-tnd"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_sparse_attn_sharedkv_layouts(dtype, batch, layout_q):
    """Multiple batches and the TND query layout."""
    from sgl_kernel_npu.operators.attention import sparse_attn_sharedkv

    data = generate_data(
        dtype,
        layout_q=layout_q,
        batch=batch,
        q_len=2,
        kv_len=384,
    )
    executor = check_accuracy(sparse_attn_sharedkv, kwargs=data, standard=_STANDARD)
    check_performance(executor, enable_perf=False)
