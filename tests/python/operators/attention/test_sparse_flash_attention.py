"""Sparse flash attention through the public attention interface.

The first cases reproduce the legacy test_sparse_flash_attention coverage (TND
query, paged KV, DeepSeek sparse selection). Additional cases cover the dense
KV layouts, the BSND query layout, padded top-k indices and the softmax LSE
outputs. The native kernel requires a single KV head and 512/64 head dims, so
the shapes stay inside that contract.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor, require_distinct_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# FP16/BF16 attention over up to 256 selected tokens accumulates more error than
# the elementwise default. The FP32 entry covers the LSE maximum/sum split,
# which differs only by the reduction order.
_STANDARD = {
    "torch.float16": AccuracyStandard(atol=2e-2, rtol=2e-2),
    "torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=5e-2),
    "torch.float32": AccuracyStandard(atol=2e-2, rtol=2e-2),
}

perf_shape = [
    (torch.bfloat16, "TND", "PA_BSND"),
]

NOPE_DIM = 512
ROPE_DIM = 64
BLOCK_SIZE = 16


def _randn(generator, shape, dtype, scale=1.0):
    value = torch.randn(shape, generator=generator, device="cpu") * scale
    return value.to(dtype).npu()


def _topk_indices(generator, selected, capacity, valid):
    """Left-aligned top-k selection with trailing -1 padding."""
    indices = torch.full((1, capacity), -1, dtype=torch.int32)
    if selected:
        order = torch.randperm(valid, generator=generator, device="cpu")[:selected]
        indices[0, :selected] = order.sort().values
    return indices


def generate_data(
    dtype,
    *,
    layout_query="TND",
    layout_kv="PA_BSND",
    heads=8,
    kv_len=128,
    selected=128,
    capacity=128,
    tokens=1,
    batch=1,
    q_len=1,
    return_softmax_lse=False,
):
    generator = torch.Generator(device="cpu").manual_seed(20261106)

    if layout_query == "TND":
        query = _randn(generator, (tokens, heads, NOPE_DIM), dtype, 0.1)
        query_rope = _randn(generator, (tokens, heads, ROPE_DIM), dtype, 0.1)
        row_indices = _topk_indices(generator, selected, capacity, kv_len)
        sparse_indices = row_indices.expand(tokens, -1).unsqueeze(1).npu().contiguous()
        data = {
            "query": query,
            "query_rope": query_rope,
            "sparse_indices": sparse_indices,
            "actual_seq_lengths_query": torch.tensor(
                [tokens], dtype=torch.int32, device="npu"
            ),
        }
    else:
        query = _randn(generator, (batch, q_len, heads, NOPE_DIM), dtype, 0.1)
        query_rope = _randn(generator, (batch, q_len, heads, ROPE_DIM), dtype, 0.1)
        row_indices = _topk_indices(generator, selected, capacity, kv_len)
        sparse_indices = (
            row_indices.expand(batch, q_len, -1).unsqueeze(2).npu().contiguous()
        )
        data = {
            "query": query,
            "query_rope": query_rope,
            "sparse_indices": sparse_indices,
        }

    if layout_kv == "PA_BSND":
        blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        physical = blocks + 1
        key = _randn(generator, (physical, BLOCK_SIZE, 1, NOPE_DIM), dtype, 0.5)
        key_rope = _randn(generator, (physical, BLOCK_SIZE, 1, ROPE_DIM), dtype, 0.5)
        block_table = (
            torch.stack(
                [
                    torch.randperm(physical, generator=generator, device="cpu")[:blocks]
                    for _ in range(batch)
                ]
            )
            .to(torch.int32)
            .npu()
        )
        data.update(
            {
                "key": key,
                "value": key,
                "key_rope": key_rope,
                "block_table": block_table,
                "actual_seq_lengths_kv": torch.full(
                    (batch,), kv_len, dtype=torch.int32, device="npu"
                ),
            }
        )
    elif layout_kv == "TND":
        key = _randn(generator, (kv_len, 1, NOPE_DIM), dtype, 0.5)
        key_rope = _randn(generator, (kv_len, 1, ROPE_DIM), dtype, 0.5)
        data.update(
            {
                "key": key,
                "value": key,
                "key_rope": key_rope,
                # The TND KV path reads the per-batch length unconditionally.
                "actual_seq_lengths_kv": torch.full(
                    (batch,), kv_len, dtype=torch.int32, device="npu"
                ),
            }
        )
    else:
        key = _randn(generator, (batch, kv_len, 1, NOPE_DIM), dtype, 0.5)
        key_rope = _randn(generator, (batch, kv_len, 1, ROPE_DIM), dtype, 0.5)
        data.update({"key": key, "value": key, "key_rope": key_rope})

    data.update(
        {
            "scale_value": 1.0 / math.sqrt(NOPE_DIM + ROPE_DIM),
            "sparse_block_size": 1,
            "layout_query": layout_query,
            "layout_kv": layout_kv,
            "sparse_mode": 0,
            "attention_mode": 2,
            "return_softmax_lse": return_softmax_lse,
        }
    )
    return data


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "layout_query,layout_kv",
    [("TND", "PA_BSND"), ("TND", "TND"), ("BSND", "BSND"), ("BSND", "PA_BSND")],
    ids=["tnd-pa", "tnd-dense", "bsnd-dense", "bsnd-pa"],
)
@pytest.mark.parametrize("return_softmax_lse", [False, True], ids=["out", "out+lse"])
def test_sparse_flash_attention_layouts(
    dtype, layout_query, layout_kv, return_softmax_lse
):
    """Query/KV layouts, with and without the DCP softmax LSE outputs."""
    from sgl_kernel_npu.operators.attention import sparse_flash_attention

    data = generate_data(
        dtype,
        layout_query=layout_query,
        layout_kv=layout_kv,
        batch=2 if layout_query == "BSND" else 1,
        q_len=2 if layout_query == "BSND" else 1,
        return_softmax_lse=return_softmax_lse,
    )
    executor = check_accuracy(sparse_flash_attention, kwargs=data, standard=_STANDARD)
    check_performance(
        executor,
        enable_perf=return_softmax_lse
        and (dtype, layout_query, layout_kv) in perf_shape,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("selected", [0, 1, 127, 128], ids=str)
def test_sparse_flash_attention_padded_indices(dtype, selected):
    """Trailing -1 padding, including the all-padding zero-output branch.

    An empty row has no finite LSE, but the two generations encode that state
    differently: arch22 masks logits with the finite -2e38 sentinel and keeps a
    non-zero softmax_sum, so ``max + log(sum)`` stays -2e38; arch35 writes
    ``(0, 0)`` which yields -inf. Both mean "no attention mass", so the huge
    negative value is normalized to -inf before comparing.
    """
    from sgl_kernel_npu.operators.attention import sparse_flash_attention

    data = generate_data(
        dtype,
        layout_query="TND",
        layout_kv="PA_BSND",
        kv_len=128,
        selected=selected,
        capacity=256,
        return_softmax_lse=True,
    )
    executor = make_executor(sparse_flash_attention, kwargs=data)
    require_distinct_impls(sparse_flash_attention, executor.impl, executor.impl_ref)
    reference = sparse_flash_attention(
        *executor.args, **{**executor.keywords, "impl": executor.impl_ref}
    )
    out, softmax_max, softmax_sum = executor()

    tolerance = _STANDARD[str(dtype)]
    torch.testing.assert_close(out, reference[0], **vars(tolerance))

    lse = softmax_max.float() + torch.log(softmax_sum.float())
    lse_ref = reference[1].float() + torch.log(reference[2].float())
    assert lse.shape == lse_ref.shape
    if selected == 0:
        # No valid key in any row: collapse both encodings to -inf.
        empty_threshold = -1e30
        lse = torch.where(
            lse < empty_threshold, torch.full_like(lse, float("-inf")), lse
        )
        lse_ref = torch.where(
            lse_ref < empty_threshold, torch.full_like(lse_ref, float("-inf")), lse_ref
        )
        assert torch.isneginf(lse).all()
        assert torch.isneginf(lse_ref).all()
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_sparse_flash_attention_grouped_heads(dtype):
    """Multiple grouped query heads over a single KV head (group > 1)."""
    from sgl_kernel_npu.operators.attention import sparse_flash_attention

    data = generate_data(
        dtype,
        layout_query="TND",
        layout_kv="PA_BSND",
        heads=64,
        kv_len=256,
        selected=256,
        capacity=256,
        tokens=2,
        return_softmax_lse=True,
    )
    check_accuracy(sparse_flash_attention, kwargs=data, standard=_STANDARD)
