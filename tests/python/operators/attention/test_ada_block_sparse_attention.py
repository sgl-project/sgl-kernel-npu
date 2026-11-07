"""AdaBlockSparseAttention accuracy through the public attention interface.

The sparse mask is block-level: entry [b, h, i, j] enables KV block j for query
block i. The device Kernel also applies causal masking inside the diagonal
block, so causal cases build block masks that already drop upper blocks.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# The Kernel accumulates the score/softmax pipeline in FP16, so FP16 outputs
# miss the shared FP16 standard by roughly one extra rounding step.
ADA_ATTENTION_STANDARD = {
    "torch.float16": AccuracyStandard(atol=2e-3, rtol=2e-3),
}

SPARSE_SIZE = 128


def _align32(value):
    return (value + 31) // 32 * 32


def make_sparse_mask(batch, heads, seq_q, seq_kv, mode, causal, dtype):
    """Build a block mask with the aligned last dimension used by production."""
    blocks_q = (seq_q + SPARSE_SIZE - 1) // SPARSE_SIZE
    blocks_kv = (seq_kv + SPARSE_SIZE - 1) // SPARSE_SIZE
    mask = torch.zeros(
        (batch, heads, blocks_q, _align32(blocks_kv)), dtype=torch.int8, device="cpu"
    )
    if mode == "dense":
        mask[..., :blocks_kv] = 1
    elif mode == "diag":
        for index in range(min(blocks_q, blocks_kv)):
            mask[..., index, index] = 1
    elif mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(7)
        values = torch.randint(
            0, 2, (batch, heads, blocks_q, blocks_kv), generator=generator, device="cpu"
        )
        mask[..., :blocks_kv] = values.to(torch.int8)
    elif mode == "empty-row":
        mask[..., :blocks_kv] = 1
        mask[:, :, 0, :] = 0
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    if causal:
        upper = torch.arange(blocks_kv, device="cpu").view(1, 1, 1, -1) > torch.arange(
            blocks_q, device="cpu"
        ).view(1, 1, -1, 1)
        mask[:, :, :, :blocks_kv] = mask[:, :, :, :blocks_kv].masked_fill(upper, 0)
    count = mask[..., :blocks_kv].sum(dim=-1, dtype=torch.int32)
    return mask.to(dtype), count


def generate_test_data(
    batch, heads, kv_heads, seq_q, seq_kv, head_dim, dtype, mode, causal, mask_dtype
):
    generator = torch.Generator(device="cpu").manual_seed(11)
    query = torch.randn(
        (batch, heads, seq_q, head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(dtype)
    key = torch.randn(
        (batch, kv_heads, seq_kv, head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(dtype)
    value = torch.randn(
        (batch, kv_heads, seq_kv, head_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(dtype)
    mask, count = make_sparse_mask(
        batch, heads, seq_q, seq_kv, mode, causal, mask_dtype
    )
    return dict(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        sparse_mask=mask.npu(),
        sparse_count_table=count.npu(),
        input_layout="BNSD",
        sparse_size=SPARSE_SIZE,
        num_heads=heads,
        num_key_value_heads=kv_heads,
        scale_value=head_dim**-0.5,
        causal=causal,
        inner_precise=1,
        pre_tokens=214748647,
        next_tokens=214748647,
        actual_seq_lengths=None,
        actual_seq_lengths_kv=None,
    )


CASES = [
    # Full block mask equals dense attention; sglang's diffusion path is non-causal.
    (1, 4, 4, 256, 256, 128, torch.float16, "dense", False, torch.int8),
    # Only the diagonal KV block is enabled.
    (1, 4, 4, 256, 256, 128, torch.float16, "diag", False, torch.int8),
    # Causal block mask with random sparsity.
    (1, 4, 4, 256, 256, 128, torch.float16, "random", True, torch.int8),
    # GQA with grouped query heads.
    (1, 8, 2, 256, 256, 128, torch.bfloat16, "diag", False, torch.int8),
    # A query block without any enabled KV block produces a zero block.
    (1, 4, 4, 256, 256, 128, torch.float16, "empty-row", False, torch.int8),
    # BOOL masks and a tail block count.
    (1, 2, 2, 384, 384, 128, torch.bfloat16, "random", False, torch.bool),
    # Single-block sequence.
    (1, 4, 4, 128, 128, 128, torch.float16, "dense", False, torch.int8),
]


def _check_case(kwargs, enable_perf):
    from sgl_kernel_npu.operators.attention import ada_block_sparse_attention

    executor = check_accuracy(
        ada_block_sparse_attention, kwargs=kwargs, standard=ADA_ATTENTION_STANDARD
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize(
    "batch,heads,kv_heads,seq_q,seq_kv,head_dim,dtype,mode,causal,mask_dtype",
    CASES,
    ids=[
        "dense-blocks-fp16",
        "diag-blocks-fp16",
        "random-causal-fp16",
        "gqa-bf16",
        "empty-query-block-fp16",
        "bool-mask-tail-blocks-bf16",
        "single-block-fp16",
    ],
)
def test_ada_block_sparse_attention(
    batch, heads, kv_heads, seq_q, seq_kv, head_dim, dtype, mode, causal, mask_dtype
):
    data = generate_test_data(
        batch,
        heads,
        kv_heads,
        seq_q,
        seq_kv,
        head_dim,
        dtype,
        mode,
        causal,
        mask_dtype,
    )
    _check_case(data, enable_perf=mode == "dense" and dtype == torch.float16)


def test_ada_block_sparse_attention_defaults_kv_heads():
    """num_key_value_heads=0 falls back to the query head count."""
    data = generate_test_data(
        1, 4, 4, 256, 256, 128, torch.float16, "diag", False, torch.int8
    )
    data["num_key_value_heads"] = 0
    _check_case(data, enable_perf=False)


@pytest.mark.extra_checks
def test_ada_block_sparse_attention_rejects_tnd():
    """The legacy plugin rejects the TND layout before touching the device."""
    from sgl_kernel_npu.operators.attention import ada_block_sparse_attention

    impl, _ = get_test_impls()
    require_test_impl(ada_block_sparse_attention, impl)
    data = generate_test_data(
        1, 2, 2, 256, 256, 64, torch.float16, "diag", False, torch.int8
    )
    data["input_layout"] = "TND"
    with pytest.raises(RuntimeError, match="does not support 'TND'"):
        ada_block_sparse_attention(**data, impl=impl)
