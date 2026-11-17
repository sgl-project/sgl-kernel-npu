"""Batch-invariance and graph-capture tests for batch-invariant FIA.

Each request's attention output must be bitwise identical alone or inside a
larger batch. Supplementary checks cover the returned OpOverloadPacket, its
.out overload and the workspace/output helpers. Only BSH without a page table
is exercised here; paged TND is covered by the SGLang integration path.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.operators.batch_invariant.batch_invariance import (
    assert_batch_invariant,
    assert_unchanged,
    require_batch_invariant_ops,
    snapshot,
)
from tests.python.utils.common import get_skip_acc_check, get_test_impls

# Two requests, 64 tokens each, 4 query heads, 2 KV heads, head size 64.
fia_shape = (2, 64, 4, 2, 64)


def generate_test_data(shape, dtype, seed=42):
    """Create BSH query/key/value with every request at full length."""
    torch.npu.manual_seed(seed)
    batches, tokens, heads, kv_heads, head_dim = shape
    query = torch.randn(batches, tokens, heads * head_dim, dtype=dtype)
    key = torch.randn(batches, tokens, kv_heads * head_dim, dtype=dtype)
    value = torch.randn(batches, tokens, kv_heads * head_dim, dtype=dtype)
    return query, key, value


def generate_kwargs(shape, batches=None):
    """Build the keyword-only FIA schema for BSH tensors."""
    total, tokens, heads, kv_heads, head_dim = shape
    if batches is None:
        batches = total
    return {
        "num_heads": heads,
        "num_key_value_heads": kv_heads,
        "input_layout": "BSH",
        "scale": head_dim**-0.5,
        "actual_seq_lengths": [tokens] * batches,
        "actual_seq_lengths_kv": [tokens] * batches,
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_infer_attention_score_batch_invariant(dtype):
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import (
        fused_infer_attention_score_batch_invariant,
    )

    impl, _ = get_test_impls()
    checked = get_skip_acc_check() is not True
    query, key, value = generate_test_data(fia_shape, dtype)
    if checked:
        snapshots = snapshot((query, key, value))
        operator = fused_infer_attention_score_batch_invariant(impl=impl)
        full, _ = operator(query, key, value, **generate_kwargs(fia_shape))
        batches = fia_shape[0]
        # A single request is the smallest independent unit of scheduling.
        for offset in (0, batches - 1):
            reference, _ = operator(
                query[offset : offset + 1],
                key[offset : offset + 1],
                value[offset : offset + 1],
                **generate_kwargs(fia_shape, batches=1),
            )
            assert_batch_invariant(
                full[offset : offset + 1],
                reference,
                path=f"batch[{offset}:{offset + 1}]",
            )
        assert_unchanged((query, key, value), snapshots)


@pytest.mark.extra_checks
def test_returns_op_overload_packet():
    """The public entry must expose .default and .out for graph capture."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import (
        fused_infer_attention_score_batch_invariant,
    )

    operator = fused_infer_attention_score_batch_invariant()
    assert isinstance(operator, torch._ops.OpOverloadPacket)
    assert operator.default is not None
    assert operator.out is not None


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_out_overload_matches_default(dtype):
    """The graph-capture chain must reproduce the default overload exactly."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import (
        fused_infer_attention_score_batch_invariant,
        fused_infer_attention_score_batch_invariant_get_max_workspace,
        fused_infer_attention_score_batch_invariant_infer_output,
    )

    if get_skip_acc_check() is True:
        pytest.skip("Accuracy checks are disabled by --skip-acc-check=true")

    impl, _ = get_test_impls()
    query, key, value = generate_test_data(fia_shape, dtype)
    kwargs = generate_kwargs(fia_shape)
    operator = fused_infer_attention_score_batch_invariant(impl=impl)
    expected, _ = operator(query, key, value, **kwargs)
    workspace = fused_infer_attention_score_batch_invariant_get_max_workspace(
        query, key, value, impl=impl, **kwargs
    )
    buffers = list(
        fused_infer_attention_score_batch_invariant_infer_output(
            query,
            value,
            impl=impl,
            input_layout=kwargs["input_layout"],
            num_heads=kwargs["num_heads"],
            num_key_value_heads=kwargs["num_key_value_heads"],
        )
    )
    assert buffers[0].shape == expected.shape
    assert buffers[0].dtype == expected.dtype
    result = operator.out(query, key, value, workspace=workspace, out=buffers, **kwargs)
    assert_batch_invariant(result[0], expected, path="out[0]")
    assert result[0].untyped_storage().data_ptr() == (
        buffers[0].untyped_storage().data_ptr()
    )
