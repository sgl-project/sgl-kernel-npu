"""LaserAttention accuracy through the public attention interface.

sglang calls the dense BNSD path with no masks, keep_prob=1 and a huge
pre_tokens window. The device Kernel never writes softmax_log_max_sum, so only
attention_out is compared; the first element is shape/dtype checked.

The legacy device Kernel only implements the FP16 path (its cube/vector
templates are instantiated with half), so every case feeds FP16 inputs. The
Kernel widens its result to FP32, which means attention_out only carries FP16
precision even though it is a float32 tensor.

The Kernel addresses attention in fixed 128-wide base blocks (H1 = S1 / 128,
H2 = S2 / 128) and never validated the shape. Its core split advances two block
rows at a time (ky = 2 modulo H1), so the sequence length must be a multiple of
256 and head_dim must be 128; other shapes give wrong results or deadlock the
cube/vector handshake.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# attention_out is float32 but computed from FP16 operands inside the Kernel,
# so it can only be as accurate as the FP16 path.
LASER_ATTENTION_STANDARD = {
    "torch.float32": AccuracyStandard(atol=2e-3, rtol=2e-3),
}


def generate_test_data(batch, heads, kv_heads, seq_len, head_dim, dtype):
    """Create BNSD inputs; GQA uses fewer key/value heads than query heads."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    shape = (batch, heads, seq_len, head_dim)
    kv_shape = (batch, kv_heads, seq_len, head_dim)
    query = torch.randn(
        shape, generator=generator, dtype=torch.float32, device="cpu"
    ).to(dtype)
    key = torch.randn(
        kv_shape, generator=generator, dtype=torch.float32, device="cpu"
    ).to(dtype)
    value = torch.randn(
        kv_shape, generator=generator, dtype=torch.float32, device="cpu"
    ).to(dtype)
    return dict(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        atten_mask=None,
        alibi_mask=None,
        drop_mask=None,
        scale_value=head_dim**-0.5,
        head_num=heads,
        input_layout="BNSD",
        keep_prob=1.0,
        pre_tokens=2147483647,
        next_tokens=1,
        is_highPrecision=True,
    )


@pytest.mark.parametrize(
    "batch,heads,kv_heads,seq_len,head_dim,dtype",
    [
        # sglang production call: dense, non-causal, FP16.
        (1, 4, 4, 256, 128, torch.float16),
        # GQA: query heads are grouped over key/value heads.
        (1, 8, 2, 512, 128, torch.float16),
        # Batch of two.
        (2, 4, 4, 256, 128, torch.float16),
        # Six sequence blocks per head.
        (1, 2, 2, 768, 128, torch.float16),
        # MHA at the largest model width.
        (1, 4, 4, 512, 128, torch.float16),
    ],
    ids=[
        "sglang-dense-fp16",
        "gqa-fp16",
        "batch2-fp16",
        "multiblock-seq-fp16",
        "wide-mha-fp16",
    ],
)
def test_laser_attention(batch, heads, kv_heads, seq_len, head_dim, dtype):
    data = generate_test_data(batch, heads, kv_heads, seq_len, head_dim, dtype)
    from sgl_kernel_npu.operators.attention import la

    impl, impl_ref = get_test_impls()
    if get_skip_acc_check():
        require_test_impl(la, impl)
        executor = make_executor(la, kwargs=data, impl=impl, impl_ref=impl_ref)
    else:
        require_distinct_impls(la, impl, impl_ref)
        result_ref = la(**data, impl=impl_ref)
        result = la(**data, impl=impl)
        assert result[0].shape == result_ref[0].shape
        assert result[0].dtype == torch.float32
        assert result[1].dtype == torch.float32
        assert_accuracy(
            result[1],
            result_ref[1],
            standard=LASER_ATTENTION_STANDARD,
            path="attention_out",
        )
        executor = make_executor(
            la, kwargs=data, impl=impl, impl_ref=impl_ref, accuracy_checked=True
        )
    check_performance(
        executor,
        enable_perf=dtype == torch.float16 and seq_len == 256 and head_dim == 128,
    )


@pytest.mark.extra_checks
def test_laser_attention_rejects_bad_rank():
    """The legacy plugin requires a 4D query."""
    from sgl_kernel_npu.operators.attention import la

    impl, _ = get_test_impls()
    require_test_impl(la, impl)
    q = torch.empty((2, 4, 8), dtype=torch.float16, device="npu")
    with pytest.raises(RuntimeError, match="dimension of la must be 4"):
        la(q, q, q, head_num=2, impl=impl)
