"""Compare both outputs and modified state with independent inputs."""

import pytest

torch = pytest.importorskip("torch")


from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Qwen3-Next TP4: 4 key heads, 8 value heads, K=V=128.
# Batch 16 is a synthetic decode load, not a captured production batch.
perf_shape = [(16, 4, 8, 128)]


def generate_test_data(shape, varlen, normalize, dtype):
    count, heads, value_heads, dim = shape
    token_shape = (1, count) if varlen else (count, 1)
    q = torch.randn(*token_shape, heads, dim, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(*token_shape, value_heads, dim, dtype=dtype)
    a = torch.randn(count, value_heads, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(value_heads, dtype=torch.float32)
    bias = torch.randn(value_heads, dtype=torch.float32)
    state = torch.randn(count, value_heads, dim, dim, dtype=torch.float32)
    indices = torch.arange(count, dtype=torch.int32)
    offsets = torch.arange(count + 1, dtype=torch.int32) if varlen else None
    args = (A_log, a, bias, 1.0, 20.0, q, k, v, b)
    kwargs = {"use_qk_l2norm_in_kernel": normalize, "cu_seqlens": offsets}
    return args, kwargs, state, indices


@pytest.mark.parametrize(
    "shape",
    [(4, 2, 4, 64), (16, 4, 8, 128)],
    ids=["edge-dim64", "qwen3-next-tp4-decode"],
)
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_gdn_decode(shape, varlen, normalize, dtype):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_decode

    args, kwargs, state, indices = generate_test_data(shape, varlen, normalize, dtype)

    # Retain the uniform 0.04 tolerances from the initial manual migration checks.
    # These overrides have not been justified by a separate error analysis.
    standard = {
        str(dtype): AccuracyStandard(atol=0.04, rtol=0.04),
        "torch.float32": AccuracyStandard(atol=0.04, rtol=0.04),
    }
    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(fused_gdn_decode, impl, impl_ref)
    else:
        require_test_impl(fused_gdn_decode, impl)

    if accuracy_checked:
        actual_state, ref_state = state.clone(), state.clone()
        reference = fused_gdn_decode(*args, ref_state, indices, impl=impl_ref, **kwargs)
        actual = fused_gdn_decode(*args, actual_state, indices, impl=impl, **kwargs)
        assert_accuracy((actual, actual_state), (reference, ref_state), standard)

    warmup, active = 5, 5
    states = None

    def executor():
        nonlocal states
        # The first call prepares every state before profiler collection starts.
        if states is None:
            states = iter([state.clone() for _ in range(1 + warmup + active)])
        return fused_gdn_decode(*args, next(states), indices, impl=impl, **kwargs)

    executor.func = fused_gdn_decode
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape and dtype == torch.bfloat16 and not varlen and normalize
        ),
        warmup=warmup,
        active=active,
    )
