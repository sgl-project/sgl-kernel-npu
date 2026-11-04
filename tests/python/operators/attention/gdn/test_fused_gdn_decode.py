"""Compare both outputs and modified state with independent inputs."""

from functools import wraps

import pytest

torch = pytest.importorskip("torch")


from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_gdn_decode(varlen, normalize, dtype):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_decode

    count, heads, value_heads, dim = 4, 2, 4, 64
    shape = (1, count) if varlen else (count, 1)
    q = torch.randn(*shape, heads, dim, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(*shape, value_heads, dim, dtype=dtype)
    a = torch.randn(count, value_heads, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(value_heads, dtype=torch.float32)
    bias = torch.randn(value_heads, dtype=torch.float32)
    state = torch.randn(count, value_heads, dim, dim, dtype=torch.float32)
    indices = torch.arange(count, dtype=torch.int32)
    offsets = torch.arange(count + 1, dtype=torch.int32) if varlen else None
    args = (A_log, a, bias, 1.0, 20.0, q, k, v, b)
    kwargs = {"use_qk_l2norm_in_kernel": normalize, "cu_seqlens": offsets}

    @wraps(fused_gdn_decode)
    def isolated_decode(*, impl="auto"):
        current_state = state.clone()
        output = fused_gdn_decode(*args, current_state, indices, impl=impl, **kwargs)
        return output, current_state

    executor = check_accuracy(
        isolated_decode,
        standard={
            str(dtype): AccuracyStandard(atol=0.04, rtol=0.04),
            "torch.float32": AccuracyStandard(atol=0.04, rtol=0.04),
        },
    )
    warmup, active = 5, 5
    states = None

    @wraps(fused_gdn_decode)
    def profile_decode(*, impl="auto"):
        nonlocal states
        # The first call prepares every state before profiler collection starts.
        if states is None:
            states = iter([state.clone() for _ in range(1 + warmup + active)])
        return fused_gdn_decode(*args, next(states), indices, impl=impl, **kwargs)

    perf_executor = make_executor(
        profile_decode,
        impl=executor.impl,
        impl_ref=executor.impl_ref,
        accuracy_checked=executor.accuracy_checked,
    )
    check_performance(
        perf_executor,
        enable_perf=(not varlen and normalize),
        warmup=warmup,
        active=active,
    )
