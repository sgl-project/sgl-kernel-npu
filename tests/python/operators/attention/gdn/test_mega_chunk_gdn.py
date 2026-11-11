"""Public mega GDN tests with an independent chunk-algebra reference."""

import pytest
import torch

from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


def generate_data(heads, groups, bounds, dtype, with_state, scale=None):
    torch.manual_seed(42)
    tokens = bounds[-1]
    q = torch.nn.functional.normalize(
        torch.randn(1, tokens, groups, 128, dtype=torch.float32), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(dtype)
    return dict(
        q=q,
        k=k,
        v=torch.randn(1, tokens, heads, 128, dtype=dtype),
        g=-torch.rand(1, tokens, heads, dtype=torch.float32) * 0.05,
        beta=torch.rand(1, tokens, heads, dtype=dtype) * 0.5 + 0.5,
        scale=scale,
        initial_state=(
            torch.randn(len(bounds) - 1, heads, 128, 128, dtype=dtype) * 0.1
            if with_state
            else None
        ),
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
    )


def check_mega_accuracy(operator, data):
    executor = make_executor(operator, kwargs=data)
    require_test_impl(operator, executor.impl)
    if get_skip_acc_check() is True:
        return executor
    require_distinct_impls(operator, executor.impl, executor.impl_ref)
    actual = executor()
    reference = operator(**data, impl=executor.impl_ref)
    assert len(actual) == len(reference) == 7
    # Preserve the legacy mega tests' max-absolute OR relative-RMS criterion,
    # extending it to the independently computed intermediate results.
    for index, (result, expected) in enumerate(zip(actual, reference)):
        if expected is None:
            assert result is None
            continue
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        assert result.device == expected.device
        difference = result.detach().float().cpu() - expected.detach().float().cpu()
        assert torch.isfinite(difference).all()
        max_abs = difference.abs().max().item()
        rms = difference.square().mean().sqrt().item()
        base = expected.detach().float().square().mean().sqrt().item()
        ratio = rms / max(base, 1e-8)
        assert (
            max_abs <= 1e-2 or ratio < 0.05
        ), f"return[{index}]: max_abs={max_abs}, relative_rms={ratio}"
    executor.accuracy_checked = True
    return executor


# Cover CANN bridge boundaries and one representative model prefill workload.
@pytest.mark.parametrize(
    "heads,groups,bounds,dtype,with_state,packed,output_final_state,scale",
    [
        # Qwen3-Next-80B-A3B-Instruct has 16 key/32 value heads and K=V=128.
        # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
        # TP4 gives Hg=4/H=8; 2048 tokens represent a prefill workload.
        pytest.param(
            8,
            4,
            (0, 2048),
            torch.bfloat16,
            False,
            False,
            True,
            None,
            id="qwen3-next-tp4-prefill",
        ),
        pytest.param(
            8,
            4,
            (0, 129),
            torch.float16,
            True,
            False,
            True,
            None,
            id="fixed-fp16-chunk-tail",
        ),
        pytest.param(
            12,
            4,
            (0, 22, 151),
            torch.bfloat16,
            True,
            True,
            True,
            0.25,
            id="packed-bf16-state-and-scale",
        ),
        pytest.param(
            8,
            4,
            (0, 1, 130),
            torch.float16,
            False,
            True,
            False,
            None,
            id="packed-without-initial-or-final-state",
        ),
    ],
)
def test_mega_chunk_gdn(
    heads,
    groups,
    bounds,
    dtype,
    with_state,
    packed,
    output_final_state,
    scale,
):
    from sgl_kernel_npu.operators.attention.gdn import mega_chunk_gdn

    data = generate_data(heads, groups, bounds, dtype, with_state, scale)
    if not packed:
        data["cu_seqlens"] = None
    data["output_final_state"] = output_final_state
    saved = {
        name: value.clone()
        for name, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    executor = check_mega_accuracy(mega_chunk_gdn, data)
    for name, value in saved.items():
        torch.testing.assert_close(data[name], value, atol=0, rtol=0)
    check_performance(
        executor,
        enable_perf=bounds == (0, 2048) and dtype == torch.bfloat16,
    )
