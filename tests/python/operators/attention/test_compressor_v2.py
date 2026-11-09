"""Basic model-shaped CompressorV2 output and in-place cache checks."""

import pytest
import torch
from sgl_kernel_npu.operators.attention import compressor_v2

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


def generate_test_data(packed, decode):
    # Borrowed model dimensions, with generated activations (not a request trace):
    # https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/517ef62/config.json
    # H=5120, KV latent D=512, ratio=2; a 255-token prefill leaves one
    # historical token for the following decode to complete its group.
    torch.manual_seed(42)
    batch, hidden, dim, ratio = 2, 5120, 512, 2
    length = 1 if decode else 256
    x = torch.randn(batch, length, hidden, dtype=torch.bfloat16)
    wkv = torch.randn(dim, hidden, dtype=x.dtype) / hidden**0.5
    wgate = torch.randn_like(wkv) / hidden**0.5
    state = torch.zeros(batch, 16, 2 * dim, dtype=torch.float32)
    kwargs = dict(state_block_table=torch.tensor([1, 0], dtype=torch.int32))
    if decode:
        compressor_v2(
            torch.randn(batch, 255, hidden, dtype=x.dtype),
            wkv,
            wgate,
            state,
            ratio,
            **kwargs,
            impl="torch_native",
        )
        kwargs["start_pos"] = torch.full((batch,), 255, dtype=torch.int32)
    if packed:
        x = x.reshape(batch * length, hidden)
        kwargs["cu_seqlens"] = torch.tensor([0, length, 2 * length], dtype=torch.int32)
    return (x, wkv, wgate, state, ratio), kwargs, (1 if decode else length // ratio)


@pytest.mark.parametrize(
    "packed,decode",
    [
        pytest.param(False, False, id="DeepSeek-V4.1-Flash-BSH-prefill"),
        pytest.param(True, True, id="DeepSeek-V4.1-Flash-TH-decode-history"),
    ],
)
def test_compressor_v2(packed, decode):
    args, kwargs, count = generate_test_data(packed, decode)
    x, wkv, wgate, initial, ratio = args
    # check_accuracy shares inputs; this stateful op needs independent caches.
    executor = make_executor(
        compressor_v2, args=(x, wkv, wgate, initial.clone(), ratio), kwargs=kwargs
    )
    if get_skip_acc_check():
        require_test_impl(compressor_v2, executor.impl)
    else:
        require_distinct_impls(compressor_v2, executor.impl, executor.impl_ref)
        actual_state, reference_state = initial.clone(), initial.clone()
        actual = compressor_v2(
            x, wkv, wgate, actual_state, ratio, **kwargs, impl=executor.impl
        )
        reference = compressor_v2(
            x, wkv, wgate, reference_state, ratio, **kwargs, impl=executor.impl_ref
        )
        # The unused output tail is unspecified, including in the packed layout.
        actual = actual[: 2 * count] if packed else actual[:, :count]
        reference = reference[: 2 * count] if packed else reference[:, :count]
        assert_accuracy((actual, actual_state), (reference, reference_state))
        executor.accuracy_checked = True
    # Prefill starts at zero, so repeated writes are idempotent. Cache cloning
    # above is excluded from the public operator's performance measurement.
    check_performance(executor, enable_perf=not decode)
