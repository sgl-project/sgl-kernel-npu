"""Stateful CompressorV2 accuracy: valid outputs plus every cache element."""

from functools import partial

import pytest
import torch
from sgl_kernel_npu.operators.attention import compressor_v2
from sgl_kernel_npu.operators.utils.registry import register_impl, register_operator

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor, require_test_impl
from tests.python.utils.performance import check_performance


@register_operator("test.compressor_v2.snapshot")
def snapshot(args, kwargs, counts, *, impl="auto"):
    x, wkv, wgate, initial, ratio = args
    state = torch.empty_strided(
        initial.shape, initial.stride(), dtype=initial.dtype, device=initial.device
    )
    state.copy_(initial)
    result = compressor_v2(x, wkv, wgate, state, ratio, **kwargs, impl=impl)
    if x.ndim == 3:
        valid = torch.cat([result[b, :count] for b, count in enumerate(counts)], dim=0)
    else:
        valid = result[: sum(counts)]
    return valid, state


for _impl in ("auto", "cann_ops_transformers_cann920", "torch_native"):
    register_impl("test.compressor_v2.snapshot", _impl)(partial(snapshot, impl=_impl))


@register_operator("test.compressor_v2.sequence")
def sequence(args, blocks, *, impl="auto"):
    x, wkv, wgate, initial, ratio = args
    state = initial.clone()
    outputs = []
    for position in (0, 2, 4):
        start = torch.full((x.shape[0],), position, dtype=torch.int32, device=x.device)
        chunk = x[:, position : position + 2].contiguous()
        result = compressor_v2(
            chunk,
            wkv,
            wgate,
            state,
            ratio,
            state_block_table=blocks,
            start_pos=start,
            impl=impl,
        )
        count = (position % ratio + 2) // ratio
        outputs.append((result[:, :count].clone(), state.clone()))
    return tuple(outputs)


for _impl in ("auto", "cann_ops_transformers_cann920", "torch_native"):
    register_impl("test.compressor_v2.sequence", _impl)(partial(sequence, impl=_impl))


def generate_test_data(
    dtype,
    *,
    packed=False,
    hidden=1024,
    dim=128,
    ratio=4,
    length=17,
    continuation=False,
    partial_length=False,
    strided=False
):
    # Synthetic representative dimensions, not traced production requests.
    torch.manual_seed(42)
    batch, capacity = 2, max(16, ratio)
    x = torch.randn((batch, length, hidden), dtype=dtype) * 0.01
    wkv = torch.randn((dim, hidden), dtype=dtype) * 0.01
    wgate = torch.randn_like(wkv) * 0.01
    state = (
        torch.randn(
            (batch * (2 if strided else 1), capacity, 2 * dim), dtype=torch.float32
        )
        * 0.01
    )
    if strided:
        state = state[::2]
    lengths = [length, length]
    used = [max(0, length - 1), max(0, length - 3)] if partial_length else lengths
    starts = [ratio - 1, 2 * ratio + 1] if continuation else [0, 0]
    kwargs = {"state_block_table": torch.tensor([1, 0], dtype=torch.int32)}
    if continuation:
        kwargs["start_pos"] = torch.tensor(starts, dtype=torch.int32)
    if partial_length:
        kwargs["seqused"] = torch.tensor(used, dtype=torch.int32)
    if packed:
        x = x.reshape(batch * length, hidden)
        kwargs["cu_seqlens"] = torch.tensor([0, length, 2 * length], dtype=torch.int32)
    counts = [(s % ratio + n) // ratio for s, n in zip(starts, used)]
    return (x, wkv, wgate, state, ratio), kwargs, counts


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "scenario", ["prefill", "continue", "partial", "wrap", "empty"]
)
def test_compressor_v2(dtype, packed, scenario):
    require_test_impl(compressor_v2, "auto")
    args, kwargs, counts = generate_test_data(
        dtype,
        packed=packed,
        continuation=scenario in ("continue", "wrap"),
        partial_length=scenario == "partial",
        strided=scenario == "wrap",
        length=0 if scenario == "empty" else (37 if scenario == "wrap" else 17),
    )
    checked = check_accuracy(snapshot, args=(args, kwargs, counts))
    executor = make_executor(
        compressor_v2,
        args=args,
        kwargs=kwargs,
        accuracy_checked=checked.accuracy_checked,
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("ratio", [2, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("dim", [128, 512])
def test_compression_ratios(ratio, dim):
    require_test_impl(compressor_v2, "auto")
    args, kwargs, counts = generate_test_data(
        torch.bfloat16, ratio=ratio, dim=dim, length=ratio + 1
    )
    checked = check_accuracy(snapshot, args=(args, kwargs, counts))
    check_performance(
        make_executor(
            compressor_v2,
            args=args,
            kwargs=kwargs,
            accuracy_checked=checked.accuracy_checked,
        ),
        enable_perf=False,
    )


def test_decode_with_history():
    require_test_impl(compressor_v2, "auto")
    # A one-token decode finishes a group only for the first batch.
    args, kwargs, counts = generate_test_data(
        torch.float16, length=1, continuation=True
    )
    checked = check_accuracy(snapshot, args=(args, kwargs, counts))
    check_performance(
        make_executor(
            compressor_v2,
            args=args,
            kwargs=kwargs,
            accuracy_checked=checked.accuracy_checked,
        ),
        enable_perf=False,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sequential_calls(dtype):
    require_test_impl(compressor_v2, "auto")
    args, kwargs, _ = generate_test_data(dtype, length=6)
    # Every step compares both valid output and retained history for the next step.
    check_accuracy(sequence, args=(args, kwargs["state_block_table"]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_variable_packed_lengths(dtype):
    require_test_impl(compressor_v2, "auto")
    x = torch.randn(12, 1024, dtype=dtype) * 0.01
    wkv = torch.randn(128, 1024, dtype=dtype) * 0.01
    wgate = torch.randn_like(wkv) * 0.01
    state = torch.randn(4, 16, 256, dtype=torch.float32) * 0.01
    kwargs = dict(
        state_block_table=torch.tensor([2, 0, 1], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 0, 3, 12], dtype=torch.int32),
        start_pos=torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    # Empty first sequence, unaligned history, compacted output and an unused block.
    check_accuracy(snapshot, args=((x, wkv, wgate, state, 4), kwargs, [0, 1, 3]))


@pytest.mark.parametrize("hidden,dim,ratio", [(5120, 512, 4), (5120, 128, 128)])
def test_representative_performance(hidden, dim, ratio):
    require_test_impl(compressor_v2, "auto")
    args, kwargs, counts = generate_test_data(
        torch.bfloat16, hidden=hidden, dim=dim, ratio=ratio, length=128
    )
    checked = check_accuracy(snapshot, args=(args, kwargs, counts))
    executor = make_executor(
        compressor_v2,
        args=args,
        kwargs=kwargs,
        accuracy_checked=checked.accuracy_checked,
    )
    # Aligned start positions make repeated writes idempotent; no reset/copy is timed.
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize("packed", [False, True], ids=["BSH", "TH"])
@pytest.mark.parametrize(
    "scenario",
    ["prefill", "decode"],
    ids=["DeepSeek-V4.1-Flash-prefill", "DeepSeek-V4.1-Flash-decode"],
)
def test_model_scenario(scenario, packed):
    # Model dimensions and ratio from text_config, not captured activations:
    # https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/517ef62/config.json
    # H=5120, KV latent D=512, ratio=2. Requests use a 256-token prompt,
    # then one-token decode that completes the preceding partial group.
    require_test_impl(compressor_v2, "auto")
    args, kwargs, counts = generate_test_data(
        torch.bfloat16,
        packed=packed,
        hidden=5120,
        dim=512,
        ratio=2,
        length=256 if scenario == "prefill" else 1,
    )
    x, wkv, wgate, state, ratio = args
    # Match normalized hidden states and fan-in-scaled projection weights.
    x = x * 100
    wkv = wkv * (100 / 5120**0.5)
    wgate = wgate * (100 / 5120**0.5)
    state.zero_()
    if scenario == "decode":
        history = torch.randn(2, 255, 5120, dtype=x.dtype)
        compressor_v2(
            history,
            wkv,
            wgate,
            state,
            ratio,
            state_block_table=kwargs["state_block_table"],
            impl="torch_native",
        )
        kwargs["start_pos"] = torch.full((2,), 255, dtype=torch.int32)
        counts = [1, 1]
    check_accuracy(snapshot, args=((x, wkv, wgate, state, ratio), kwargs, counts))
