"""GDN chunk output reconstruction, including packed prefill boundaries."""

import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# Generic GDN output stage. Qwen3-Next uses Hg=16/H=32/K=V=128;
# TP4 uses Hg=4/H=8.
# Chunk width 64 follows its NPU GDN caller; sequence lengths are synthetic.
perf_shape = [(1, 2048, 16, 32, 128, 128, 64), (1, 2048, 4, 8, 128, 128, 64)]

# Full-GDN integration scenarios, projected onto this output stage's input shapes.
INTEGRATION_CASES = [
    ((0, 6), 0.0),
    ((0, 31), 0.0),
    ((0, 64), 0.0),
    ((0, 100), 0.0),
    ((0, 127), 0.0),
    ((0, 3584, 7168), 0.0),
    ((0, 3584, 7168), 0.5),
    ((0, 256, 500, 1000), 0.0),
    ((0, 256, 500, 1000), 0.5),
    ((0, 15, 100, 300, 1200, 2000), 0.0),
    ((0, 64, 100, 300, 1200, 2000), 0.0),
    ((0, 64, 300, 1200, 2000), 0.0),
    ((0, 100, 300, 1200, 2000), 0.0),
    ((0, 128, 300, 1200, 2000), 0.0),
    ((0, 256, 300, 1200, 2000), 0.0),
]


def check_integration_accuracy(operator, kwargs):
    """Use max-absolute or relative-RMS error for the full-GDN input domains."""
    executor = make_executor(operator, kwargs=kwargs)
    if get_skip_acc_check() is True:
        require_test_impl(operator, executor.impl)
        return executor
    require_distinct_impls(operator, executor.impl, executor.impl_ref)
    reference = operator(**kwargs, impl=executor.impl_ref)
    actual = executor()
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    assert actual.device == reference.device
    difference = (reference.detach() - actual.detach()).flatten()
    abs_error = difference.abs().max().item()
    error = difference.square().mean().sqrt().item()
    base = reference.detach().flatten().square().mean().sqrt().item()
    error_ratio = error / (base + 1e-8)
    assert (
        abs_error <= 1e-6 or error_ratio < 0.005
    ), f"output diff: {abs_error:.6f} ratio: {error_ratio:.6f}"
    executor.accuracy_checked = True
    return executor


def generate_data(
    shape,
    dtype,
    boundaries=None,
    mask_p=0.0,
    mode="normal",
    index_dtype=torch.int64,
    device=None,
    integration=False,
):
    batch, tokens, key_heads, heads, key_dim, value_dim, chunk = shape
    rng = torch.Generator(device="cpu").manual_seed(44)
    device = torch.get_default_device() if device is None else device
    amplitude = 1.0 if integration else 0.1
    q = (
        torch.randn(batch, tokens, key_heads, key_dim, generator=rng, device="cpu")
        * amplitude
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, key_heads, key_dim, generator=rng, device="cpu"),
        dim=-1,
    ).to(dtype)
    v = (
        torch.randn(batch, tokens, heads, value_dim, generator=rng, device="cpu")
        * amplitude
    ).to(dtype)
    bounds = (0, tokens) if boundaries is None else boundaries
    chunks = sum(
        (end - start + chunk - 1) // chunk
        for start, end in zip(bounds[:-1], bounds[1:])
    )
    h = (
        torch.randn(
            batch, chunks, heads, key_dim, value_dim, generator=rng, device="cpu"
        )
        * amplitude
    ).to(dtype)
    g = torch.empty(batch, tokens, heads, dtype=torch.float32, device="cpu")
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        for start in range(bos, eos, chunk):
            end = min(start + chunk, eos)
            steps = torch.rand(batch, end - start, heads, generator=rng, device="cpu")
            steps = (
                torch.nn.functional.logsigmoid(steps) if integration else -0.05 * steps
            )
            if mask_p:
                steps *= (
                    torch.rand(batch, end - start, heads, generator=rng, device="cpu")
                    > mask_p
                )
            g[:, start:end] = steps.cumsum(1)
    if mode == "zero_state":
        h.zero_()
    elif mode == "no_decay":
        g.zero_()
    elif mode not in ("normal", "offset_gate"):
        raise ValueError(mode)
    data = dict(
        q=q.to(device),
        k=k.to(device),
        v=v.to(device),
        h=h.to(device),
        g=g.to(device),
        cu_seqlens=(
            None
            if boundaries is None
            else torch.tensor(boundaries, dtype=index_dtype, device=device)
        ),
        chunk_size=chunk,
    )
    if mode == "offset_gate":
        backing = torch.full(
            (data["g"].numel() + 8,), 7.0, dtype=data["g"].dtype, device=device
        )
        data["g"] = backing[8:].view_as(data["g"]).copy_(data["g"])
    return data


@pytest.mark.parametrize("heads", [8, 4])
@pytest.mark.parametrize(
    "boundaries,mask_p",
    [
        pytest.param(bounds, mask, id=f"packed-{'-'.join(map(str, bounds))}-mask{mask}")
        for bounds, mask in INTEGRATION_CASES
    ],
)
def test_chunk_o_integration_shapes(heads, boundaries, mask_p, monkeypatch):
    from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    kwargs = generate_data(
        (1, boundaries[-1], heads, heads, 128, 128, 64),
        torch.float16,
        boundaries,
        mask_p,
        integration=True,
    )
    executor = check_integration_accuracy(chunk_fwd_o, kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "shape,boundaries",
    [
        pytest.param(perf_shape[0], None, id="qwen3-next-tp1-prefill"),
        pytest.param(perf_shape[1], None, id="qwen3-next-tp4-prefill"),
        pytest.param(
            (1, 2048, 16, 48, 128, 128, 64), None, id="qwen3.5-27b-tp1-prefill"
        ),
        pytest.param((1, 1, 1, 1, 1, 1, 16), None, id="minimum"),
        pytest.param((1, 33, 1, 3, 17, 33, 32), None, id="feature-and-chunk-tails"),
        pytest.param(
            (1, 82, 2, 4, 17, 33, 16), (0, 1, 17, 82), id="packed-grouped-tail"
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "zero_state", "no_decay"])
def test_chunk_o(shape, boundaries, dtype, mode):
    from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

    kwargs = generate_data(shape, dtype, boundaries, mode=mode)
    executor = check_accuracy(chunk_fwd_o, kwargs=kwargs)
    check_performance(
        executor,
        enable_perf=shape in perf_shape
        and dtype == torch.bfloat16
        and mode == "normal",
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_chunk_o_packed_index_and_scale(index_dtype):
    from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

    kwargs = generate_data(
        (1, 33, 1, 2, 32, 48, 16), torch.float32, (0, 1, 33), index_dtype=index_dtype
    )
    kwargs["scale"] = 0.25
    executor = check_accuracy(chunk_fwd_o, kwargs=kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
def test_chunk_o_requires_gate():
    from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

    impl = get_test_impls()[0]
    require_test_impl(chunk_fwd_o, impl)
    kwargs = generate_data((1, 16, 1, 1, 16, 16, 16), torch.bfloat16)
    kwargs["g"] = None
    with pytest.raises(AttributeError, match="transpose"):
        chunk_fwd_o(**kwargs, impl=impl)


@pytest.mark.extra_checks
def test_chunk_o_inputs_unchanged():
    from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

    impl = get_test_impls()[0]
    require_test_impl(chunk_fwd_o, impl)
    kwargs = generate_data((1, 33, 1, 2, 32, 48, 16), torch.bfloat16, (0, 1, 33))
    inputs = {
        name: value for name, value in kwargs.items() if isinstance(value, torch.Tensor)
    }
    saved = {name: value.clone() for name, value in inputs.items()}
    output = chunk_fwd_o(**kwargs, impl=impl)
    assert output.shape == kwargs["v"].shape
    assert output.data_ptr() not in {value.data_ptr() for value in inputs.values()}
    assert output.dtype == kwargs["v"].dtype
    assert output.device == kwargs["v"].device
    assert_accuracy(
        {name: value.view(torch.uint8) for name, value in inputs.items()},
        {name: value.view(torch.uint8) for name, value in saved.items()},
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_o_gate_storage_offset(dtype):
    from sgl_kernel_npu.operators.attention.gdn import chunk_fwd_o

    kwargs = generate_data((1, 17, 1, 1, 16, 16, 16), dtype, mode="offset_gate")
    assert kwargs["g"].storage_offset() == 8
    executor = check_accuracy(chunk_fwd_o, kwargs=kwargs)
    check_performance(executor, enable_perf=False)
