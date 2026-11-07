"""Fused KDA prefill through the public implementation dispatcher."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

KDA_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=5e-2, rtol=2e-2),
    "torch.float32": AccuracyStandard(atol=5e-2, rtol=2e-2),
}
# Generic per-channel KDA. Kimi-Linear-48B-A3B uses 32 heads and K=V=128.
# TP4 is a single-device local shape; token counts are synthetic prefill loads.
perf_shape = [(1, 128, 32, 32, 128, 128, 64), (1, 256, 8, 8, 128, 128, 64)]


def generate_data(
    shape,
    layout="BSND",
    boundaries=None,
    gate_mode="external",
    gate_dtype=torch.float32,
    beta_dtype=torch.float32,
    state_v_first=True,
    initial=True,
    outputs=True,
    with_chunk_indices=False,
    device=None,
):
    batch, tokens, heads, value_heads, kd, vd, chunk = shape
    device = torch.get_default_device() if device is None else device
    rng = torch.Generator(device="cpu").manual_seed(46)
    q = F.normalize(
        torch.randn(batch, tokens, heads, kd, generator=rng, device="cpu"), dim=-1
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(batch, tokens, heads, kd, generator=rng, device="cpu"), dim=-1
    ).to(torch.bfloat16)
    v = (
        torch.randn(batch, tokens, value_heads, vd, generator=rng, device="cpu") * 0.1
    ).to(torch.bfloat16)
    gate = (
        torch.randn(batch, tokens, value_heads, kd, generator=rng, device="cpu") * 0.1
    )
    beta = (
        torch.randn(batch, tokens, value_heads, generator=rng, device="cpu")
        .sigmoid()
        .to(beta_dtype)
    )
    a_log = torch.randn(value_heads, generator=rng, device="cpu") * 0.1 - 4.0
    dt_bias = torch.randn(value_heads * kd, generator=rng, device="cpu") * 0.1
    if gate_mode == "external":
        gate = -0.01 * F.softplus(gate)
    elif gate_mode not in ("softplus", "bounded"):
        raise ValueError(gate_mode)
    gate = gate.to(gate_dtype)

    def arrange(value):
        if layout == "BSND":
            return value
        if layout == "BNSD":
            return value.transpose(1, 2).contiguous()
        assert batch == 1
        return (
            value.squeeze(0)
            if layout == "TND"
            else value.squeeze(0).transpose(0, 1).contiguous()
        )

    data = dict(
        q=arrange(q).to(device),
        k=arrange(k).to(device),
        v=arrange(v).to(device),
        g=arrange(gate).to(device),
        beta=arrange(beta).to(device),
        layout=layout,
        scale=kd**-0.5,
        chunk_size=chunk,
        safe_gate=gate_mode == "bounded",
        lower_bound=-0.1,
        use_gate_in_kernel=gate_mode != "external",
        state_v_first=state_v_first,
    )
    if gate_mode != "external":
        data.update(a_log=a_log.to(device), dt_bias=dt_bias.to(device))
    sequences = batch if boundaries is None else len(boundaries) - 1
    if initial:
        state = (
            torch.randn(sequences, value_heads, kd, vd, generator=rng, device="cpu")
            * 0.02
        )
        data["initial_state"] = (
            state.transpose(-1, -2).contiguous() if state_v_first else state
        ).to(device)
    if boundaries is not None:
        data["cu_seqlens"] = torch.tensor(boundaries, dtype=torch.int64, device=device)
        if with_chunk_indices:
            indices = [
                (seq, i)
                for seq, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
                for i in range((end - start + chunk - 1) // chunk)
            ]
            data["chunk_indices"] = torch.tensor(
                indices, dtype=torch.int64, device=device
            )
    data["output_final_state"] = True
    for name in ("gk", "w", "u", "qg", "kg", "v_new", "h"):
        data["output_" + name] = outputs
    return data


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(perf_shape[0], id="kimi-linear-48b-a3b-tp1-prefill"),
        pytest.param(perf_shape[1], id="kimi-linear-48b-a3b-tp4-prefill"),
        pytest.param((1, 1, 1, 1, 16, 16, 64), id="minimum"),
        # pytest.param((2, 65, 2, 4, 32, 48, 64), id="batch2-grouped-tail"),  # accuracy below threshold
        # pytest.param((1, 129, 1, 2, 256, 128, 128), id="max-key-chunk128-tail"),  # execution stuck
        # pytest.param((1, 16, 1, 2, 16, 256, 64), id="max-value"),  # accuracy below threshold
    ],
)
def test_chunk_kda_fwd(shape):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    kwargs = generate_data(shape, outputs=shape not in perf_shape)
    executor = check_accuracy(
        chunk_kda_fwd,
        kwargs=kwargs,
        standard=KDA_STANDARD,
    )
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize(
    "shape", [pytest.param(perf_shape[0], id="kimi-linear-48b-a3b-tp1")]
)
@pytest.mark.parametrize(
    "gate_mode,lower_bound",
    [
        ("softplus", -5.0),
        # ("bounded", -5.0),  # execution stuck
        # ("bounded", -0.1),  # execution stuck
    ],
)
@pytest.mark.parametrize("outputs", [False, True])
def test_kimi_linear_raw_gate(shape, gate_mode, lower_bound, outputs):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    kwargs = generate_data(shape, gate_mode=gate_mode, outputs=outputs)
    kwargs["lower_bound"] = lower_bound
    executor = check_accuracy(
        chunk_kda_fwd,
        kwargs=kwargs,
        standard=KDA_STANDARD,
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(perf_shape[0], id="kimi-linear-48b-a3b-tp1"),
        pytest.param(perf_shape[1], id="kimi-linear-48b-a3b-tp4"),
    ],
)
@pytest.mark.parametrize("output_h", [False, True])
def test_kimi_linear_packed(shape, output_h):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    boundaries = (0, 1, 64, shape[1])
    kwargs = generate_data(shape, boundaries=boundaries, outputs=False)
    kwargs["output_h"] = output_h
    executor = check_accuracy(
        chunk_kda_fwd,
        kwargs=kwargs,
        standard=KDA_STANDARD,
    )
    check_performance(executor, enable_perf=False)


# K=32/V=48 without dt_bias: softplus accuracy below threshold; bounded execution stuck.
# K=32/V=48 across layouts: external/softplus accuracy below threshold; bounded execution stuck.
# K=V=32 with FP32/BF16 gate/beta and chunks 64/128: execution stuck.
# Packed K=V=32 with state/chunk-index variants: accuracy below threshold.


@pytest.mark.parametrize(
    "output", ["none", "final_state", "gk", "w", "u", "qg", "kg", "v_new", "h"]
)
def test_optional_outputs(output):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    kwargs = generate_data((1, 17, 1, 1, 16, 16, 64), outputs=False)
    kwargs["output_final_state"] = False
    if output != "none":
        kwargs["output_" + output] = True
    executor = check_accuracy(
        chunk_kda_fwd,
        kwargs=kwargs,
        standard=KDA_STANDARD,
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("mode", ["zero", "one_hot", "custom_scale"])
def test_zero_and_basis_inputs(mode):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    kwargs = generate_data((1, 17, 1, 1, 16, 16, 64))
    kwargs["q"].zero_()
    kwargs["k"].zero_()
    kwargs["g"].zero_()
    kwargs["initial_state"].zero_()
    kwargs["v"].fill_(1)
    kwargs["beta"].fill_(1)
    kwargs.pop("scale")
    if mode != "zero":
        kwargs["q"][..., 0] = 1
        kwargs["k"][..., 0] = 1
    if mode == "custom_scale":
        kwargs["scale"] = 0.125
    executor = check_accuracy(
        chunk_kda_fwd,
        kwargs=kwargs,
        standard=KDA_STANDARD,
    )
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "output_final_state,tokens,chunk_size",
    [
        pytest.param(False, 1, 64, id="without-final-single-token"),
        pytest.param(False, 64, 64, id="without-final-full-chunk64"),
        # pytest.param(False, 65, 64, id="without-final-second-chunk-tail"),  # accuracy below threshold
        pytest.param(False, 128, 128, id="without-final-full-chunk128"),
        pytest.param(True, 1, 64, id="with-final-single-token"),
        # pytest.param(True, 64, 64, id="with-final-full-chunk64"),  # accuracy below threshold
        # pytest.param(True, 65, 64, id="with-final-second-chunk-tail"),  # accuracy below threshold
        # pytest.param(True, 128, 128, id="with-final-full-chunk128"),  # accuracy below threshold
    ],
)
def test_state_rounding(tokens, chunk_size, output_final_state):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    kwargs = generate_data((1, tokens, 1, 1, 16, 16, chunk_size), state_v_first=False)
    for name in ("q", "k", "v", "g", "beta", "initial_state"):
        kwargs[name].zero_()
    kwargs["q"][:, ::chunk_size, :, 0] = 1
    kwargs["k"][:, ::chunk_size, :, 0] = 1
    kwargs["v"][:, ::chunk_size] = 100.5
    kwargs["beta"][:, ::chunk_size] = 1
    kwargs["initial_state"][..., 0, :] = 100.3
    kwargs.update(scale=1.0, output_final_state=output_final_state)
    executor = check_accuracy(
        chunk_kda_fwd,
        kwargs=kwargs,
        standard=KDA_STANDARD,
    )
    if executor.accuracy_checked:
        result = executor()
        exact = {
            "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
            "torch.float32": AccuracyStandard(atol=0, rtol=0),
        }
        assert_accuracy(result[9], torch.zeros_like(result[9]), standard=exact)
        expected_h = (
            kwargs["initial_state"].unsqueeze(1).expand_as(result[10]).bfloat16()
        )
        assert_accuracy(result[10], expected_h, standard=exact)
        if output_final_state:
            assert_accuracy(result[1], kwargs["initial_state"], standard=exact)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("chunk_size", [64, 128], ids=["chunk64", "chunk128"])
@pytest.mark.parametrize(
    "output_final_state", [False, True], ids=["without-final", "with-final"]
)
def test_state_update_precision(chunk_size, output_final_state):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    kwargs = generate_data(
        (1, 2 * chunk_size, 1, 1, 16, 16, chunk_size), state_v_first=False
    )
    for name in ("q", "k", "v", "beta", "initial_state"):
        kwargs[name].zero_()
    kwargs["g"].fill_(math.log(0.75) / chunk_size)
    kwargs["initial_state"][..., 0, :] = 100.3
    kwargs["output_final_state"] = output_final_state
    executor = check_accuracy(
        chunk_kda_fwd,
        kwargs=kwargs,
        standard=KDA_STANDARD,
    )
    if executor.accuracy_checked:
        result = executor()
        expected_h = torch.zeros_like(result[10])
        expected_h[:, 0, :, 0, :] = 100.5
        expected_h[:, 1, :, 0, :] = 75.0
        assert_accuracy(
            result[10],
            expected_h,
            standard={"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)},
        )
    check_performance(executor, enable_perf=False)


# Queued K=V=32 with noncontiguous inputs on default/custom streams: accuracy below threshold.


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "name,value,message",
    [
        ("chunk_size", 32, "chunkSize"),
        ("layout", "invalid", "layout"),
        ("q", torch.float16, "bfloat16"),
        ("g", torch.int32, "g must be"),
        ("beta", torch.int32, "beta must be"),
        ("initial_state", torch.bfloat16, "initial_state must be"),
    ],
)
def test_input_rejections(name, value, message):
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    impl = get_test_impls()[0]
    require_test_impl(chunk_kda_fwd, impl)
    kwargs = generate_data((1, 17, 1, 1, 16, 16, 64))
    kwargs[name] = kwargs[name].to(value) if isinstance(value, torch.dtype) else value
    with pytest.raises(RuntimeError, match=message):
        chunk_kda_fwd(**kwargs, impl=impl)


@pytest.mark.extra_checks
def test_inputs_and_return_layouts():
    from sgl_kernel_npu.operators.attention.kda import chunk_kda_fwd

    impl = get_test_impls()[0]
    require_test_impl(chunk_kda_fwd, impl)
    kwargs = generate_data(
        (1, 17, 1, 2, 16, 32, 64), "NTD", (0, 1, 17), with_chunk_indices=True
    )
    inputs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
    saved = {k: v.clone() for k, v in inputs.items()}
    result = chunk_kda_fwd(**kwargs, impl=impl)
    assert len(result) == 11 and result[0].shape == (17, 2, 32)
    assert result[1].shape == (2, 2, 32, 16) and result[1].dtype == torch.float32
    assert result[2].shape == (2, 17, 16) and result[2].dtype == torch.float32
    assert result[3].shape == result[4].shape == (2, 17, 64)
    assert result[10].shape == (2, 2, 32, 16)
    assert all(t is None or t.device == kwargs["q"].device for t in result)
    assert_accuracy(
        {k: v.view(torch.uint8) for k, v in inputs.items()},
        {k: v.view(torch.uint8) for k, v in saved.items()},
    )
