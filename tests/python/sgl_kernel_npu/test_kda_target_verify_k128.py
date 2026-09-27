"""Accuracy coverage for the standalone K=128 split-K target verify candidate.

The reference below defines this candidate's ``rsqrt(sum(x*x) + 1e-12)``
normalization.  Passing these tests does not establish equivalence to the
different normalization formulas in other target-verify implementations.
"""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  Register NPU before importing the Triton module.
from sgl_kernel_npu.fla.kda_target_verify_k128 import kda_target_verify_k128_npu

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")

HEADS = 12
KEY_DIM = 128
VALUE_DIM = 128
SCALE = KEY_DIM**-0.5
SENTINEL = -3.125


def _make_case(batch, steps, layout, gate_mode, scale_mode, state_dtype, qk_scale=1.0):
    generator = torch.Generator(device="cpu").manual_seed(
        20260927 + batch * 100 + steps
    )
    tokens = batch * steps

    def normal(shape, scale=1.0):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    # The split preserves the packed projection's token stride, as in SGLang.
    packed_cpu = normal((tokens, HEADS * (2 * KEY_DIM + VALUE_DIM)))
    packed_cpu[:, : HEADS * 2 * KEY_DIM] *= qk_scale
    packed_cpu = packed_cpu.to(torch.bfloat16)
    packed = packed_cpu.to("npu")

    def unpack(tensor):
        q, k, v = tensor.split(
            (HEADS * KEY_DIM, HEADS * KEY_DIM, HEADS * VALUE_DIM), dim=-1
        )
        return (
            q.reshape(1, tokens, HEADS, KEY_DIM),
            k.reshape(1, tokens, HEADS, KEY_DIM),
            v.reshape(1, tokens, HEADS, VALUE_DIM),
        )

    q_cpu, k_cpu, v_cpu = unpack(packed_cpu)
    q, k, v = unpack(packed)
    a_log_cpu = normal((1, 1, HEADS, 1), 0.1) - 1.0
    dt_bias_cpu = normal((HEADS, KEY_DIM), 0.1)
    raw_a_cpu = normal((tokens, HEADS, KEY_DIM), 0.2)
    raw_b_cpu = normal((tokens, HEADS), 0.5)
    if gate_mode == "preactivated":
        a_cpu = -5.0 * torch.sigmoid(
            a_log_cpu.reshape(HEADS, 1).exp() * (raw_a_cpu + dt_bias_cpu)
        )
        b_cpu = raw_b_cpu.sigmoid()
        a = a_cpu.to("npu").unsqueeze(0)
        b = b_cpu.to("npu").unsqueeze(0)
    else:
        a_cpu, b_cpu = raw_a_cpu, raw_b_cpu
        packed_gate = torch.cat((raw_a_cpu.flatten(1), raw_b_cpu), dim=-1).to("npu")
        a_flat, b = packed_gate.split((HEADS * KEY_DIM, HEADS), dim=-1)
        a = a_flat.reshape(1, tokens, HEADS, KEY_DIM)
        b = b.unsqueeze(0)

    initial_cpu = normal((batch + 2, HEADS, VALUE_DIM, KEY_DIM), 0.02).to(state_dtype)
    if layout == "framework":
        # NPU temporal cache is a V/K-transposed view with the same logical shape.
        initial = initial_cpu.transpose(-1, -2).contiguous().to("npu").transpose(-1, -2)
        assert all(not tensor.is_contiguous() for tensor in (q, k, v))
        assert not initial.is_contiguous()
    else:
        q, k, v, a, b = (tensor.contiguous() for tensor in (q, k, v, a, b))
        initial = initial_cpu.to("npu")
        assert all(tensor.is_contiguous() for tensor in (q, k, v, a, b, initial))

    # Non-identity slots expose accidental assumptions that request index == slot.
    initial_indices_cpu = torch.arange(batch, dtype=torch.int64).flip(0)
    snapshot_indices_cpu = torch.arange(1, batch + 1, dtype=torch.int64).flip(0)
    snapshots = torch.full(
        (batch + 2, steps, HEADS, VALUE_DIM, KEY_DIM),
        SENTINEL,
        dtype=state_dtype,
        device="npu",
    )
    kwargs = dict(
        A_log=a_log_cpu.to("npu"),
        # The measured candidate supports the framework's flat bias tensor.
        dt_bias=dt_bias_cpu.flatten().to("npu"),
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial,
        initial_state_indices=initial_indices_cpu.to(device="npu", dtype=torch.int32),
        intermediate_states_buffer=snapshots,
        intermediate_state_indices=snapshot_indices_cpu.to(
            device="npu", dtype=torch.int32
        ),
        cache_steps=steps,
        gates_are_preactivated=gate_mode == "preactivated",
    )
    if gate_mode == "raw_safe":
        kwargs["lower_bound"] = -5.0
    if scale_mode == "explicit":
        kwargs["scale"] = SCALE
    else:
        assert "scale" not in kwargs

    reference_inputs = dict(
        q=q_cpu.float()[0],
        k=k_cpu.float()[0],
        v=v_cpu.float()[0],
        a=a_cpu,
        b=b_cpu,
        A_log=a_log_cpu.reshape(HEADS),
        dt_bias=dt_bias_cpu,
        initial=initial_cpu,
        initial_indices=initial_indices_cpu,
        snapshot_indices=snapshot_indices_cpu,
        batch=batch,
        steps=steps,
        gate_mode=gate_mode,
    )
    return kwargs, reference_inputs


def _reference(inputs):
    """CPU FP32 recurrence; intermediates stay FP32 even for BF16 state storage."""
    batch, steps = inputs["batch"], inputs["steps"]
    q = inputs["q"] * torch.rsqrt(inputs["q"].square().sum(-1, keepdim=True) + 1e-12)
    k = inputs["k"] * torch.rsqrt(inputs["k"].square().sum(-1, keepdim=True) + 1e-12)
    q = q * SCALE
    if inputs["gate_mode"] == "preactivated":
        log_decay, beta = inputs["a"], inputs["b"]
    else:
        gate_input = inputs["a"] + inputs["dt_bias"]
        exp_a = inputs["A_log"].exp().unsqueeze(-1)
        if inputs["gate_mode"] == "raw_safe":
            log_decay = -5.0 * torch.sigmoid(exp_a * gate_input)
        else:
            log_decay = -exp_a * F.softplus(gate_input)
        beta = inputs["b"].sigmoid()

    output = torch.empty((1, batch * steps, HEADS, VALUE_DIM), dtype=torch.float32)
    snapshots = torch.empty(
        (batch, steps, HEADS, VALUE_DIM, KEY_DIM), dtype=torch.float32
    )
    for request in range(batch):
        state = inputs["initial"][inputs["initial_indices"][request]].float().clone()
        for step in range(steps):
            token = request * steps + step
            state = state * log_decay[token].exp().unsqueeze(-2)
            predicted = torch.einsum("hvk,hk->hv", state, k[token])
            delta = (inputs["v"][token] - predicted) * beta[token].unsqueeze(-1)
            state = state + delta.unsqueeze(-1) * k[token].unsqueeze(-2)
            output[0, token] = torch.einsum("hvk,hk->hv", state, q[token])
            snapshots[request, step] = state
    return output, snapshots


def _assert_close(actual, expected):
    tolerance = 5e-3 if actual.dtype == torch.bfloat16 else 1e-5
    actual = actual.detach().cpu().float()
    expected = expected.float()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _check_result(output, kwargs, inputs, expected):
    expected_output, expected_states = expected
    _assert_close(output, expected_output)
    actual_pool = kwargs["intermediate_states_buffer"].detach().cpu()
    actual_states = actual_pool[inputs["snapshot_indices"]]
    for step in range(inputs["steps"]):
        _assert_close(actual_states[:, step], expected_states[:, step])
    # State source is read-only, and unused scratch slots must remain untouched.
    torch.testing.assert_close(
        kwargs["initial_state_source"].detach().cpu(), inputs["initial"], rtol=0, atol=0
    )
    assert torch.all(actual_pool[0] == SENTINEL)
    assert torch.all(actual_pool[-1] == SENTINEL)


@pytest.mark.parametrize(
    "batch,steps,layout,gate_mode,scale_mode,state_dtype",
    [
        pytest.param(
            8,
            4,
            "framework",
            "preactivated",
            "omitted",
            torch.float32,
            id="B8-S4-framework",
        ),
        pytest.param(
            4,
            8,
            "framework",
            "preactivated",
            "omitted",
            torch.float32,
            id="B4-S8-framework",
        ),
        pytest.param(
            8,
            8,
            "framework",
            "preactivated",
            "omitted",
            torch.float32,
            id="B8-S8-framework",
        ),
        pytest.param(
            8,
            4,
            "contiguous",
            "preactivated",
            "explicit",
            torch.float32,
            id="contiguous-preactivated",
        ),
        pytest.param(
            8,
            4,
            "framework",
            "raw_softplus",
            "explicit",
            torch.float32,
            id="framework-raw-softplus",
        ),
        pytest.param(
            8,
            4,
            "contiguous",
            "raw_softplus",
            "omitted",
            torch.float32,
            id="contiguous-raw-softplus",
        ),
        pytest.param(
            8,
            4,
            "framework",
            "raw_safe",
            "omitted",
            torch.float32,
            id="framework-raw-safe",
        ),
        pytest.param(
            8,
            4,
            "contiguous",
            "raw_safe",
            "explicit",
            torch.float32,
            id="contiguous-raw-safe",
        ),
        pytest.param(
            8,
            4,
            "framework",
            "preactivated",
            "omitted",
            torch.bfloat16,
            id="bf16-state-preactivated",
        ),
        pytest.param(
            8,
            4,
            "contiguous",
            "raw_softplus",
            "explicit",
            torch.bfloat16,
            id="bf16-state-softplus",
        ),
        pytest.param(
            8,
            4,
            "framework",
            "raw_safe",
            "explicit",
            torch.bfloat16,
            id="bf16-state-safe",
        ),
    ],
)
def test_kda_target_verify_k128(
    batch, steps, layout, gate_mode, scale_mode, state_dtype
):
    kwargs, inputs = _make_case(
        batch, steps, layout, gate_mode, scale_mode, state_dtype
    )
    expected = _reference(inputs)
    output = kda_target_verify_k128_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)


@pytest.mark.parametrize("qk_scale", [0.0, 1e-7], ids=["zero", "near-zero"])
def test_kda_target_verify_k128_small_norm(qk_scale):
    # Fixed B8/S4/H12/K128/V128; exercise this candidate's epsilon explicitly.
    kwargs, inputs = _make_case(
        8, 4, "framework", "preactivated", "omitted", torch.float32, qk_scale=qk_scale
    )
    expected = _reference(inputs)
    output = kda_target_verify_k128_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)
    if qk_scale == 0.0:
        assert torch.count_nonzero(output).item() == 0


def test_kda_target_verify_k128_graph():
    kwargs, inputs = _make_case(
        8, 4, "framework", "preactivated", "omitted", torch.float32
    )
    expected = _reference(inputs)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    # Compile and allocate before capture, using the same stream as capture.
    with torch.npu.stream(stream):
        for _ in range(3):
            kda_target_verify_k128_npu(**kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output = kda_target_verify_k128_npu(**kwargs)
    torch.npu.synchronize()
    for _ in range(2):
        output.fill_(float("nan"))
        kwargs["intermediate_states_buffer"].fill_(SENTINEL)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        _check_result(output, kwargs, inputs, expected)
