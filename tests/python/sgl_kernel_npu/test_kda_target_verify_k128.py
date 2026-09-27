"""Accuracy coverage for the K=V=128 split-K target verify dispatch.

The CPU reference defines this fast path's ``rsqrt(sum(x*x) + 1e-12)``
normalization, including its behavior on zero and near-zero inputs.
"""

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  Register NPU before importing the Triton module.
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")

HEADS = 12
KEY_DIM = 128
VALUE_DIM = 128
SCALE = KEY_DIM**-0.5
SENTINEL = -3.125


def _make_case(
    batch,
    steps,
    layout,
    gate_mode,
    scale_mode,
    state_dtype,
    qk_scale=1.0,
    heads=(HEADS, HEADS, HEADS),
    negative_initial=False,
    negative_snapshot=False,
    index_dtype=torch.int32,
    infer_gates=False,
):
    generator = torch.Generator(device="cpu").manual_seed(
        20260927 + batch * 100 + steps
    )
    tokens = batch * steps
    h_q, h_k, h_v = heads

    def normal(shape, scale=1.0):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    # The split preserves the packed projection's token stride, as in SGLang.
    packed_cpu = normal((tokens, (h_q + h_k) * KEY_DIM + h_v * VALUE_DIM))
    packed_cpu[:, : (h_q + h_k) * KEY_DIM] *= qk_scale
    packed_cpu = packed_cpu.to(torch.bfloat16)
    packed = packed_cpu.to("npu")

    def unpack(tensor):
        q, k, v = tensor.split((h_q * KEY_DIM, h_k * KEY_DIM, h_v * VALUE_DIM), dim=-1)
        return (
            q.reshape(1, tokens, h_q, KEY_DIM),
            k.reshape(1, tokens, h_k, KEY_DIM),
            v.reshape(1, tokens, h_v, VALUE_DIM),
        )

    q_cpu, k_cpu, v_cpu = unpack(packed_cpu)
    q, k, v = unpack(packed)
    a_log_cpu = normal((1, 1, h_k, 1), 0.1) - 1.0
    dt_bias_cpu = normal((h_k, KEY_DIM), 0.1)
    raw_a_cpu = normal((tokens, h_k, KEY_DIM), 0.2)
    raw_b_cpu = normal((tokens, h_v), 0.5)
    if gate_mode == "preactivated":
        # The public API accepts externally activated bounded gates.
        a_cpu = -5.0 * torch.sigmoid(
            a_log_cpu.reshape(h_k, 1).exp() * (raw_a_cpu + dt_bias_cpu)
        )
        b_cpu = raw_b_cpu.sigmoid()
        a = a_cpu.to("npu").unsqueeze(0)
        b = b_cpu.to("npu").unsqueeze(0)
    else:
        assert gate_mode == "raw_softplus"
        a_cpu, b_cpu = raw_a_cpu, raw_b_cpu
        packed_gate = torch.cat((raw_a_cpu.flatten(1), raw_b_cpu), dim=-1).to("npu")
        a_flat, b = packed_gate.split((h_k * KEY_DIM, h_v), dim=-1)
        a = a_flat.reshape(1, tokens, h_k, KEY_DIM)
        b = b.unsqueeze(0)
        if infer_gates:
            # Omitting both leading singletons selects raw-gate mode.
            a, b = a.squeeze(0), b.squeeze(0)

    initial_cpu = normal((batch + 2, h_v, VALUE_DIM, KEY_DIM), 0.02).to(state_dtype)
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
    if negative_initial:
        initial_indices_cpu[0] = -1
    if negative_snapshot:
        snapshot_indices_cpu[0] = -1
    snapshots = torch.full(
        (batch + 2, steps, h_v, VALUE_DIM, KEY_DIM),
        SENTINEL,
        dtype=state_dtype,
        device="npu",
    )
    kwargs = dict(
        A_log=a_log_cpu.to("npu"),
        dt_bias=dt_bias_cpu.to("npu"),
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial,
        initial_state_indices=initial_indices_cpu.to(device="npu", dtype=index_dtype),
        intermediate_states_buffer=snapshots,
        intermediate_state_indices=snapshot_indices_cpu.to(
            device="npu", dtype=index_dtype
        ),
        cache_steps=steps,
    )
    if not infer_gates:
        kwargs["gates_are_preactivated"] = gate_mode == "preactivated"
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
        A_log=a_log_cpu.reshape(h_k),
        dt_bias=dt_bias_cpu,
        initial=initial_cpu,
        initial_indices=initial_indices_cpu,
        snapshot_indices=snapshot_indices_cpu,
        batch=batch,
        steps=steps,
        gate_mode=gate_mode,
        heads=heads,
    )
    return kwargs, reference_inputs


def _reference(inputs):
    """CPU FP32 recurrence; intermediates stay FP32 even for BF16 state storage."""
    batch, steps = inputs["batch"], inputs["steps"]
    h_q, h_k, h_v = inputs["heads"]
    q = inputs["q"] * torch.rsqrt(inputs["q"].square().sum(-1, keepdim=True) + 1e-12)
    k = inputs["k"] * torch.rsqrt(inputs["k"].square().sum(-1, keepdim=True) + 1e-12)
    q = q.repeat_interleave(h_v // h_q, dim=1) * SCALE
    k = k.repeat_interleave(h_v // h_k, dim=1)
    if inputs["gate_mode"] == "preactivated":
        log_decay, beta = inputs["a"], inputs["b"]
    else:
        gate_input = inputs["a"] + inputs["dt_bias"]
        exp_a = inputs["A_log"].exp().unsqueeze(-1)
        log_decay = -exp_a * F.softplus(gate_input)
        beta = inputs["b"].sigmoid()
    log_decay = log_decay.repeat_interleave(h_v // h_k, dim=1)

    output = torch.empty((1, batch * steps, h_v, VALUE_DIM), dtype=torch.float32)
    snapshots = torch.empty(
        (batch, steps, h_v, VALUE_DIM, KEY_DIM), dtype=torch.float32
    )
    for request in range(batch):
        initial_index = int(inputs["initial_indices"][request])
        if initial_index < 0:
            state = torch.zeros((h_v, VALUE_DIM, KEY_DIM), dtype=torch.float32)
        else:
            state = inputs["initial"][initial_index].float().clone()
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
    valid_requests = inputs["snapshot_indices"] >= 0
    written_slots = inputs["snapshot_indices"][valid_requests]
    actual_states = actual_pool[written_slots]
    for step in range(inputs["steps"]):
        _assert_close(actual_states[:, step], expected_states[valid_requests, step])
    # State source is read-only; every unwritten scratch slot stays untouched,
    # including a slot whose request uses the negative snapshot sentinel.
    torch.testing.assert_close(
        kwargs["initial_state_source"].detach().cpu(), inputs["initial"], rtol=0, atol=0
    )
    unwritten_slots = torch.ones(actual_pool.shape[0], dtype=torch.bool)
    unwritten_slots[written_slots] = False
    assert torch.all(actual_pool[unwritten_slots] == SENTINEL)


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
    ],
)
def test_kda_target_verify_k128(
    batch, steps, layout, gate_mode, scale_mode, state_dtype
):
    kwargs, inputs = _make_case(
        batch, steps, layout, gate_mode, scale_mode, state_dtype
    )
    expected = _reference(inputs)
    output = kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)


@pytest.mark.parametrize("qk_scale", [0.0, 1e-7], ids=["zero", "near-zero"])
def test_kda_target_verify_k128_small_norm(qk_scale):
    # Fixed B8/S4/H12/K128/V128; exercise the selected fast-path epsilon.
    kwargs, inputs = _make_case(
        8, 4, "framework", "preactivated", "omitted", torch.float32, qk_scale=qk_scale
    )
    expected = _reference(inputs)
    output = kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)
    if qk_scale == 0.0:
        assert torch.count_nonzero(output).item() == 0


def test_kda_target_verify_k128_grouped_heads():
    # Preserve PR #802's B4/S8/Hq=Hk=4/Hv=16/K128/V128 grouped-head case.
    kwargs, inputs = _make_case(
        4,
        8,
        "framework",
        "preactivated",
        "explicit",
        torch.bfloat16,
        heads=(4, 4, 16),
    )
    expected = _reference(inputs)
    output = kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)


@pytest.mark.parametrize(
    "negative_initial,negative_snapshot",
    [(True, False), (False, True), (True, True)],
    ids=["zero-initial-state", "skip-snapshot", "both-negative"],
)
def test_kda_target_verify_k128_negative_indices(negative_initial, negative_snapshot):
    kwargs, inputs = _make_case(
        8,
        4,
        "framework",
        "preactivated",
        "omitted",
        torch.float32,
        negative_initial=negative_initial,
        negative_snapshot=negative_snapshot,
    )
    expected = _reference(inputs)
    output = kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)


def test_kda_target_verify_k128_int64_indices():
    kwargs, inputs = _make_case(
        8,
        4,
        "framework",
        "preactivated",
        "omitted",
        torch.float32,
        index_dtype=torch.int64,
    )
    expected = _reference(inputs)
    output = kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)


@pytest.mark.parametrize("gate_mode", ["preactivated", "raw_softplus"])
def test_kda_target_verify_k128_inferred_gate_mode(gate_mode):
    kwargs, inputs = _make_case(
        8,
        4,
        "framework",
        gate_mode,
        "omitted",
        torch.float32,
        infer_gates=True,
    )
    assert "gates_are_preactivated" not in kwargs
    if gate_mode == "preactivated":
        assert kwargs["a"].ndim == 4 and kwargs["b"].ndim == 3
    else:
        assert kwargs["a"].ndim == 3 and kwargs["b"].ndim == 2
    expected = _reference(inputs)
    output = kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    _check_result(output, kwargs, inputs, expected)


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
            kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output = kda_target_verify_npu(**kwargs)
    torch.npu.synchronize()
    for _ in range(2):
        output.fill_(float("nan"))
        kwargs["intermediate_states_buffer"].fill_(SENTINEL)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        _check_result(output, kwargs, inputs, expected)
