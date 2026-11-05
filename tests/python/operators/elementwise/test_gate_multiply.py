"""Public gate_multiply operator: BF16 multiply/cast fusion for GDN.

GDN's sigmoid-after-normalization path multiplies a BF16 normalized activation
by an FP32 sigmoid gate and casts the FP32 product back to BF16. The public
operator fuses that multiply and cast. The Triton backend reproduces the plain
PyTorch expression bitwise, so accuracy is checked with zero tolerance against
the independent torch_native reference, and the queued, graph-replay and guard
flows that check_accuracy cannot express are covered by dedicated tests.

The model cases use GDN value-head widths derived from model configurations,
with synthetic token counts and TP1/TP2 sharding. Per-token flattened views and
the [tokens * local value heads, 128] view used around gated normalization are
both covered. Qwen3.6-35B-A3B and Qwen3.8-27B normally use swish-gated
normalization, so their cases cover compatible shapes only, not an existing
integration of this sigmoid-gate operator.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from sgl_kernel_npu.implementation.triton.elementwise.gate_multiply import (
    FUSION_MIN_ELEMENTS,
)

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Both backends must match bit for bit, not within a BF16 tolerance.
EXACT = {"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)}

# Model-derived widths at or above the fusion threshold, with synthetic token
# counts. Only contiguous cases that enter the fused kernel are profiled.
perf_shape = [
    (4096, 4096),
    (16384, 4096),
    (4096, 6144),
    (16384, 6144),
    (16384, 2048),
    (16384, 3072),
]


def generate_data(shape, mode, seed=100326):
    """Create BF16 y and FP32 gate on the NPU device supplied by conftest.

    shape is the common tensor shape. mode selects the value pattern:
    normal   random signed y and a saturated sigmoid gate in (0, 1);
    special  signed zeros, infinities and NaNs on both sides of the product.
    The returned tensors follow the operator contract and are independent.
    """
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    if mode == "normal":
        y = torch.randn(shape, dtype=torch.bfloat16)
        gate = torch.sigmoid(torch.randn(shape) * 20)
        return y, gate
    if mode == "special":
        y = torch.ones(shape, dtype=torch.bfloat16)
        gate = torch.ones(shape, dtype=torch.float32)
        flat_y = y.view(-1)
        flat_gate = gate.view(-1)
        count = min(8, flat_y.numel())
        specials_y = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0, 0.5],
            dtype=torch.bfloat16,
        )
        specials_gate = torch.tensor(
            [1.0, 1.0, 0.0, -1.0, 1.0, float("nan"), float("inf"), 0.0],
            dtype=torch.float32,
        )
        flat_y[:count] = specials_y[:count]
        flat_gate[:count] = specials_gate[:count]
        return y, gate
    raise ValueError(f"Unknown data mode: {mode}")


@pytest.mark.parametrize("mode", ["normal", "special"])
@pytest.mark.parametrize(
    "shape",
    [
        # Below the fusion threshold: the Triton backend uses the PyTorch path.
        pytest.param((1,), id="edge-single-element"),
        pytest.param((4097,), id="edge-small-1d"),
        pytest.param((FUSION_MIN_ELEMENTS - 1,), id="edge-just-below-threshold"),
        # At and above the threshold: the fused kernel engages.
        pytest.param((FUSION_MIN_ELEMENTS,), id="threshold-exact"),
        pytest.param((FUSION_MIN_ELEMENTS + 1,), id="threshold-plus-one"),
        # Width = linear_num_value_heads * linear_value_head_dim / attention TP.
        # https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json
        # https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
        # https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json
        # The model-derived cases cover both fallback and fused sizes.
        # Qwen3.6-35B-A3B TP1: 32 value heads * 128 = 4096 per token.
        pytest.param((4096, 4096), id="gdn-4096-tokens"),
        pytest.param((16384, 4096), id="gdn-16384-tokens"),
        # Qwen3.8-27B and Qwen3.8-Flash-Next TP1: 48 * 128 = 6144.
        pytest.param((4096, 6144), id="gdn-tp1-width6144-4096-tokens"),
        pytest.param((16384, 6144), id="gdn-tp1-width6144-16384-tokens"),
        # Qwen3.6-35B-A3B TP2: 16 * 128 = 2048 per rank.
        pytest.param((4096, 2048), id="gdn-tp2-width2048-4096-tokens"),
        pytest.param((16384, 2048), id="gdn-tp2-width2048-16384-tokens"),
        # Qwen3.8-27B and Qwen3.8-Flash-Next TP2: 24 * 128 = 3072.
        pytest.param((4096, 3072), id="gdn-tp2-width3072-4096-tokens"),
        pytest.param((16384, 3072), id="gdn-tp2-width3072-16384-tokens"),
        # Head-flattened views around gated normalization, for the same models.
        pytest.param((4096 * 32, 128), id="gdn-tp1-heads32-4096-tokens"),
        pytest.param((4096 * 48, 128), id="gdn-tp1-heads48-4096-tokens"),
        pytest.param((4096 * 24, 128), id="gdn-tp2-heads24-4096-tokens"),
    ],
)
def test_gate_multiply(shape, mode):
    """Both backends agree bitwise across the fusion threshold and specials."""
    from sgl_kernel_npu.operators.elementwise import gate_multiply

    y, gate = generate_data(shape, mode)
    # NaN products compare unequal under assert_close, so skip the exact flow
    # for the special pattern and verify NaN placement separately below.
    if mode == "special":
        from sgl_kernel_npu.operators.elementwise import gate_multiply as op

        result = op(y, gate, impl="triton")
        reference = op(y, gate, impl="torch_native")
        assert torch.equal(torch.isnan(result), torch.isnan(reference))
        finite = ~torch.isnan(reference)
        assert torch.equal(
            result[finite].view(torch.int16), reference[finite].view(torch.int16)
        )
        return
    executor = check_accuracy(gate_multiply, args=(y, gate), standard=EXACT)
    # An accuracy failure raises before performance collection.
    check_performance(executor, enable_perf=shape in perf_shape and mode == "normal")


def test_gate_multiply_strided_fallback():
    """Strided inputs below the kernel's contiguity requirement stay exact."""
    from sgl_kernel_npu.operators.elementwise import gate_multiply

    y = torch.randn(8192, 4098, device="npu", dtype=torch.bfloat16)[:, ::2]
    gate = torch.sigmoid(torch.randn(8192, 4098, device="npu"))[:, ::2]
    result = gate_multiply(y, gate, impl="triton")
    reference = gate_multiply(y, gate, impl="torch_native")
    assert torch.equal(result, reference)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((FUSION_MIN_ELEMENTS,), id="threshold-exact"),
        pytest.param((4097,), id="below-threshold"),
    ],
)
def test_gate_multiply_queued_temporary_inputs(shape, impl_pair):
    """Queued launches must not read arguments the caller already dropped.

    Each call owns its tensor arguments until the device executes it. This
    flow is not expressible through check_accuracy: the arguments are clones
    released as each call returns, and the results are compared only after one
    synchronization.
    """
    from sgl_kernel_npu.operators.elementwise import gate_multiply

    impl = impl_pair[0]
    y, gate = generate_data(shape, "normal")
    reference = gate_multiply(y, gate, impl=impl).clone()
    torch.npu.synchronize()
    outputs = [gate_multiply(y.clone(), gate.clone(), impl=impl) for _ in range(3)]
    torch.npu.synchronize()
    for out in outputs:
        assert torch.equal(out, reference)


def test_gate_multiply_graph_replay(impl_pair):
    """Capture a warmed fused shape, then replay it over rewritten inputs.

    Graph capture needs its own flow: the comparison runs after replaying the
    captured launch over mutated device buffers, which check_accuracy's
    repeated direct calls cannot express.
    """
    from sgl_kernel_npu.operators.elementwise import gate_multiply

    impl, impl_ref = impl_pair
    y, gate = generate_data((FUSION_MIN_ELEMENTS,), "normal")
    for _ in range(3):
        gate_multiply(y, gate, impl=impl)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = gate_multiply(y, gate, impl=impl)
    for _ in range(5):
        y.normal_()
        gate.uniform_()
        graph.replay()
        torch.npu.synchronize()
        reference = gate_multiply(y, gate, impl=impl_ref)
        assert torch.equal(out, reference)


@pytest.mark.parametrize(
    "invalid, message",
    [
        pytest.param("y-float32", "matching BF16 y", id="y-not-bf16"),
        pytest.param("gate-bf16", "matching BF16 y", id="gate-not-fp32"),
        pytest.param("shape-mismatch", "matching BF16 y", id="shape-mismatch"),
        pytest.param("device-cpu", "same NPU", id="device-cpu"),
        pytest.param("requires-grad", "inference only", id="requires-grad"),
    ],
)
def test_gate_multiply_rejects_invalid_arguments(invalid, message):
    """Every contract violation raises ValueError for its own reason.

    The expected message is asserted so a case cannot pass by raising
    ValueError somewhere else.
    """
    from sgl_kernel_npu.operators.elementwise import gate_multiply

    y = torch.ones(8, device="npu", dtype=torch.bfloat16)
    gate = torch.ones(8, device="npu", dtype=torch.float32)
    if invalid == "y-float32":
        y = y.float()
    elif invalid == "gate-bf16":
        gate = gate.bfloat16()
    elif invalid == "shape-mismatch":
        gate = gate[:4]
    elif invalid == "device-cpu":
        y, gate = y.cpu(), gate.cpu()
    elif invalid == "requires-grad":
        y = y.requires_grad_()
    else:
        raise AssertionError(f"Unhandled invalid case: {invalid}")
    with pytest.raises(ValueError, match=message):
        gate_multiply(y, gate)
