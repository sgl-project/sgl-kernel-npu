"""TorchDynamo / torch.compile tests for split_qkv_rmsnorm_rope custom ops.

CPU tests mock the Triton kernel with the torch_native reference so CI without
Ascend NPU hardware still verifies fullgraph=True tracing, FakeTensor meta
inference, and validation contracts. NPU tests run the real Triton kernel under
npugraph_ex when an Ascend device is available.
"""

import importlib
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

native_impl = importlib.import_module(
    "sgl_kernel_npu.implementation.torch_native.norm.split_qkv_rmsnorm_rope"
)
op_mod = importlib.import_module("sgl_kernel_npu.operators.norm.split_qkv_rmsnorm_rope")


def _has_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def _make_split_qkv_inputs(device: str, dtype: torch.dtype):
    n_tokens, q_hidden, kv_hidden, head_dim = 4, 256, 128, 64
    return {
        "input": torch.randn(
            n_tokens, q_hidden + 2 * kv_hidden, device=device, dtype=dtype
        ),
        "sin": torch.randn(
            n_tokens, 1, 1, head_dim, device=device, dtype=torch.float32
        ),
        "cos": torch.randn(
            n_tokens, 1, 1, head_dim, device=device, dtype=torch.float32
        ),
        "q_hidden_size": q_hidden,
        "kv_hidden_size": kv_hidden,
        "head_dim": head_dim,
        "eps": 1e-6,
        "q_weight": torch.ones(head_dim, device=device, dtype=dtype),
        "k_weight": torch.ones(head_dim, device=device, dtype=dtype),
    }


def _make_split_qkvgate_inputs(device: str, dtype: torch.dtype):
    n_tokens, q_hidden, kv_hidden, head_dim, rope_dim = 4, 256, 128, 64, 32
    return {
        "input": torch.randn(
            n_tokens, 2 * q_hidden + 2 * kv_hidden, device=device, dtype=dtype
        ),
        "sin": torch.randn(
            n_tokens, 1, 1, rope_dim, device=device, dtype=torch.float32
        ),
        "cos": torch.randn(
            n_tokens, 1, 1, rope_dim, device=device, dtype=torch.float32
        ),
        "q_hidden_size": q_hidden,
        "kv_hidden_size": kv_hidden,
        "head_dim": head_dim,
        "rope_dim": rope_dim,
        "eps": 1e-6,
        "q_weight": torch.ones(head_dim, device=device, dtype=dtype),
        "k_weight": torch.ones(head_dim, device=device, dtype=dtype),
    }


def test_split_qkv_rmsnorm_rope_dynamo_fullgraph_cpu():
    """Custom op traces cleanly with fullgraph=True and zero graph breaks on CPU."""
    torch._dynamo.reset()
    inputs = _make_split_qkv_inputs("cpu", torch.float32)

    captured_targets = []

    def inspect_backend(gm, example_inputs):
        captured_targets.extend(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
        return gm.forward

    def fn(x, sin, cos, qw, kw):
        return op_mod.split_qkv_rmsnorm_rope(
            x,
            sin,
            cos,
            inputs["q_hidden_size"],
            inputs["kv_hidden_size"],
            inputs["head_dim"],
            inputs["eps"],
            qw,
            kw,
            impl="triton",
        )

    with mock.patch.object(op_mod, "_triton_fn", native_impl.split_qkv_rmsnorm_rope):
        ref = fn(
            inputs["input"],
            inputs["sin"],
            inputs["cos"],
            inputs["q_weight"],
            inputs["k_weight"],
        )
        compiled = torch.compile(
            fn, fullgraph=True, dynamic=False, backend=inspect_backend
        )
        out = compiled(
            inputs["input"],
            inputs["sin"],
            inputs["cos"],
            inputs["q_weight"],
            inputs["k_weight"],
        )

    assert any("split_qkv_rmsnorm_rope_triton" in t for t in captured_targets)
    for expected, actual in zip(ref, out):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_split_qkvgate_gemma_rmsnorm_rope_dynamo_fullgraph_cpu():
    """Gated Gemma variant traces cleanly with fullgraph=True on CPU."""
    torch._dynamo.reset()
    inputs = _make_split_qkvgate_inputs("cpu", torch.float32)

    captured_targets = []

    def inspect_backend(gm, example_inputs):
        captured_targets.extend(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
        return gm.forward

    def fn(x, sin, cos, qw, kw):
        return op_mod.split_qkvgate_gemma_rmsnorm_rope(
            x,
            sin,
            cos,
            inputs["q_hidden_size"],
            inputs["kv_hidden_size"],
            inputs["head_dim"],
            inputs["rope_dim"],
            inputs["eps"],
            qw,
            kw,
            impl="triton",
        )

    with mock.patch.object(
        op_mod, "_gate_triton_fn", native_impl.split_qkvgate_gemma_rmsnorm_rope
    ):
        ref = fn(
            inputs["input"],
            inputs["sin"],
            inputs["cos"],
            inputs["q_weight"],
            inputs["k_weight"],
        )
        compiled = torch.compile(
            fn, fullgraph=True, dynamic=False, backend=inspect_backend
        )
        out = compiled(
            inputs["input"],
            inputs["sin"],
            inputs["cos"],
            inputs["q_weight"],
            inputs["k_weight"],
        )

    assert any("split_qkvgate_gemma_rmsnorm_rope_triton" in t for t in captured_targets)
    for expected, actual in zip(ref, out):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_inputs_preserve_assertion_error_cpu():
    """Validation runs before custom_op dispatch so AssertionError is preserved."""
    qkv = _make_split_qkv_inputs("cpu", torch.float32)
    qkv["is_neox_style"] = 1
    with pytest.raises(AssertionError):
        op_mod.split_qkv_rmsnorm_rope(**qkv, impl="triton")

    gate = _make_split_qkvgate_inputs("cpu", torch.float32)
    gate["head_dim"] = 64.0
    with pytest.raises(AssertionError):
        op_mod.split_qkvgate_gemma_rmsnorm_rope(**gate, impl="triton")


@pytest.mark.skipif(not _has_npu(), reason="requires an Ascend NPU")
def test_split_qkv_rmsnorm_rope_npugraph_ex_fullgraph():
    """Real Triton kernel matches eager bit-for-bit under npugraph_ex fullgraph=True."""
    import npugraph_ex

    torch._dynamo.reset()
    inputs = _make_split_qkv_inputs("npu", torch.bfloat16)

    def fn(x, sin, cos, qw, kw):
        return op_mod.split_qkv_rmsnorm_rope(
            x,
            sin,
            cos,
            inputs["q_hidden_size"],
            inputs["kv_hidden_size"],
            inputs["head_dim"],
            inputs["eps"],
            qw,
            kw,
            impl="triton",
        )

    ref = fn(
        inputs["input"],
        inputs["sin"],
        inputs["cos"],
        inputs["q_weight"],
        inputs["k_weight"],
    )
    compiled = torch.compile(
        fn,
        fullgraph=True,
        dynamic=False,
        backend=npugraph_ex.get_npu_backend(),
    )
    out = compiled(
        inputs["input"],
        inputs["sin"],
        inputs["cos"],
        inputs["q_weight"],
        inputs["k_weight"],
    )
    for expected, actual in zip(ref, out):
        assert (expected - actual).abs().max().item() == 0.0


@pytest.mark.skipif(not _has_npu(), reason="requires an Ascend NPU")
def test_split_qkvgate_gemma_rmsnorm_rope_npugraph_ex_fullgraph():
    """Real gated Gemma Triton kernel matches eager bit-for-bit under npugraph_ex."""
    import npugraph_ex

    torch._dynamo.reset()
    inputs = _make_split_qkvgate_inputs("npu", torch.bfloat16)

    def fn(x, sin, cos, qw, kw):
        return op_mod.split_qkvgate_gemma_rmsnorm_rope(
            x,
            sin,
            cos,
            inputs["q_hidden_size"],
            inputs["kv_hidden_size"],
            inputs["head_dim"],
            inputs["rope_dim"],
            inputs["eps"],
            qw,
            kw,
            impl="triton",
        )

    ref = fn(
        inputs["input"],
        inputs["sin"],
        inputs["cos"],
        inputs["q_weight"],
        inputs["k_weight"],
    )
    compiled = torch.compile(
        fn,
        fullgraph=True,
        dynamic=False,
        backend=npugraph_ex.get_npu_backend(),
    )
    out = compiled(
        inputs["input"],
        inputs["sin"],
        inputs["cos"],
        inputs["q_weight"],
        inputs["k_weight"],
    )
    for expected, actual in zip(ref, out):
        assert (expected - actual).abs().max().item() == 0.0
