import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance


def generate_test_data(dtype, rows, gate, bias, contiguous=False):
    torch.npu.manual_seed(42)
    # Keep padded-row coverage while allowing contiguous model-shaped inputs.
    storage_width = 128 if contiguous else 256
    x = torch.randn(rows, storage_width, dtype=dtype)[:, :128]
    weight = (0.5 + torch.rand(128)).to(dtype)
    bias_value = (0.1 * torch.randn(128)).to(dtype) if bias else None
    z = torch.randn(rows, storage_width, dtype=dtype)[:, :128] if gate else None
    return x, weight, bias_value, z


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "rows,size,rms,gate,before,bias,eps,contiguous",
    [
        pytest.param(1, None, False, False, True, False, 1e-5, False, id="layernorm"),
        pytest.param(
            17,
            32,
            False,
            True,
            False,
            True,
            1e-5,
            False,
            id="layernorm-pre-gate-groups",
        ),
        pytest.param(
            17,
            64,
            False,
            True,
            True,
            True,
            1e-5,
            False,
            id="layernorm-post-gate-groups",
        ),
        pytest.param(3, 64, True, False, True, True, 1e-5, False, id="rmsnorm-bias"),
        pytest.param(
            3, 32, True, True, False, False, 1e-5, False, id="rmsnorm-pre-gate"
        ),
        pytest.param(
            3, 32, True, True, True, True, 1e-5, False, id="rmsnorm-post-gate"
        ),
        # Qwen3.5-35B-A3B flattens [T, 32 / TP, 128] before gated RMSNorm.
        # It uses eps=1e-6, no bias/groups and norm_before_gate=True.
        pytest.param(
            32,
            None,
            True,
            True,
            True,
            False,
            1e-6,
            True,
            id="qwen3-5-35b-a3b-decode-t1-tp1",
        ),
        pytest.param(
            128,
            None,
            True,
            True,
            True,
            False,
            1e-6,
            True,
            id="qwen3-5-35b-a3b-decode-t16-tp4",
        ),
        pytest.param(
            8192,
            None,
            True,
            True,
            True,
            False,
            1e-6,
            True,
            id="qwen3-5-35b-a3b-prefill-t256-tp1",
        ),
    ],
)
# Compare gating order, norm kind and group-major FP32 statistics with an independent output.
def test_layer_norm_fwd(rows, size, rms, gate, before, bias, eps, contiguous, dtype):
    from sgl_kernel_npu.operators.norm import layer_norm_fwd

    x, weight, bias_value, z = generate_test_data(dtype, rows, gate, bias, contiguous)
    executor = check_accuracy(
        layer_norm_fwd,
        args=(x, weight, bias_value, eps),
        kwargs={
            "z": z,
            "group_size": size,
            "norm_before_gate": before,
            "is_rms_norm": rms,
            "activation": "silu",
        },
    )
    out, mean, rstd = executor()
    groups = 128 // (128 if size is None else size)
    assert out.shape == x.shape and out.dtype == dtype
    assert rstd.shape == (rows * groups,) and rstd.dtype == torch.float32
    if rms:
        assert mean is None
    else:
        assert mean.shape == rstd.shape and mean.dtype == torch.float32
    check_performance(executor)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
# Use separate preallocated outputs so accuracy cannot pass by comparing one shared buffer.
def test_layer_norm_preallocated_output(dtype):
    from sgl_kernel_npu.operators.norm import layer_norm_fwd

    x, weight, bias, z = generate_test_data(dtype, 3, True, True)
    x_before = x.clone()
    dut_storage = torch.full((3, 256), -17.0, dtype=dtype)
    ref_storage = torch.full_like(dut_storage, -17.0)
    dut_out, ref_out = dut_storage[:, :128], ref_storage[:, :128]
    kwargs = {"z": z, "group_size": 32, "is_rms_norm": True}
    expected = layer_norm_fwd(
        x, weight, bias, 1e-5, out=ref_out, impl="torch_native", **kwargs
    )
    actual = layer_norm_fwd(x, weight, bias, 1e-5, out=dut_out, impl="triton", **kwargs)
    assert actual[0] is dut_out and expected[0] is ref_out
    assert_accuracy(actual, expected)
    torch.testing.assert_close(
        dut_storage[:, 128:],
        torch.full_like(dut_storage[:, 128:], -17.0),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        ref_storage[:, 128:],
        torch.full_like(ref_storage[:, 128:], -17.0),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)
