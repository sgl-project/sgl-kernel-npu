"""NPU parity and host-contract tests for ``tree.compressor_tree_project``.

The kernel under test is the production Triton port; the expected values come
from the independent CPU oracle ``tree_reference.compressor_tree_project_ref``
(D5), never from the kernel itself.

Semantics source: vllm-ascend @ 8947f1f9f
  ``csrc/attention/compressor_tree_project/{torch_binding.h,README.md}``
  see ``docs/tree_ops_semantics.md`` §2.

Hosts without ``torch_npu`` / ``triton`` skip the whole module, so this file is
collectible on a CPU-only box (the local Windows box has neither).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from sgl_kernel_npu.tree import compressor_tree_project  # noqa: E402
from tree_fixtures import make_project_case  # noqa: E402
from tree_reference import compressor_tree_project_ref  # noqa: E402

# Wave-1 acceptance tolerances for FP32 node rows (§0 D1: tolerance-based).
RTOL = 2e-3
ATOL = 2e-3

# The kernel tiles the node dimension by BLOCK_M=16; n not a multiple of it
# exercises the masked tail.
BLOCK_M = 16


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _npu_device():
    if not torch.npu.is_available():
        pytest.skip("An Ascend NPU is required")
    return torch.device("npu")


def _reference(case):
    """Run the CPU oracle on a clone so the fixture tensors stay pristine."""
    nodes = case["nodes"].clone()
    compressor_tree_project_ref(
        case["x"],
        case["wkv"],
        case["wgate"],
        nodes,
        case["ape"],
        case["positions"],
        case["cu"],
        case["ratio"],
        case["head_dim"],
    )
    return nodes


def _run(case):
    """Return ``(kernel output on CPU, reference output on CPU)``."""
    expected = _reference(case)
    device = _npu_device()

    nodes = case["nodes"].to(device)
    compressor_tree_project(
        case["x"].to(device),
        case["wkv"].to(device),
        case["wgate"].to(device),
        nodes,
        case["ape"].to(device),
        case["positions"].to(device),
        case["cu"].to(device),
        case["ratio"],
        case["head_dim"],
    )
    torch.npu.synchronize()
    return nodes.cpu(), expected


def _valid_case(**overrides):
    """Small, fast, valid case used by the contract-rejection tests."""
    params = dict(seed=17, ratio=4, head_dim=128, hidden=1024, n=4)
    params.update(overrides)
    return make_project_case(**params)


def _call(case):
    return compressor_tree_project(
        case["x"],
        case["wkv"],
        case["wgate"],
        case["nodes"],
        case["ape"],
        case["positions"],
        case["cu"],
        case["ratio"],
        case["head_dim"],
    )


# ---------------------------------------------------------------------------
# numerical parity against the independent reference
# ---------------------------------------------------------------------------


def test_native_seven_arg_abi_derives_ratio_and_head_dim():
    """The real node_pool caller passes only seven args; APE is the authority."""
    case = make_project_case(seed=23, ratio=4, head_dim=512, hidden=1024, n=5)
    nodes = case["nodes"].to("npu")
    compressor_tree_project(
        case["x"].to("npu"),
        case["wkv"].to("npu"),
        case["wgate"].to("npu"),
        nodes,
        case["ape"].to("npu"),
        case["positions"].to("npu"),
        case["cu"].to("npu"),
    )
    torch.npu.synchronize()

    expected = make_project_case(
        seed=23, ratio=4, head_dim=512, hidden=1024, n=5
    )
    compressor_tree_project_ref(
        expected["x"],
        expected["wkv"],
        expected["wgate"],
        expected["nodes"],
        expected["ape"],
        expected["positions"],
        expected["cu"],
        expected["ratio"],
        expected["head_dim"],
    )
    torch.testing.assert_close(
        nodes.cpu(), expected["nodes"], rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "ratio,head_dim",
    [(4, 300), (8, 128)],
)
def test_native_seven_arg_rejects_unsupported_derived_spec(ratio, head_dim):
    """Derived ratio/head_dim must still fail fast, exactly like the binding."""
    case = make_project_case(
        seed=29, ratio=ratio, head_dim=head_dim, hidden=1024, n=2
    )
    with pytest.raises(ValueError):
        compressor_tree_project(
            case["x"].to("npu"),
            case["wkv"].to("npu"),
            case["wgate"].to("npu"),
            case["nodes"].to("npu"),
            case["ape"].to("npu"),
            case["positions"].to("npu"),
            case["cu"].to("npu"),
        )


@pytest.mark.parametrize("ratio,head_dim", [(4, 128), (4, 512), (128, 512)])
@pytest.mark.parametrize("n", [1, 2, 17])
def test_parity_against_reference(ratio, head_dim, n):
    """Every supported (ratio, head_dim) specialization matches the oracle."""
    case = make_project_case(
        seed=ratio + head_dim + n,
        ratio=ratio,
        head_dim=head_dim,
        hidden=1024,
        n=n,
    )
    actual, expected = _run(case)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n", [1, 5, 17, 31])
def test_tail_rows_not_multiple_of_block_m(n):
    """Masked BLOCK_M tail must not read or write outside rows 1..n."""
    case = make_project_case(seed=200 + n, ratio=4, head_dim=128, hidden=1024, n=n)
    assert n % BLOCK_M != 0
    actual, expected = _run(case)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_row0_sentinel_is_bit_identical():
    """Row 0 is reserved: the kernel must never touch it."""
    case = make_project_case(seed=11, ratio=4, head_dim=128, hidden=1024, n=5)
    sentinel = case["nodes"][0].clone()
    assert bool((sentinel == 7.0).all())

    actual, _ = _run(case)
    assert torch.equal(actual[0], sentinel)


# ---------------------------------------------------------------------------
# APE placement
# ---------------------------------------------------------------------------


def test_ape_applies_only_to_the_score_half():
    """Zero weights + known APE: kv half is exactly 0, score half is APE[r]."""
    ratio, head_dim, n = 4, 128, 4
    half = 2 * head_dim  # c == 2 for ratio 4
    case = make_project_case(seed=13, ratio=ratio, head_dim=head_dim, hidden=1024, n=n)
    case["wkv"].zero_()
    case["wgate"].zero_()
    case["ape"] = (
        torch.arange(ratio * half, dtype=torch.float32).reshape(ratio, half) * 1e-3
    )
    case["positions"] = torch.tensor([3, 0, 2, 1], dtype=torch.int32)

    actual, expected = _run(case)

    torch.testing.assert_close(
        actual[1:, 0, :half],
        torch.zeros((n, half), dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual[1:, 0, half:],
        case["ape"][case["positions"].to(torch.int64)],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# host-contract rejections (validation only; nothing is launched)
# ---------------------------------------------------------------------------


def test_rejects_non_bf16_x():
    case = _valid_case()
    case["x"] = case["x"].to(torch.float32)
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_non_bf16_weights():
    case = _valid_case()
    case["wkv"] = case["wkv"].to(torch.float16)
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_non_contiguous_nodes():
    case = _valid_case()
    width = case["nodes"].shape[2]
    buffer = torch.zeros(case["nodes"].shape[0], 1, 2 * width, dtype=torch.float32)
    nodes = buffer[..., :width]
    assert nodes.shape == case["nodes"].shape
    assert nodes.numel() == case["nodes"].numel()
    assert not nodes.is_contiguous()
    case["nodes"] = nodes
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_non_float32_nodes():
    case = _valid_case()
    case["nodes"] = case["nodes"].to(torch.bfloat16)
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_row_count_mismatch():
    case = _valid_case()
    case["nodes"] = case["nodes"][:-1]  # n rows instead of n + 1
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_cu_not_arange():
    case = _valid_case()
    case["cu"] = case["cu"].clone()
    case["cu"][1] = 3
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_positions_outside_ratio():
    case = _valid_case()
    case["positions"] = case["positions"].clone()
    case["positions"][0] = case["ratio"]
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_unsupported_head_dim():
    case = _valid_case()
    with pytest.raises(ValueError):
        _call(dict(case, head_dim=256))


def test_rejects_unsupported_ratio():
    case = _valid_case()
    with pytest.raises(ValueError):
        _call(dict(case, ratio=8))


def test_rejects_ape_head_dim_mismatch():
    """``head_dim`` must agree with the APE width implied by ``ratio``."""
    case = _valid_case()
    case["ape"] = torch.zeros(
        case["ratio"], 2 * case["head_dim"] * 2, dtype=torch.float32
    )
    with pytest.raises(ValueError):
        _call(case)


def test_rejects_weight_shape_mismatch():
    case = _valid_case()
    case["wgate"] = case["wgate"][:-1]
    with pytest.raises(ValueError):
        _call(case)
