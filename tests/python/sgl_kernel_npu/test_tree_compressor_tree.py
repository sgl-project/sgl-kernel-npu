"""NPU parity tests for ``sgl_kernel_npu.tree.compressor_tree``.

Semantics: ``docs/tree_ops_semantics.md`` §3 (source pin vllm-ascend
``8947f1f9f`` ``csrc/attention/compressor_tree``).  The independent CPU oracle is
``tree_reference.compressor_tree_ref``; the deterministic fixtures are
``tree_fixtures.make_compressor_case``.

The whole module skips cleanly on a host without ``torch_npu`` or Triton, so the
CPU-only suite stays runnable.
"""

import torch

import pytest

pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from sgl_kernel_npu.tree.compressor_tree import compressor_tree
from tree_fixtures import make_compressor_case
from tree_reference import compressor_tree_ref

RTOL = 1e-2
ATOL = 1e-2
DEVICE = "npu"

# (ratio, head_dim): the four constexpr specializations of the op (D6).
SPECIALIZATIONS = [(4, 128), (4, 512), (128, 128), (128, 512)]
# (n, budget, depth); n == batch * (budget + 1) with the root included.
TREE_SHAPES = [(1, 0, 0), (5, 4, 2), (17, 16, 4)]
# (page_pad, cos_pad) pairs: padded state rows and cos/sin rows.
PAD_PAIRS = [(0, 0), (8, 0), (64, 128), (0, 128)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _emitting_prefix(ratio, depth):
    """Prefix that puts the deepest node on an emit boundary (``pos + 1 == ratio``)."""
    return max(0, int(ratio) - 1 - int(depth))


def _device_args(case):
    """Move a fixture case to the NPU, keeping the padded state/cos/sin strides.

    ``Tensor.to(device)`` does not preserve strides for non-dense views, so the
    padded backing buffers are moved first and re-sliced on the device.
    """
    args = dict(case["args"])
    buffers = case.get("padding_views") or []
    if buffers:
        moved = [buffer.to(DEVICE) for buffer in buffers]
        args["state"] = moved[0][..., : args["state"].shape[2]]
        args["cos"] = moved[1][:, : args["cos"].shape[1]]
        args["sin"] = moved[2][:, : args["sin"].shape[1]]
    for name, value in list(args.items()):
        if not torch.is_tensor(value):
            continue
        if buffers and name in ("state", "cos", "sin"):
            continue
        args[name] = value.to(DEVICE)
    return args


def _launch(args):
    """Call the kernel, mapping the fixture's ``eps`` to the public ``epsilon``."""
    kwargs = dict(args)
    epsilon = kwargs.pop("eps")
    return compressor_tree(**kwargs, epsilon=epsilon)


def _run(case):
    out = _launch(_device_args(case))
    torch.npu.synchronize()
    return out


def _reference(case):
    return compressor_tree_ref(**case["args"])


def _assert_close(out, ref):
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == tuple(ref.shape)
    torch.testing.assert_close(out.float().cpu(), ref.float(), rtol=RTOL, atol=ATOL)


def _emit_mask(case):
    """Rows whose ``(pos + 1) % ratio`` is zero, i.e. the rows that carry data."""
    ratio = int(case["args"]["ratio"])
    positions = case["spec"].positions.to(torch.int64)
    return ((positions + 1) % ratio) == 0


def _vary_norm(case):
    """Swap the uniform weight for a deterministic non-uniform positive one.

    ``make_compressor_case`` can only produce a constant ``norm``, which would
    hide a wrong weight index; replacing it keeps both the oracle and the kernel
    on the same tensor.
    """
    size = case["args"]["norm"].numel()
    generator = torch.Generator().manual_seed(7 + size)
    case["args"]["norm"] = (
        torch.rand(size, generator=generator, dtype=torch.float32) + 0.5
    )
    return case


def _case(**overrides):
    """A ratio-4 case with a mixed emitting/non-emitting tree, plus overrides."""
    kwargs = dict(
        ratio=4,
        head_dim=128,
        batch=1,
        budget=4,
        depth=2,
        prefixes=[1],
        rows=4,
    )
    kwargs.update(overrides)
    return _vary_norm(make_compressor_case(**kwargs))


def _base_args(**overrides):
    args = dict(_case()["args"])
    args.update(overrides)
    return args


# ---------------------------------------------------------------------------
# numerical parity: ratio x head_dim x tree shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,head_dim", SPECIALIZATIONS)
@pytest.mark.parametrize("n,budget,depth", TREE_SHAPES)
def test_accepts_real_caller_4d_nodes():
    """node_pool.py allocates nodes as [n+1, 1, 1, 2*c*d]."""
    case = _case()
    expected = _reference(case)
    args = _device_args(case)
    n = int(args["nodes"].size(0)) - 1
    args["nodes"] = args["nodes"].view(n + 1, 1, 1, -1)
    _assert_close(_launch(args), expected)


def test_matches_reference(ratio, head_dim, n, budget, depth):
    case = _case(
        ratio=ratio,
        head_dim=head_dim,
        batch=1,
        budget=budget,
        depth=depth,
        prefixes=[_emitting_prefix(ratio, depth)],
    )
    assert case["n"] == n

    out = _run(case)
    ref = _reference(case)
    _assert_close(out, ref)
    assert _emit_mask(case).any(), "fixture must contain at least one emitting node"


@pytest.mark.parametrize("ratio,head_dim", SPECIALIZATIONS)
@pytest.mark.parametrize("page_pad,cos_pad", PAD_PAIRS)
def test_padded_row_strides(ratio, head_dim, page_pad, cos_pad):
    """Padded state rows and cos/sin rows must be honoured through the strides."""
    rope_dim = 64
    case = _case(
        ratio=ratio,
        head_dim=head_dim,
        prefixes=[_emitting_prefix(ratio, 2)],
        page_pad=page_pad,
        cos_pad=cos_pad,
    )
    row_width = case["args"]["state"].shape[2]

    args = _device_args(case)
    if page_pad:
        assert args["state"].stride(1) == row_width + page_pad
    if cos_pad:
        assert args["cos"].stride(0) == rope_dim + cos_pad

    _assert_close(_launch(args), _reference(case))


def test_non_emitting_rows_are_exact_zero():
    case = _case(
        batch=2,
        budget=4,
        depth=2,
        prefixes=[_emitting_prefix(4, 2), _emitting_prefix(4, 0)],
    )
    out = _run(case)
    _assert_close(out, _reference(case))

    mask = _emit_mask(case)
    assert mask.any() and not mask.all()
    zero_rows = out[~mask.to(out.device)]
    assert torch.count_nonzero(zero_rows).item() == 0


# ---------------------------------------------------------------------------
# C4 previous/current window halves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [128, 512])
def test_c4_half_selection(head_dim):
    """``poison_halves`` makes the two projected halves differ in sign.

    With ``prefix=1`` only the second half reaches the window, so the pre-RoPE
    tail of every emitting row must be uniformly negative; selecting the wrong
    half (``w % c`` vs ``w // c``) flips it and breaks parity with the oracle.
    """
    case = _case(head_dim=head_dim, poison_halves=True)
    out = _run(case)
    ref = _reference(case)
    _assert_close(out, ref)

    mask = _emit_mask(case).to(out.device)
    begin = head_dim - int(case["args"]["rope_dim"])
    tail = ref.to(out.device)[mask][:, :begin].float()
    assert (tail < 0).all(), "the poisoned lower half should dominate the window"


# ---------------------------------------------------------------------------
# masked slots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,head_dim", [(4, 128), (128, 512)])
def test_invalid_ancestors_and_bad_pages(ratio, head_dim):
    for extra in ({"invalid_ancestor": True}, {"bad_page": True}):
        case = _case(
            ratio=ratio,
            head_dim=head_dim,
            prefixes=[_emitting_prefix(ratio, 2)],
            **extra,
        )
        _assert_close(_run(case), _reference(case))


def test_all_slots_masked_emits_zero_row():
    """A node whose whole window is unreachable must emit an exact zero row."""
    case = _case()
    assert torch.count_nonzero(_reference(case)).item() > 0, "the case must be live"

    args = dict(case["args"])
    args["ancestors"] = torch.full_like(args["ancestors"], -1)
    args["table"] = torch.full_like(args["table"], -1)
    masked = dict(case, args=args)

    assert _emit_mask(masked).any()
    out = _run(masked)
    assert torch.count_nonzero(out.float()).item() == 0
    _assert_close(out, _reference(masked))


# ---------------------------------------------------------------------------
# RoPE position and layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,head_dim,prefix", [(4, 512, 5), (128, 512, 125)])
def test_rope_position_lags_by_ratio(ratio, head_dim, prefix):
    """``ropePos = pos + 1 - ratio``, not ``pos``.

    The lag is ``ratio - 1`` rows, so C128 reads a cos/sin row 127 rows away
    from ``pos``: using ``pos`` instead cannot stay inside the tolerance.
    """
    case = _case(ratio=ratio, head_dim=head_dim, prefixes=[prefix])
    spec = case["spec"]
    emit = _emit_mask(case)
    positions = spec.positions.to(torch.int64)[emit]
    rope_positions = positions + 1 - ratio
    assert (rope_positions < positions).all() and (rope_positions >= 0).all()
    assert (positions - rope_positions == ratio - 1).all()

    cos = case["args"]["cos"]
    for pos, rope_pos in zip(positions.tolist(), rope_positions.tolist()):
        assert not torch.equal(cos[pos], cos[rope_pos]), "cos rows must differ"

    _assert_close(_run(case), _reference(case))


@pytest.mark.parametrize("head_dim", [128, 512])
def test_only_the_rope_tail_is_rotated(head_dim):
    """cos/sin columns past ``rope_dim`` and the state row padding are never read."""
    rope_dim = 64
    case = _case(head_dim=head_dim, prefixes=[5], page_pad=8, cos_pad=128)
    state_buffer, cos_buffer, sin_buffer = case["padding_views"]
    row_width = case["args"]["state"].shape[2]

    poison = float("nan")
    state_buffer[..., row_width:] = poison
    cos_buffer[:, rope_dim:] = poison
    sin_buffer[:, rope_dim:] = poison

    out = _run(case)
    assert torch.isfinite(out.float()).all()
    _assert_close(out, _reference(case))


# ---------------------------------------------------------------------------
# RMSNorm epsilon placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-6, 1.0])
def test_epsilon_inside_sqrt(eps):
    case = _case(eps=eps)
    _assert_close(_run(case), _reference(case))


def test_epsilon_placement_is_observable():
    """Inside vs outside the sqrt must differ well beyond the tolerance."""
    small = _reference(_case(eps=1e-6))
    large = _reference(_case(eps=1.0))
    mask = _emit_mask(_case())
    diff = (small.float() - large.float()).abs()[mask].max()
    assert diff > 10 * ATOL


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------


def test_output_contract():
    case = _case()
    out = _run(case)
    d = int(case["args"]["norm"].numel())
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (case["n"], d)
    assert out.device.type == "npu"


# ---------------------------------------------------------------------------
# host contract violations (validated before any launch; CPU tensors are enough)
# ---------------------------------------------------------------------------


def test_rejects_unsupported_ratio():
    with pytest.raises(ValueError):
        _launch(_base_args(ratio=3))


def test_rejects_unsupported_rope_dim():
    with pytest.raises(ValueError):
        _launch(_base_args(rope_dim=32))


def test_rejects_non_positive_epsilon():
    args = _base_args()
    args.pop("eps")
    with pytest.raises(ValueError):
        compressor_tree(**args, epsilon=0.0)


def test_rejects_wrong_prefix_dtype():
    args = _base_args()
    args["prefix"] = args["prefix"].to(torch.int64)
    with pytest.raises(ValueError):
        _launch(args)


def test_rejects_non_contiguous_prefix():
    args = _base_args()
    args["prefix"] = args["prefix"].repeat(2)[::2]
    with pytest.raises(ValueError):
        _launch(args)


def test_rejects_mismatched_row_counts():
    args = _base_args()
    args["prefix"] = args["prefix"][:-1]
    with pytest.raises(ValueError):
        _launch(args)


def test_rejects_bad_state_layout():
    args = _base_args()
    args["state"] = args["state"].transpose(1, 2)
    with pytest.raises(ValueError):
        _launch(args)


def test_rejects_bad_state_width():
    args = _base_args()
    args["state"] = args["state"][..., :-1]
    with pytest.raises(ValueError):
        _launch(args)


def test_rejects_bad_ancestor_width():
    args = _base_args()
    args["ancestors"] = args["ancestors"][:, :0]
    with pytest.raises(ValueError):
        _launch(args)


def test_rejects_rank_one_page_table():
    args = _base_args()
    args["table"] = args["table"][0]
    with pytest.raises(ValueError):
        _launch(args)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
