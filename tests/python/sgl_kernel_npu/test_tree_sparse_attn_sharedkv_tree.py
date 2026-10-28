"""NPU parity and contract tests for ``npu_sparse_attn_sharedkv_tree``.

The operator is the Triton port of the DSV4 DSpark tree attention kernel
(vllm-ascend commit ``8947f1f9f``).  Expected values come exclusively from the
independent CPU oracle in ``tree_reference.py`` (decision D5); production code
never imports the oracle.

The module is skipped, not failed, on hosts without ``torch_npu`` or Triton;
that is how the CPU-only CI image runs it.  The numerical GREEN gate is the NPU
command documented in ``tree/README.md``.
"""

import pytest
import torch

# These guards must run before importing the package: the repository root
# ``sgl_kernel_npu/__init__.py`` loads ``libsgl_kernel_npu.so`` and torch_npu.
pytest.importorskip("torch_npu")
pytest.importorskip("triton")

if not torch.npu.is_available():
    pytest.skip("an Ascend NPU is required", allow_module_level=True)

from sgl_kernel_npu.tree.sparse_attn_sharedkv_tree import (  # noqa: E402
    npu_sparse_attn_sharedkv_tree,
)
from tree_fixtures import make_attention_case  # noqa: E402
from tree_reference import sparse_attn_sharedkv_tree_ref  # noqa: E402

DEVICE = torch.device("npu:0")
OUT_TOL = 2e-2
LSE_TOL = 5e-2

_META_KEYS = ("spec", "cmp_mode")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _case_args(case):
    """Strip fixture-only metadata, keeping exactly the public kwargs."""
    return {k: v for k, v in case.items() if k not in _META_KEYS}


def _to_device(args):
    return {
        k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
        for k, v in args.items()
    }


def _run(case, **overrides):
    args = _case_args(case)
    args.update(overrides)
    npu_args = _to_device(args)
    q = npu_args.pop("q")
    out, lse = npu_sparse_attn_sharedkv_tree(
        q, return_softmax_lse=True, **npu_args
    )
    torch.npu.synchronize()
    return out.float().cpu(), lse.float().cpu()


def _assert_parity(case, **overrides):
    args = _case_args(case)
    args.update(overrides)
    ref_out, ref_lse = sparse_attn_sharedkv_tree_ref(
        **args, return_softmax_lse=True
    )
    out, lse = _run(case, **overrides)
    torch.testing.assert_close(out, ref_out.float(), rtol=OUT_TOL, atol=OUT_TOL)
    torch.testing.assert_close(lse, ref_lse.float(), rtol=LSE_TOL, atol=LSE_TOL)
    return out, ref_out, lse, ref_lse


def _pad_topk(case, width):
    """Widen C4 ``cmp_sparse_indices`` with ``-1`` padding."""
    idx = case["cmp_sparse_indices"]
    if idx.shape[-1] >= width:
        return
    padded = torch.full(
        (idx.shape[0], idx.shape[1], int(width)), -1, dtype=torch.int32
    )
    padded[..., : idx.shape[-1]] = idx
    case["cmp_sparse_indices"] = padded


def _base_case(cmp_mode="c4", *, seed=0, t=17, window=127, dtype=torch.float16,
               heads=8, sink=False, ratio=None, prefixes=None):
    if t == 1:
        budget, depth = 0, 0
    else:
        budget, depth = 16, 5
    if ratio is None:
        ratio = 128 if cmp_mode == "c128" else 4
    case = make_attention_case(
        seed=seed,
        ratio=ratio,
        batch=1,
        budget=budget,
        depth=depth,
        window=window,
        sink=sink,
        cmp_mode=cmp_mode,
        dtype=dtype,
        heads=heads,
        prefixes=prefixes,
    )
    # Keep the fixture's [T, N1] per-row sink table for T > 1: this exercises
    # the production ``sink_row=pid_t`` path and the fixed oracle indexing.
    return case


# ---------------------------------------------------------------------------
# parity across modes / dtypes / sinks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmp_mode,ratio,dtype,heads,sink",
    [
        ("c4", 4, torch.float16, 8, False),
        ("c4", 4, torch.bfloat16, 64, True),
        ("c128", 128, torch.float16, 8, False),
        ("c128", 128, torch.bfloat16, 64, True),
    ],
)
def test_compressed_modes(cmp_mode, ratio, dtype, heads, sink):
    # A 252-prefix row makes C128 reach the tree pool (step 3 < ancestor width).
    prefixes = [252] if cmp_mode == "c128" else None
    case = _base_case(
        cmp_mode,
        seed=3,
        t=17,
        dtype=dtype,
        heads=heads,
        sink=sink,
        ratio=ratio,
        prefixes=prefixes,
    )
    _assert_parity(case)


@pytest.mark.parametrize("t", [1, 17])
def test_token_counts(t):
    case = _base_case("c4", seed=4, t=t, dtype=torch.float16, heads=8)
    assert case["q"].shape[0] == t
    _assert_parity(case)


@pytest.mark.parametrize(
    "dtype,sink",
    [(torch.float16, False), (torch.bfloat16, True)],
)
def test_swa_only(dtype, sink):
    case = _base_case("swa", seed=6, dtype=dtype, heads=8, sink=sink)
    assert case["cmp_kv"] is None
    _assert_parity(case)


@pytest.mark.parametrize("window,expected_width", [(127, 128), (255, 256)])
def test_ori_widths(window, expected_width):
    case = _base_case("c4", seed=8, window=window, dtype=torch.float16, heads=8)
    assert case["ori_sparse_indices"].shape[-1] == expected_width
    _assert_parity(case)


@pytest.mark.parametrize("topk", [512, 1024])
def test_c4_topk_widths(topk):
    case = _base_case("c4", seed=9, dtype=torch.bfloat16, heads=8)
    _pad_topk(case, topk)
    assert case["cmp_sparse_indices"].shape[-1] == topk
    _assert_parity(case)


@pytest.mark.parametrize(
    "cmp_mode,ratio,prefixes",
    [("c4", 4, None), ("c128", 128, [252])],
)
def test_invalid_ancestors(cmp_mode, ratio, prefixes):
    case = _base_case(
        cmp_mode,
        seed=10,
        dtype=torch.float16,
        heads=8,
        ratio=ratio,
        prefixes=prefixes,
    )
    ancestors = case["tree_ancestors"].clone()
    depths = case["spec"].depths
    for row in range(ancestors.shape[0]):
        depth = int(depths[row])
        if depth > 0:
            ancestors[row, depth] = -1
    case["tree_ancestors"] = ancestors
    _assert_parity(case)


# ---------------------------------------------------------------------------
# empty rows, slot identity, metadata, sinks, LSE
# ---------------------------------------------------------------------------


def test_empty_row_returns_exact_zeros():
    case = _base_case("swa", seed=11, sink=True, dtype=torch.float16, heads=8)
    slots = case["ori_sparse_indices"].clone()
    slots[0].fill_(-1)
    case["ori_sparse_indices"] = slots
    out, _, lse, _ = _assert_parity(case)
    assert torch.count_nonzero(out[0]) == 0
    assert torch.count_nonzero(lse[0]) == 0


def test_reversed_slot_order_is_invariant():
    case = _base_case("swa", seed=12, dtype=torch.float16, heads=8)
    slots = case["ori_sparse_indices"].clone()
    for row in range(slots.shape[0]):
        row_slots = slots[row, 0]
        valid = row_slots[row_slots >= 0]
        if valid.numel() == 0:
            continue
        row_slots.fill_(-1)
        slots[row, 0, : valid.numel()] = torch.flip(valid, [0])
    case["ori_sparse_indices"] = slots
    _assert_parity(case)


def test_unreferenced_ori_pages_are_poisoned():
    case = _base_case("swa", seed=13, dtype=torch.float16, heads=8)
    args = _case_args(case)
    slots = args["ori_sparse_indices"]
    rows = int(args["ori_kv"].shape[1])
    pages = int(args["ori_kv"].shape[0])
    used = set()
    for row in range(int(slots.shape[0])):
        for slot in slots[row, 0].tolist():
            if slot < 0:
                break
            used.add(slot // rows)
    unused = [page for page in range(pages) if page not in used]
    assert unused, "fixture should leave spare physical pages"
    poisoned = args["ori_kv"].clone()
    poisoned[unused] = 4096.0
    case["ori_kv"] = poisoned
    out, _, _, _ = _assert_parity(case)
    assert torch.isfinite(out).all()


def test_metadata_contents_are_ignored():
    case = _base_case("c4", seed=14, dtype=torch.float16, heads=8)
    zeros = torch.zeros(1024, dtype=torch.int32)
    noise = torch.randint(
        -(2**31),
        2**31 - 1,
        (1024,),
        dtype=torch.int32,
        generator=torch.Generator().manual_seed(0),
    )
    out_a, lse_a = _run(case, metadata=zeros)
    out_b, lse_b = _run(case, metadata=noise)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    torch.testing.assert_close(lse_a, lse_b, rtol=0, atol=0)


def test_sink_layouts_are_equivalent():
    case = _base_case(
        "c4", seed=15, t=1, sink=True, dtype=torch.float16, heads=8
    )
    sinks = case["sinks"]
    assert sinks.shape == (1, 8)
    for variant in (sinks, sinks[0].contiguous(), sinks.unsqueeze(1)):
        variant_case = dict(case)
        variant_case["sinks"] = variant
        _assert_parity(variant_case)


def test_lse_omitted_returns_empty_tensor():
    case = _base_case("c4", seed=16, dtype=torch.float16, heads=8)
    args = _to_device(_case_args(case))
    q = args.pop("q")
    out, lse = npu_sparse_attn_sharedkv_tree(
        q, return_softmax_lse=False, **args
    )
    torch.npu.synchronize()
    assert out.shape == q.shape
    assert lse.numel() == 0


# ---------------------------------------------------------------------------
# host contract
# ---------------------------------------------------------------------------


def test_contract_violations_raise_value_error():
    case = _base_case("c4", seed=17, t=1, dtype=torch.float16, heads=8)
    base = _to_device(_case_args(case))
    q = base.pop("q")

    def call(**overrides):
        return npu_sparse_attn_sharedkv_tree(q, **{**base, **overrides})

    with pytest.raises(ValueError):
        call(layout_q="BSND")
    with pytest.raises(ValueError):
        call(layout_kv="BSND")
    with pytest.raises(ValueError):
        call(ori_mask_mode=3)
    with pytest.raises(ValueError):
        call(cmp_mask_mode=4)
    with pytest.raises(ValueError):
        call(cmp_ratio=8)
    with pytest.raises(ValueError):
        npu_sparse_attn_sharedkv_tree(q.float(), **base)
    with pytest.raises(ValueError):
        npu_sparse_attn_sharedkv_tree(q[..., :128].contiguous(), **base)
    with pytest.raises(ValueError):
        call(metadata=torch.zeros(8, dtype=torch.int32, device=DEVICE))
    with pytest.raises(ValueError):
        call(tree_ancestors=None)
    with pytest.raises(ValueError):
        call(ori_kv=base["ori_kv"].repeat(1, 1, 2, 1))
