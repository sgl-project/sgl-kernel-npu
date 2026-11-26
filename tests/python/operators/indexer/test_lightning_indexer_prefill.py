"""Public lightning_indexer_prefill operator: selection, grouping and guards.

Shapes follow one ratio-4 causal prefill request: Q holds the last ``rows``
queries as ``[rows, 4, 128]`` and K holds ``sequence_length // 4`` compressed
keys. Row counts are representative prefill chunks, not captured tensors.

Two properties are checked separately because only one of them is exact:

- Against the independent ``torch_native`` reference, the selected key set and
  the non-increasing score order must match. The exact permutation of keys with
  equal or one-ulp-apart scores must not, because the backends accumulate the
  per-head sum in different orders.
- Between the grouped and independent CANN launch geometries, the ordered
  indices must be bitwise identical, including ties and -1 padding. That is the
  invariant four-query batching has to preserve.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# Prefill chunks selected for performance collection; the grouped path needs
# at least 1024 rows and 512 keys to engage.
perf_shape = [(16384, 32768), (8313, 44408)]


def generate_data(rows, sequence_length, zero=False):
    """Create NPU inputs using the device context supplied by conftest."""
    torch.npu.manual_seed(4201)
    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16)
    k = torch.randn(sequence_length // 4, 1, 128, dtype=torch.bfloat16)
    if zero:
        q.zero_()
        k.zero_()
    return q, k


def assert_same_selection(actual, reference, q, k, sequence_length):
    """Compare a selection against an independent reference, tolerating ties.

    Both arguments are ordered int32 ``[rows, 512]`` index tensors. Neither the
    permutation nor the set of selected keys is reproducible across backends:
    the two accumulate the per-head sum in different orders, so keys whose
    scores agree to within a float32 ulp can be permuted, and a tie straddling
    the Top-512 cut can put a different key on either side of it. What is
    well defined is the multiset of selected *scores*, the number of valid
    entries, and the non-increasing order. This checks those three.

    The score is recomputed here rather than imported, so the check does not
    reuse either implementation's code. Rows are chunked because the per-head
    dot product is proportional to rows x 4 x keys.
    """
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype == torch.int32
    assert actual.device == reference.device
    assert torch.equal(
        (actual >= 0).sum(1), (reference >= 0).sum(1)
    ), "valid entry counts differ from the reference"
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        return
    assert bool(
        ((actual < keys) & (actual >= -1)).all()
    ), "returned key indices are out of range"
    width = min(keys, 512)
    chunk = max(1, (64 << 20) // (4 * keys * 4))
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        dot = torch.einsum("qhd,kd->qhk", q[begin:end].float(), k[:, 0].float())
        score = dot.relu().sum(dim=1)
        picked = []
        for indices in (actual, reference):
            window = indices[begin:end, :width]
            valid = window >= 0
            taken = score.gather(1, window.clamp_min(0).long())
            picked.append(taken.masked_fill(~valid, -torch.inf))
        # The order within a tie is free, so compare the scores as a multiset.
        torch.testing.assert_close(
            torch.sort(picked[0], dim=-1).values,
            torch.sort(picked[1], dim=-1).values,
            rtol=1e-5,
            atol=1e-4,
            equal_nan=False,
            msg=lambda m: "selected scores differ from the reference: " + m,
        )
        # Within one row the returned keys must still be ranked by score. The
        # tolerance is loose because the score accumulates a 128-element dot
        # product and a four-head sum in float32, and the two backends
        # accumulate them in different orders; a real mis-sort moves a key by
        # the score spread, which is three orders of magnitude larger.
        valid = picked[0] != -torch.inf
        step = picked[0][:, :-1] - picked[0][:, 1:]
        both = valid[:, :-1] & valid[:, 1:]
        tolerance = 1e-3 + picked[0][:, :-1].abs() * 1e-5
        assert bool(
            (step[both] >= -tolerance[both]).all()
        ), "returned keys are not ordered by non-increasing score"


@pytest.mark.parametrize(
    "rows,sequence_length",
    [
        # Case 01: Minimum nonempty request; one compressed key.
        pytest.param(7, 7, id="single-key"),
        # Case 02: Fewer rows than one key; selection is empty.
        pytest.param(3, 3, id="no-visible-key"),
        # Case 03: Unaligned rows, fewer keys than the 512-wide output.
        pytest.param(65, 2052, id="narrow-partial-output"),
        # Case 04: Just below the grouping threshold; independent path.
        pytest.param(1023, 4096, id="below-group-threshold"),
        # Case 05-08: Each of the four grouping phases.
        pytest.param(1024, 4096, id="group-phase-0"),
        pytest.param(1025, 4096, id="group-phase-1"),
        pytest.param(1026, 4096, id="group-phase-2"),
        pytest.param(1027, 4096, id="group-phase-3"),
        # Case 09: Square prefill chunk.
        pytest.param(4096, 4096, id="prefill-4096x4096"),
        # Case 10-11: Production prefill chunks; also profiled.
        pytest.param(16384, 32768, id="prefill-16384x32768"),
        pytest.param(8313, 44408, id="prefill-8313x44408"),
        # Case 12: Zero rows returns a typed empty-row output.
        pytest.param(0, 4096, id="zero-rows"),
    ],
)
def test_lightning_indexer_prefill(rows, sequence_length):
    from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

    q, k = generate_data(rows, sequence_length)
    kwargs = dict(q=q, k=k, sequence_length=sequence_length)
    impl, impl_ref = get_test_impls()
    executor = make_executor(lightning_indexer_prefill, kwargs=kwargs)
    if get_skip_acc_check() is not True:
        require_distinct_impls(lightning_indexer_prefill, impl, impl_ref)
        reference = lightning_indexer_prefill(**kwargs, impl=impl_ref)
        assert_same_selection(executor(), reference, q, k, sequence_length)
        executor.accuracy_checked = True
    else:
        require_test_impl(lightning_indexer_prefill, impl)
    check_performance(executor, enable_perf=(rows, sequence_length) in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize("zero", [False, True], ids=["random", "all-zero"])
@pytest.mark.parametrize(
    "rows,sequence_length",
    [
        pytest.param(1024, 2048, id="group-phase-1-small-keys"),
        pytest.param(1025, 4096, id="group-phase-2"),
        pytest.param(1026, 4097, id="unaligned-length"),
        pytest.param(4096, 16384, id="prefill-4096x16384"),
    ],
)
def test_lightning_indexer_prefill_grouping_is_exact(rows, sequence_length, zero):
    """Grouped and independent launches must agree bit for bit.

    All-zero inputs make every score equal, so this also pins the tie order:
    four-query batching must not reorder tied keys relative to the independent
    launch. An accuracy reference cannot express this, because the property is
    between two configurations of the same implementation.
    """
    from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

    impl, _ = get_test_impls()
    if impl == "torch_native":
        pytest.skip("grouping geometry exists only in the CANN implementation")
    require_test_impl(lightning_indexer_prefill, impl)
    q, k = generate_data(rows, sequence_length, zero=zero)
    grouped = lightning_indexer_prefill(
        q, k, sequence_length=sequence_length, group4=True, impl=impl
    )
    independent = lightning_indexer_prefill(
        q, k, sequence_length=sequence_length, group4=False, impl=impl
    )
    torch.npu.synchronize()
    assert torch.equal(grouped, independent)


@pytest.mark.extra_checks
def test_lightning_indexer_prefill_visible_counts():
    """Each row must return exactly its causally visible key count."""
    from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill
    from sgl_kernel_npu.operators.indexer.lightning_indexer_prefill import (
        visible_key_counts,
    )

    impl, _ = get_test_impls()
    require_test_impl(lightning_indexer_prefill, impl)
    for rows, sequence_length in [(7, 7), (65, 2052), (1025, 4096)]:
        q, k = generate_data(rows, sequence_length)
        out = lightning_indexer_prefill(
            q, k, sequence_length=sequence_length, impl=impl
        )
        counts = visible_key_counts(rows, sequence_length, q.device).clamp(max=512)
        assert torch.equal((out >= 0).sum(1).int(), counts)
        assert bool(((out < k.shape[0]) | (out == -1)).all())


@pytest.mark.extra_checks
def test_lightning_indexer_prefill_queued_calls_agree():
    """Queued launches must not read inputs the caller already replaced."""
    from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

    impl, _ = get_test_impls()
    require_test_impl(lightning_indexer_prefill, impl)
    q, k = generate_data(1025, 4096)
    first = lightning_indexer_prefill(q, k, sequence_length=4096, impl=impl).clone()
    torch.npu.synchronize()
    queued = [
        lightning_indexer_prefill(q.clone(), k.clone(), sequence_length=4096, impl=impl)
        for _ in range(3)
    ]
    torch.npu.synchronize()
    for out in queued:
        assert torch.equal(out, first)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "invalid, message",
    [
        pytest.param("length-negative", "nonnegative int32", id="length-negative"),
        pytest.param("length-zero", "disagree", id="length-zero"),
        pytest.param("length-float", "nonnegative int32", id="length-float"),
        pytest.param("length-bool", "nonnegative int32", id="length-bool"),
        pytest.param("length-overflow", "nonnegative int32", id="length-overflow"),
        pytest.param("length-mismatch", "disagree", id="length-mismatch"),
        pytest.param("group4-int", "group4 must be a bool", id="group4-int"),
        pytest.param("dtype-float32", "must be BF16", id="dtype-float32"),
        pytest.param("q-noncontiguous", "must be contiguous", id="q-noncontiguous"),
        pytest.param("q-rank", "expected Q", id="q-wrong-heads"),
        pytest.param("cpu", "same NPU", id="cpu-tensors"),
    ],
)
def test_lightning_indexer_prefill_rejects_invalid_arguments(invalid, message):
    """Every contract violation raises ValueError for its own reason."""
    from sgl_kernel_npu.operators.indexer import lightning_indexer_prefill

    q, k = generate_data(1024, 2048)
    sequence_length, group4 = 2048, True
    if invalid == "length-negative":
        sequence_length = -1
    elif invalid == "length-zero":
        sequence_length = 0
    elif invalid == "length-float":
        sequence_length = 2048.0
    elif invalid == "length-bool":
        sequence_length = True
    elif invalid == "length-overflow":
        sequence_length = 2**31
    elif invalid == "length-mismatch":
        sequence_length = 2044
    elif invalid == "group4-int":
        group4 = 1
    elif invalid == "dtype-float32":
        q = q.float()
    elif invalid == "q-noncontiguous":
        # Keep the declared shape so the contiguity check is the one that fires.
        q = torch.randn(q.shape[0], 4, 256, dtype=torch.bfloat16)[:, :, ::2]
    elif invalid == "q-rank":
        q = q[:, :2].contiguous()
    elif invalid == "cpu":
        q, k = q.cpu(), k.cpu()
    else:
        raise ValueError(f"Unknown invalid case: {invalid}")
    with pytest.raises(ValueError, match=message):
        lightning_indexer_prefill(q, k, sequence_length=sequence_length, group4=group4)
