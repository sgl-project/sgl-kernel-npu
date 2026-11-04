import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance

# Synthetic decode/prefill workloads with LongCat2's n=5, k=4 configuration.
# These are representative configurations, not captured production tensors.
perf_shape = [(16, 1, 5, 4), (1, 2048, 5, 4)]


def generate_data(batch_size, request_length, prefix_length, oe_n, oe_k, mode):
    """Create an updated token table and packed request metadata."""
    legacy = mode == "legacy"
    vocabulary = 32 if legacy else 65536
    mod_base = 5 if legacy else 65537
    mods = [mod_base + 2 * i + 1 for i in range((oe_n - 1) * oe_k)]
    weights = [[pow(vocabulary, delta, mod) for delta in range(oe_n)] for mod in mods]
    offsets = [0]
    for mod in mods:
        offsets.append(offsets[-1] + mod)

    if legacy:
        table = torch.tensor(
            [[3, 4, 5, -1, -1, -1], [7, 8, 9, 10, -1, -1]],
            dtype=torch.int32,
        )
        rows = [1, 0]
        columns = [1, 1]
        lengths = [2, 2]
        tokens = torch.tensor([8, 9, 4, 5], dtype=torch.int32)
    else:
        context_length = prefix_length + request_length + 3
        table = (
            torch.arange(batch_size * context_length, dtype=torch.int32)
            .reshape(batch_size, context_length)
            .mul(101)
            .add(7)
            .remainder(vocabulary)
        )
        rows = list(reversed(range(batch_size)))
        columns = [prefix_length] * batch_size
        lengths = [request_length] * batch_size
        if mode == "zero_request":
            lengths[1] = 0
        elif mode == "sentinel":
            # Stop at a negative current token and at an interior sentinel;
            # positive tokens preceding the sentinel must not contribute.
            table[rows[0], prefix_length] = -1
            table[rows[-1], prefix_length - 1] = -1
        tokens = torch.cat(
            [
                table[row, col : col + length]
                for row, col, length in zip(rows, columns, lengths)
            ]
        )

    ends = []
    total = 0
    for length in lengths:
        total += length
        ends.append(total)
    return dict(
        oe_weights=torch.tensor(weights, dtype=torch.int32).reshape(
            oe_n - 1, oe_k, oe_n
        ),
        oe_mods=torch.tensor(mods, dtype=torch.int32).reshape(oe_n - 1, oe_k),
        exclusive_oe_embeder_size_sums=torch.tensor(offsets, dtype=torch.int32),
        tokens=tokens,
        exclusive_req_len_sums=torch.tensor(ends, dtype=torch.int32),
        oe_token_table=table,
        row_indices=torch.tensor(rows, dtype=torch.int64),
        column_starts=torch.tensor(columns, dtype=torch.int32),
        batch_size=batch_size,
        oe_n=oe_n,
        oe_k=oe_k,
        max_context_len=table.shape[1],
    )


@pytest.mark.parametrize(
    "batch_size,request_length,prefix_length,oe_n,oe_k,mode",
    [
        pytest.param(2, 2, 1, 3, 2, "legacy", id="existing-n3-k2-example"),
        pytest.param(16, 1, 1024, 5, 4, "normal", id="longcat2-batched-decode"),
        pytest.param(1, 2048, 16, 5, 4, "normal", id="longcat2-chunked-prefill"),
        pytest.param(1, 1, 0, 2, 1, "normal", id="edge-first-token-minimum"),
        pytest.param(3, 7, 2, 5, 4, "normal", id="edge-short-history-odd-length"),
        pytest.param(3, 3, 4, 5, 4, "sentinel", id="edge-negative-history-sentinel"),
        pytest.param(3, 2, 1, 5, 4, "zero_request", id="edge-zero-length-request"),
    ],
)
def test_compute_n_gram_ids(
    batch_size, request_length, prefix_length, oe_n, oe_k, mode
):
    from sgl_kernel_npu.operators.embedding import compute_n_gram_ids

    impl, _ = get_test_impls()
    # Skip unsupported DUT architectures before resolving the reference.
    require_test_impl(compute_n_gram_ids, impl)
    inputs = generate_data(batch_size, request_length, prefix_length, oe_n, oe_k, mode)
    # Integer outputs are compared exactly by the shared accuracy checker.
    executor = check_accuracy(compute_n_gram_ids, kwargs=inputs)
    check_performance(
        executor,
        enable_perf=(batch_size, request_length, oe_n, oe_k) in perf_shape
        and mode == "normal",
    )
