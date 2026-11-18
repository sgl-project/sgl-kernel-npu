"""Top-K set comparison against independent dequantized PyTorch math."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check, make_executor
from tests.python.utils.performance import check_performance

CASES = [
    ((513,), (1,), (3,)),
    ((513,), (128,), (0,)),
    pytest.param(((513, 257), (2, 3), (0, 1)), marks=pytest.mark.extra_checks),
    pytest.param(((33,), (4,), (0,)), marks=pytest.mark.extra_checks),
    pytest.param(((1,), (4,), (0,)), marks=pytest.mark.extra_checks),
]
CASE_IDS = [
    "deepseek-v4-flash-c4-decode-q1-n64-d128-topk512-page32",
    "deepseek-v4-flash-c4-prefill-q128-n64-d128-topk512-page32",
    "synthetic-ragged-prefill",
    "synthetic-page-boundary",
    "synthetic-empty-causal-prefix",
]


def generate_data(case, mode, scale_divisor=64):
    lengths, queries, residuals = case
    dtype = torch.float8_e4m3fn if mode == 1 else torch.int8
    scale_dtype = torch.float32 if mode == 1 else torch.float16
    # A5 QuantLightningIndexer requires FP32 weights; A3 uses FP16.
    weight_dtype = torch.float32 if mode == 1 else torch.float16
    pages_per_batch = (max(lengths) + 31) // 32
    pages = len(lengths) * pages_per_batch
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(
        len(lengths), pages_per_batch
    )
    q = torch.zeros(sum(queries), 64, 128, device="cpu")
    q[:, :32, :4] = 1
    q[:, 32:, :4] = -1
    k = torch.zeros(pages, 32, 1, 128, device="cpu")
    k[..., :4] = 1
    scales = torch.ones(pages, 32, 1, dtype=scale_dtype)
    for b, length in enumerate(lengths):
        for j in range(length):
            scales[int(table[b, j // 32]), j % 32, 0] = 2 ** (j / scale_divisor)
    args = (
        q.to(dtype).view(torch.uint8).to(table.device).view(dtype),
        k.to(dtype).view(torch.uint8).to(table.device).view(dtype),
        torch.ones(sum(queries), 64, dtype=weight_dtype),
        torch.full((sum(queries), 64), 0.5, dtype=scale_dtype),
        scales,
        512,
        mode,
    )
    cu = torch.tensor([0, *queries], dtype=torch.int32).cumsum(0).to(torch.int32)
    common = dict(
        cu_seqlens_q=cu,
        seqused_k=torch.tensor(lengths, dtype=torch.int32),
        cmp_residual_k=torch.tensor(residuals, dtype=torch.int32),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
    )
    return args, common, table


def hardware_mode():
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    return 1 if get_hardware().generation == Generation.ARCH35 else 2


def _cpu_tensor(value):
    if not isinstance(value, torch.Tensor):
        return value
    if str(value.dtype).startswith("torch.float8_"):
        return value.view(torch.uint8).cpu().view(value.dtype)
    return value.cpu()


def reference_inputs(args, kwargs, kwargs_ref, impl_ref):
    """Keep eager Top-K off NPU and copy bytes without an FP8 device cast."""
    kwargs = {**kwargs, **kwargs_ref}
    if impl_ref != "torch_native":
        return args, kwargs
    return (
        tuple(_cpu_tensor(value) for value in args),
        {key: _cpu_tensor(value) for key, value in kwargs.items()},
    )


def paired_inputs(data):
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )

    from tests.python.utils.common import (
        get_skip_acc_check,
        get_test_impls,
        require_distinct_impls,
        require_test_impl,
    )

    args, common, table = data
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is True:
        require_test_impl(quant_lightning_indexer, impl)
    else:
        require_distinct_impls(quant_lightning_indexer, impl, impl_ref)
    metadata_args = (64, 1, 128, 512, args[-1])
    metadata = quant_lightning_indexer_metadata(*metadata_args, impl=impl, **common)
    kwargs = dict(common, block_table=table, metadata=metadata, return_value=0)
    kwargs_ref = None
    if get_skip_acc_check() is not True:
        ref_args, ref_common = reference_inputs(metadata_args, common, {}, impl_ref)
        kwargs_ref = dict(
            metadata=quant_lightning_indexer_metadata(
                *ref_args,
                impl=impl_ref,
                **ref_common,
            )
        )
    return args, kwargs, kwargs_ref, metadata_args, common


perf_shape = [CASES[1]]


def assert_indexer_accuracy(actual, expected):
    # Compare the selected set; accelerator kernels can return a different order.
    actual_indices, actual_values = actual
    expected_indices, expected_values = expected
    assert_accuracy(
        (actual_indices.cpu().sort(dim=-1).values, actual_values.cpu()),
        (expected_indices.cpu().sort(dim=-1).values, expected_values.cpu()),
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_quant_lightning_indexer(case):
    from sgl_kernel_npu.operators.indexer import quant_lightning_indexer

    args, kwargs, kwargs_ref, _, _ = paired_inputs(generate_data(case, hardware_mode()))
    executor = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    if get_skip_acc_check() is not True:
        reference_args, reference_kwargs = reference_inputs(
            args, kwargs, kwargs_ref, executor.impl_ref
        )
        expected = quant_lightning_indexer(
            *reference_args, **reference_kwargs, impl=executor.impl_ref
        )
        assert_indexer_accuracy(executor(), expected)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=case in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize("case", CASES[:2], ids=CASE_IDS[:2])
def test_quant_lightning_indexer_queued_lifetimes(case):
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )

    if get_skip_acc_check() is True:
        pytest.skip("Queued lifetime regression requires accuracy comparison")
    args, kwargs, kwargs_ref, metadata_args, common = paired_inputs(
        generate_data(case, hardware_mode())
    )
    executor = make_executor(quant_lightning_indexer, args=args, kwargs=kwargs)
    reference_args, reference_kwargs = reference_inputs(
        args, kwargs, kwargs_ref, executor.impl_ref
    )
    expected = quant_lightning_indexer(
        *reference_args, **reference_kwargs, impl=executor.impl_ref
    )
    results = []
    for _ in range(4):
        # Each iteration replaces the previous metadata Tensor. The queued
        # bridge must own its inputs and workspace until the ACL call runs.
        metadata = quant_lightning_indexer_metadata(
            *metadata_args, impl=executor.impl, **common
        )
        results.append(
            quant_lightning_indexer(
                *args, **{**kwargs, "metadata": metadata}, impl=executor.impl
            )
        )
        # Encourage allocator reuse without a host read or synchronization.
        torch.empty(1 << 20, dtype=torch.uint8, device=args[0].device).fill_(0xA5)
    for result in results:
        assert_indexer_accuracy(result, expected)


@pytest.mark.parametrize("ratio,query_length,heads", [(1, 1, 32), (2, 5, 64)])
def test_a3_v41_three_modes_and_graph_replay(ratio, query_length, heads):
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )
    from sgl_kernel_npu.operators.indexer.quant_lightning_indexer import (
        torch_npu_soc_family,
    )
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version

    if torch_npu_soc_family() != 250 or not detect_cann_version().startswith("9.2."):
        pytest.skip("V4.1 INT8 candidate modes require A3 + CANN 9.2")
    length = 16449  # More than 2048 blocks; exercise actual candidate pruning.
    with torch.device("cpu"):
        args, common, table = generate_data(
            ((length,), (query_length,), (0,)), 2, scale_divisor=2048
        )
        # Keep token-level TopK scores distinct despite FP16 scale rounding.
        for position in range(length - 1024, length):
            args[4][table[0, position // 32], position % 32] = (
                position - (length - 1024) + 1
            )
    args = tuple(
        value[:, :heads].contiguous() if i in (0, 2, 3) else value
        for i, value in enumerate(args)
    )
    common["cmp_ratio"] = ratio
    if ratio == 1:
        common["cmp_residual_k"] = None
    args = tuple(
        value.to("npu") if isinstance(value, torch.Tensor) else value for value in args
    )
    device_common = {
        name: value.to("npu") if isinstance(value, torch.Tensor) else value
        for name, value in common.items()
    }
    metadata = quant_lightning_indexer_metadata(heads, 1, 128, 512, 2, **device_common)
    kwargs = dict(
        device_common,
        block_table=table.to("npu"),
        metadata=metadata,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    source = quant_lightning_indexer(*args, **kwargs, candidate_mode=1)
    cpu_args = tuple(_cpu_tensor(value) for value in args)
    cpu_common = {
        name: _cpu_tensor(value)
        for name, value in dict(
            common,
            block_table=table,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
        ).items()
    }
    expected = quant_lightning_indexer(
        *cpu_args, **cpu_common, candidate_mode=1, impl="torch_native"
    )
    assert torch.equal(source[0].cpu().sort(-1).values, expected[0].sort(-1).values)
    assert torch.equal(source[2].cpu().sort(-1).values, expected[2].sort(-1).values)

    # Capture source -> consumer plus fresh device metadata. Both the candidate
    # output and metadata must use replay inputs, including changing KV lengths.
    def select():
        refreshed = quant_lightning_indexer_metadata(
            heads, 1, 128, 512, 2, **device_common
        )
        metadata.copy_(refreshed)
        source = quant_lightning_indexer(*args, **kwargs, candidate_mode=1)
        consumer = quant_lightning_indexer(
            *args, **kwargs, candidate_mode=2, candidate_topk_index=source[2]
        )
        return source, consumer

    select()  # Load the lazy extension before capture.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_source, captured_consumer = select()
    args[4].reciprocal_()
    device_common["seqused_k"].sub_(128)
    graph.replay()
    eager_source, eager_consumer = select()
    torch.npu.synchronize()
    for captured, eager in (
        (captured_source[0], eager_source[0]),
        (captured_source[2], eager_source[2]),
        (captured_consumer[0], eager_consumer[0]),
    ):
        assert torch.equal(captured.cpu().sort(-1).values, eager.cpu().sort(-1).values)
    ordinary = quant_lightning_indexer(*args, **kwargs, candidate_mode=3)
    assert ordinary[0].shape == (query_length, 1, 512)


def test_a3_consumer_dirty_candidates_and_replay():
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer,
        quant_lightning_indexer_metadata,
    )
    from sgl_kernel_npu.operators.indexer.quant_lightning_indexer import (
        torch_npu_soc_family,
    )
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version

    if torch_npu_soc_family() != 250 or not detect_cann_version().startswith("9.2."):
        pytest.skip("V4.1 INT8 candidate modes require A3 + CANN 9.2")
    length = 32769  # Exercise fallback -> membership-table transition on replay.
    with torch.device("cpu"):
        case = ((length, length - 64), (2, 3), (0, 0))
        args, common, table = generate_data(case, 2, scale_divisor=4096)
        args[4].fill_(1)  # Candidate eligibility, rather than scores, decides the set.
        candidates = torch.full((5, 1, 2048), -1, dtype=torch.int32)
        domains = [
            torch.arange(320, 384),
            torch.arange(384, 448),
            torch.arange(320, 448, 2),
            torch.arange(321, 449, 2),
            torch.arange(352, 416),
        ]
        for row, blocks in enumerate(domains):
            # Exactly 64 distinct blocks = 512 visible keys, all below the
            # earliest query's causal boundary. Duplicates change no eligibility.
            scrambled = blocks.flip(0).to(torch.int32)
            candidates[row, 0, :64] = scrambled
            candidates[row, 0, 64:128] = scrambled
            candidates[row, 0, 128:131] = torch.tensor(
                [-7, length // 8 + 4096, 2**31 - 1], dtype=torch.int32
            )
        expected = torch.stack(
            [(blocks[:, None] * 8 + torch.arange(8)).flatten() for blocks in domains]
        ).int()
    args = tuple(
        value[:, :32].contiguous() if i in (0, 2, 3) else value
        for i, value in enumerate(args)
    )
    args = tuple(
        value.to("npu") if isinstance(value, torch.Tensor) else value for value in args
    )
    common["cmp_ratio"] = 1
    common["cmp_residual_k"] = None
    common = {
        key: value.to("npu") if isinstance(value, torch.Tensor) else value
        for key, value in common.items()
    }
    candidates = candidates.to("npu")
    metadata = quant_lightning_indexer_metadata(32, 1, 128, 512, 2, **common)
    kwargs = dict(common, block_table=table.to("npu"), metadata=metadata)

    def select():
        metadata.copy_(quant_lightning_indexer_metadata(32, 1, 128, 512, 2, **common))
        return quant_lightning_indexer(
            *args, **kwargs, candidate_mode=2, candidate_topk_index=candidates
        )[0]

    eager = select()
    assert torch.equal(
        eager.cpu().view(5, 512).sort(-1).values, expected.sort(-1).values
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = select()
    # The long case crosses from the legacy mask into the membership table
    # during replay, without a different captured shape or a new operator.
    for shift in range(1, 4):
        candidates.copy_(candidates.roll(1, 0))
        common["seqused_k"].sub_(128)
        graph.replay()
        assert torch.equal(
            captured.cpu().view(5, 512).sort(-1).values,
            expected.roll(shift, 0).sort(-1).values,
        )
