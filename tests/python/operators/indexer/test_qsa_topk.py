"""FP32 score[R,M] -> int32[R,K], K=512/2048, sequence-relative indices.

Reference: independent torch_native selection. Exact score multiset, membership,
uniqueness, ordered short rows and -1 suffix; equal-score indices may differ.
Qwen3.8-Flash-Next selects 512 blocks from contiguous MQA scores. Its replicated
indexer keeps the same heads under TP/attention DP/EP. Local B1/B8 with W4
verification gives T4/T32. Decode graph M=262144/4=65536; live lengths vary.
SGLang qsa_indexer.py caps packed scores at 128 MiB; metadata.py supplies causal
request ranges. CP8 uses local zigzag queries against global keys. The CP case
tests only that input contract, not distributed execution. Values are synthetic.
"""

import pytest
import torch
from sgl_kernel_npu.operators.indexer import qsa_topk
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

pytestmark = pytest.mark.skipif(
    get_hardware().generation not in {Generation.ARCH22, Generation.ARCH35},
    reason="These model operators target arch22 and arch35",
)

perf_shape = [(8, 65536), (32, 65536), (512, 65536), (4096, 1024)]


def generate_data(rows, width, topk, pattern):
    torch.manual_seed(11)
    storage = torch.randn(
        rows, width + (19 if pattern == "strided" else 0), dtype=torch.float32
    )
    score = storage[:, :width]
    if pattern == "ties":
        score.zero_()
    elif pattern == "binary":
        score.copy_((score > 0).float())
    starts = torch.zeros(rows, dtype=torch.int32)
    lengths = torch.full((rows,), width, dtype=torch.int32)
    if pattern in ("prefill", "prefix", "packed_prefix", "cp8"):
        positions = torch.arange(rows, dtype=torch.int32)
        if pattern == "prefix":
            positions += width * 4 - rows
        elif pattern == "packed_prefix":
            starts[rows // 2 :] = width // 2
            positions = positions % (rows // 2) + width // 2 * 4 - rows // 2
        elif pattern == "cp8":
            positions = torch.cat(
                (torch.arange(256, 512), torch.arange(3584, 3840))
            ).int()
        lengths = (positions + 1) // 4
        columns = torch.arange(width)[None, :]
        score.masked_fill_(
            (columns < starts[:, None]) | (columns >= (starts + lengths)[:, None]),
            -float("inf"),
        )
    elif pattern == "verify":
        lengths = width - 1 + ((torch.arange(rows, dtype=torch.int32) % 4 + 1) // 4)
    elif rows >= 4:
        lengths[:4] = torch.tensor(
            [0, 1, min(topk, width), min(topk + 1, width)], dtype=torch.int32
        )
        if pattern != "decode":
            starts[1] = min(3, max(0, width - 1))
        score[0] = float("nan")
    return score, lengths, topk, starts


def check(score, lengths, starts, result, k):
    assert result.shape == (score.shape[0], k)
    assert result.dtype == torch.int32 and result.device == score.device
    assert result.is_contiguous()
    lens = lengths.cpu().long()
    begin = torch.zeros_like(lens) if starts is None else starts.cpu().long()
    output = result.cpu().long()
    for base in range(0, score.shape[0], 32):
        cpu = score[base : base + 32].cpu()
        ls, ss = lens[base : base + 32], begin[base : base + 32]
        ids = output[base : base + 32]
        # Batch the large homogeneous prefill oracle without changing tolerance.
        if ls.numel() and bool((ls == ls[0]).all() & (ss == ss[0]).all()) and ls[0] > k:
            l, s = int(ls[0]), int(ss[0])
            assert ((ids >= 0) & (ids < l)).all()
            ordered = ids.sort(1).values
            assert not (ordered[:, 1:] == ordered[:, :-1]).any()
            valid = cpu[:, s : s + l]
            torch.testing.assert_close(
                valid.gather(1, ids).sort(1).values,
                valid.topk(k, dim=1).values.sort(1).values,
                rtol=0,
                atol=0,
            )
            continue
        for i in range(len(ls)):
            l, s = int(ls[i]), int(ss[i])
            n = min(l, k)
            chosen = ids[i, :n]
            assert ((chosen >= 0) & (chosen < l)).all()
            assert chosen.unique().numel() == n
            assert (ids[i, n:] == -1).all()
            if l <= k:
                assert torch.equal(chosen, torch.arange(l, device="cpu"))
            else:
                valid = cpu[i, s : s + l]
                torch.testing.assert_close(
                    valid[chosen].sort().values,
                    valid.topk(k).values.sort().values,
                    rtol=0,
                    atol=0,
                )


def check_accuracy(args):
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check():
        require_test_impl(qsa_topk, impl)
        return make_executor(qsa_topk, args=args, impl=impl, impl_ref=impl_ref)
    require_distinct_impls(qsa_topk, impl, impl_ref)
    score, lengths, topk, starts = args
    # check() permits any correct tie choice, unlike a tensor-index assert_close.
    for name in [impl, impl_ref]:
        check(score, lengths, starts, qsa_topk(*args, impl=name), topk)
    return make_executor(
        qsa_topk, args=args, impl=impl, impl_ref=impl_ref, accuracy_checked=True
    )


@pytest.mark.parametrize(
    "rows,width,topk,pattern",
    [
        pytest.param(
            1, 65536, 512, "decode", id="qwen3-8-flash-next-decode-b1-m65536-k512"
        ),
        pytest.param(
            8, 65536, 512, "decode", id="qwen3-8-flash-next-decode-b8-m65536-k512"
        ),
        pytest.param(
            4, 65536, 512, "verify", id="qwen3-8-flash-next-verify-b1-w4-t4-m65536-k512"
        ),
        pytest.param(
            32,
            65536,
            512,
            "verify",
            id="qwen3-8-flash-next-verify-b8-w4-t32-m65536-k512",
        ),
        pytest.param(
            4096, 1024, 512, "prefill", id="qwen3-8-flash-next-prefill-t4096-m1024-k512"
        ),
        pytest.param(
            512, 65536, 512, "prefix", id="qwen3-8-flash-next-prefix-t512-m65536-k512"
        ),
        pytest.param(
            256,
            131072,
            512,
            "packed_prefix",
            id="qwen3-8-flash-next-packed-b2-prefix-t256-m131072-k512",
        ),
        pytest.param(
            512,
            1024,
            512,
            "cp8",
            id="qwen3-8-flash-next-cp8-prefill-local-t512-m1024-k512",
        ),
        pytest.param(0, 4096, 512, "random", id="edge-empty"),
        pytest.param(8, 256, 512, "random", id="edge-short-row-padding"),
        pytest.param(33, 8193, 512, "strided", id="edge-strided-row-and-column-tail"),
        pytest.param(8, 4096, 512, "ties", id="edge-k512-equal-scores"),
        pytest.param(8, 4096, 512, "binary", id="edge-k512-binary-scores"),
        pytest.param(8, 4096, 2048, "random", id="edge-large-k2048"),
        pytest.param(8, 4096, 2048, "ties", id="edge-k2048-equal-scores"),
        pytest.param(2, 262145, 512, "random", id="edge-wide-selection-k512"),
        pytest.param(2, 262145, 2048, "random", id="edge-wide-selection-k2048"),
    ],
)
def test_qsa_topk_accuracy(rows, width, topk, pattern):
    # A length-3 row keeps [0, 1, 2] and pads the remaining slots with -1.
    args = generate_data(rows, width, topk, pattern)
    saved = [x.clone() for x in [args[0], args[1], args[3]]]
    executor = check_accuracy(args)
    for x, before in zip([args[0], args[1], args[3]], saved):
        torch.testing.assert_close(x, before, atol=0, rtol=0, equal_nan=True)
    check_performance(
        executor,
        enable_perf=(rows, width) in perf_shape and topk == 512,
    )


@pytest.mark.parametrize(
    "topk,width",
    [
        pytest.param(512, 65536, id="qwen3-8-flash-next-decode-b8-m65536-k512"),
        pytest.param(512, 256, id="edge-short-row-padding"),
        pytest.param(2048, 262145, id="edge-wide-selection-k2048"),
    ],
)
def test_qsa_topk_graph_updates(topk, width):
    args = generate_data(8, width, topk, "random")
    score, lengths, _, starts = args
    if width == 65536:
        score.nan_to_num_()
        starts.zero_()
        lengths.fill_(1024)
    _check_graph_updates(
        qsa_topk,
        args,
        [
            lambda: score.neg_(),
            lambda: (score.nan_to_num_(), starts.zero_(), lengths.fill_(width)),
            lambda: (score.nan_to_num_(), lengths.fill_(17)),
            lambda: starts.fill_(3),
        ],
        compare=lambda out: check(score, lengths, starts, out, topk),
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("bad", ["dtype", "topk", "bounds_dtype", "bounds_stride"])
def test_qsa_topk_rejects_metadata(bad):
    args = list(generate_data(8, 4096, 512, "random"))
    if bad == "dtype":
        args[0] = args[0].half()
    elif bad == "topk":
        args[2] = 513
    elif bad == "bounds_dtype":
        args[1] = args[1].long()
    else:
        args[1] = torch.stack((args[1], args[1]), -1)[:, 0]
    with pytest.raises(ValueError):
        qsa_topk(*args)


def _tensors(args):
    return [value for value in args if isinstance(value, torch.Tensor)]


def _assert_unchanged(args, saved):
    for value, before in zip(_tensors(args), saved):
        torch.testing.assert_close(value, before, atol=0, rtol=0, equal_nan=True)


def _check_graph_updates(
    operator, args, updates, *, standard=None, kwargs=None, compare=None
):
    """Capture the public default; compare each replay to an independent reference.

    updates mutate contents at stable addresses. References run outside capture.
    compare can supply a semantic comparator for nonunique TopK outputs.
    """
    kwargs = dict(kwargs or {})
    inputs = _tensors(args)
    pointers = [x.data_ptr() for x in inputs]
    for _ in range(3):
        operator(*args, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operator(*args, **kwargs)
    for change in [None, *updates]:
        if change is not None:
            change()
        saved = [x.clone() for x in inputs]
        graph.replay()
        torch.npu.synchronize()
        if compare is None:
            reference = operator(*args, **kwargs, impl="torch_native")
            assert_accuracy(output, reference, standard)
        else:
            compare(output)
        assert [x.data_ptr() for x in inputs] == pointers
        _assert_unchanged(args, saved)
    return graph, output
