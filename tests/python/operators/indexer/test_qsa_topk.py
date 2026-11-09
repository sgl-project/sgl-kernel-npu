"""FP32 score[R,M] -> int32[R,K], K=512/2048, sequence-relative indices.

Reference: independent torch_native selection. Exact score multiset, membership,
uniqueness, ordered short rows and -1 suffix; equal-score indices may differ.
"""

import pytest
import torch
from sgl_kernel_npu.operators.indexer import qsa_topk

from tests.python.operators.qsa_test_utils import ARCH22_ONLY, check_graph_updates
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

pytestmark = ARCH22_ONLY

perf_shape = [(8, 4096), (256, 16384)]


def generate_data(rows, width, topk, pattern):
    torch.manual_seed(11)
    storage = torch.randn(rows, width + 19, dtype=torch.float32)
    score = storage[:, :width]
    if pattern == "ties":
        score.zero_()
    elif pattern == "binary":
        score.copy_((score > 0).float())
    starts = torch.zeros(rows, dtype=torch.int32)
    lengths = torch.full((rows,), width, dtype=torch.int32)
    if rows >= 4:
        lengths[:4] = torch.tensor(
            [0, 1, min(topk, width), min(topk + 1, width)], dtype=torch.int32
        )
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


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "rows,width",
    [(0, 4096), (8, 256), (8, 4096), (33, 8193), (256, 16384), (2, 262145)],
)
@pytest.mark.parametrize("pattern", ["random", "ties", "binary"])
def test_qsa_topk_accuracy(rows, width, topk, pattern):
    args = generate_data(rows, width, topk, pattern)
    saved = [x.clone() for x in [args[0], args[1], args[3]]]
    executor = check_accuracy(args)
    for x, before in zip([args[0], args[1], args[3]], saved):
        torch.testing.assert_close(x, before, atol=0, rtol=0, equal_nan=True)
    check_performance(
        executor, enable_perf=(rows, width) in perf_shape and pattern == "random"
    )


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("width", [256, 4096, 262145])
def test_qsa_topk_graph_updates(topk, width):
    args = generate_data(8, width, topk, "random")
    score, lengths, _, starts = args
    check_graph_updates(
        qsa_topk,
        args,
        [
            lambda: score.neg_(),
            lambda: (score.nan_to_num_(), lengths.fill_(17)),
            lambda: starts.fill_(3),
        ],
        compare=lambda out: check(score, lengths, starts, out, topk),
    )


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
