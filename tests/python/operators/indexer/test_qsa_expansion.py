"""Blocks[R,K] + positions/lengths[R] -> compact int32 token indices.

Independent scalar reference. Exact equality, including tail placement and -1
padding. K=512/2048; the current model uses ratio=4 and output width 2051.
"""

import pytest
import torch
from sgl_kernel_npu.operators.indexer import expand_qsa_blocks
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance

pytestmark = pytest.mark.skipif(
    get_hardware().generation != Generation.ARCH22,
    reason="These model operator contracts are validated only on arch22",
)

perf_shape = [(1, 512, 4), (128, 512, 4)]


def generate_data(rows, k, ratio, mode):
    columns = torch.arange(k, dtype=torch.int32)[None, :]
    count = torch.arange(rows, dtype=torch.int32) % (k + 1)
    if mode == "full":
        count.fill_(k)
    elif mode == "empty":
        count.zero_()
    blocks = torch.where(columns < count[:, None], columns, -1).contiguous()
    lengths = count * ratio + torch.arange(rows, dtype=torch.int32) % ratio
    if mode == "empty":
        lengths.zero_()
    positions = lengths - 1
    return blocks, positions, lengths, ratio, k * ratio


@pytest.mark.parametrize(
    "rows",
    [0, 1, 8, 127, 128, 129],
    ids=[
        "empty",
        "single-token",
        "decode-batch",
        "before-row-boundary",
        "row-boundary",
        "row-tail",
    ],
)
@pytest.mark.parametrize(
    "k,ratio",
    [(512, 4), (2048, 4), (512, 3)],
    ids=["model-k512-ratio4", "large-k2048-ratio4", "ratio3-edge"],
)
@pytest.mark.parametrize("mode", ["full", "short", "empty"])
def test_qsa_expansion_accuracy(rows, k, ratio, mode):
    # At ratio 4, block 2 expands to token indices [8, 9, 10, 11].
    args = generate_data(rows, k, ratio, mode)
    saved = [x.clone() for x in args[:3]]
    executor = check_accuracy(expand_qsa_blocks, args=args)
    assert executor().is_contiguous()
    _assert_unchanged(args, saved)
    check_performance(
        executor, enable_perf=(rows, k, ratio) in perf_shape and mode == "short"
    )


@pytest.mark.parametrize("rows", [8, 128], ids=["decode-batch", "row-boundary"])
def test_qsa_expansion_graph_updates(rows):
    args = generate_data(rows, 512, 4, "short")
    replacement = generate_data(rows, 512, 4, "full")

    def update():
        for x, new in zip(args[:3], replacement[:3]):
            x.copy_(new)

    _check_graph_updates(expand_qsa_blocks, args, [update])


def test_qsa_expansion_short_tail_and_strided_inputs():
    args = list(generate_data(2, 512, 4, "short"))
    args[0][:, 0] = 0
    args[1].fill_(6)
    args[2].fill_(7)
    args[:3] = [torch.stack((x, x), -1)[..., 0] for x in args[:3]]
    check_accuracy(expand_qsa_blocks, args=args)
    out = expand_qsa_blocks(*args)
    torch.testing.assert_close(
        out[:, :7], torch.arange(7, dtype=torch.int32).expand(2, -1), atol=0, rtol=0
    )
    assert (out[:, 7:] == -1).all()


@pytest.mark.extra_checks
@pytest.mark.parametrize("bad", ["ratio", "topk", "dtype", "shape"])
def test_qsa_expansion_rejects_metadata(bad):
    args = list(generate_data(2, 512, 4, "short"))
    if bad == "ratio":
        args[3] = 1
    elif bad == "topk":
        args[4] = 2047
    elif bad == "dtype":
        args[0] = args[0].float()
    else:
        args[1] = args[1][:1]
    with pytest.raises(ValueError):
        expand_qsa_blocks(*args)


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
