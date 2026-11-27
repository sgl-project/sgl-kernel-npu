"""Numerical and graph replay checks for prefill plan."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.sparsity_driven_kv_offload.row_metadata import prefill_plan


def exact(actual, expected):
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)


def capture(fn):
    warm = torch.npu.Stream()
    warm.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(warm):
        for _ in range(3):
            fn()
    torch.npu.current_stream().wait_stream(warm)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = fn()
    return graph, outputs


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("padded", [False, True])
def test_prefill_plan_current_replay(dtype, padded):
    torch.npu.set_device(0)
    table = torch.arange(80, device="npu", dtype=dtype).reshape(5, 16)
    requests = torch.tensor([1, 0, -1, 2], device="npu", dtype=dtype)
    lengths = torch.tensor([4, 2, 0, 3], device="npu", dtype=dtype)
    widths = [3, 3, 3, 3] if padded else [2, 1, 0, 3]
    starts = [0, widths[0], sum(widths[:2]), sum(widths[:3])]
    metadata = torch.tensor(
        starts + [2, 1, 0, 0] + [2, 1, 0, 3] + widths, device="npu", dtype=torch.int64
    )
    current_valid = torch.ones(sum(widths), device="npu", dtype=torch.bool)
    current_valid[starts[3] + 1] = False

    def candidate():
        return prefill_plan(
            requests,
            lengths,
            table,
            9,
            4,
            5,
            16,
            current_metadata=metadata,
            current_valid=current_valid,
            max_current_width=max(widths),
        )

    def check(result):
        source, destination, valid, mapping = result
        src, dst, live, host = mapping
        expected_source = torch.cat((table[1, :4], table[0, :2], table[2, :3])).long()
        torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        current = torch.arange(sum(widths), device="npu") + 1000
        host_data = torch.arange(80, device="npu") + 100
        actual = torch.empty(9, device="npu", dtype=torch.int64)
        actual[destination[valid & host]] = host_data[source[valid & host]]
        actual[dst[live]] = current[src[live]]
        reqs, lens = requests.cpu().tolist(), lengths.cpu().tolist()
        meta, cv = metadata.cpu().tolist(), current_valid.cpu().tolist()
        expected = []
        for r, (req, length) in enumerate(zip(reqs, lens)):
            if req < 0:
                continue
            start, prefix, extend = meta[r], meta[4 + r], meta[8 + r]
            for pos in range(length):
                idx = start + pos - prefix
                if req > 0 and prefix <= pos < prefix + extend and cv[idx]:
                    expected.append(1000 + idx)
                else:
                    expected.append(100 + req * 16 + pos)
        torch.testing.assert_close(actual.cpu(), torch.tensor(expected), rtol=0, atol=0)

    check(candidate())
    graph, outputs = capture(candidate)
    current_valid.logical_not_()
    metadata[4] = 3
    metadata[8] = 1
    graph.replay()
    torch.npu.synchronize()
    check(outputs)


def test_prefill_zero_total_initializes_padding():
    torch.npu.set_device(0)
    requests = torch.tensor([0, -1], device="npu")
    lengths = torch.zeros(2, device="npu", dtype=torch.int64)
    table = torch.zeros((2, 4), device="npu", dtype=torch.int32)
    metadata = torch.tensor([0, 2, 0, 0, 0, 0, 2, 2], device="npu")
    current_valid = torch.ones(4, device="npu", dtype=torch.bool)
    source, destination, valid, mapping = prefill_plan(
        requests,
        lengths,
        table,
        0,
        0,
        2,
        4,
        current_metadata=metadata,
        current_valid=current_valid,
        max_current_width=2,
    )
    assert source.numel() == destination.numel() == valid.numel() == 0
    src, dst, current_live, host = mapping
    exact(
        (src, dst, current_live),
        (
            torch.arange(4),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.bool),
        ),
    )
    assert host.numel() == 0
