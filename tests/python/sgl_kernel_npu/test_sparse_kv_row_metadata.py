"""Exact and dynamic graph checks for tensor-only sparse-KV row metadata."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.sparsity_driven_kv_offload import row_metadata as ops


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
@pytest.mark.parametrize("draft", [False, True])
def test_mtp_publication_replay(dtype, draft):
    torch.npu.set_device(0)
    requests = torch.tensor([0, 1, 2, 9], dtype=dtype, device="npu")
    lengths = torch.full((4,), 64, dtype=dtype, device="npu")
    positions = torch.arange(18, dtype=dtype, device="npu")
    locations = torch.arange(18, dtype=dtype, device="npu")
    counts = torch.tensor([3, 2, 1, 3], dtype=dtype, device="npu") if draft else None

    def reference():
        req = requests.long()[:, None].expand(4, 4).reshape(-1)
        pos = positions[:16].long()
        live = (req > 0) & (req < 4) & (pos >= 0) & (pos < 128)
        if counts is not None:
            live &= (torch.arange(4, device="npu")[None] < counts[:, None]).reshape(-1)
        live = torch.cat((live, torch.zeros(2, device="npu", dtype=torch.bool)))
        live &= (locations >= 0) & (locations < 32)
        pos = torch.cat((pos, torch.full((2,), -1, device="npu", dtype=torch.long)))
        return torch.arange(18, device="npu"), locations.long(), pos, live

    def candidate():
        return ops.mtp_rows(
            requests,
            lengths,
            positions,
            locations,
            counts,
            4,
            4,
            128,
            publication_rows=18,
            host_capacity=32,
        )

    exact(candidate(), reference())
    graph, outputs = capture(candidate)
    positions.add_(1)
    locations[4] = -1
    if counts is not None:
        counts.sub_(1)
    graph.replay()
    torch.npu.synchronize()
    exact(outputs, reference())


@pytest.mark.parametrize("size", [31, 32, 33, 64, 65])
def test_split_descriptors_replay(size):
    torch.npu.set_device(0)
    sources = [
        torch.empty((32, 1, width), device="npu", dtype=torch.bfloat16)
        for width in (512, 64)
    ]
    destinations = [torch.empty_like(tensor) for tensor in sources]
    source_rows = torch.arange(size, device="npu", dtype=torch.int32) % 32
    destination_rows = source_rows.flip(0).long()
    valid = torch.arange(size, device="npu") % 3 != 0

    def reference():
        sr = torch.where(valid, source_rows, 0).long()
        dr = torch.where(valid, destination_rows, 0).long()
        src, dst, lengths = [], [], []
        for source, destination in zip(sources, destinations):
            width = source.shape[-1] * source.element_size()
            src.append(source.data_ptr() + sr * width)
            dst.append(destination.data_ptr() + dr * width)
            lengths.append(valid.int() * width)
        return (
            torch.cat(src),
            torch.cat(dst),
            torch.cat(lengths),
            torch.tensor([2 * size], device="npu", dtype=torch.int32),
        )

    def candidate():
        return ops.copy_rows_metadata(
            sources, destinations, source_rows, destination_rows, valid
        )

    exact(candidate(), reference())
    graph, outputs = capture(candidate)
    source_rows.add_(1).remainder_(32)
    destination_rows.add_(3).remainder_(32)
    valid.logical_not_()
    graph.replay()
    torch.npu.synchronize()
    exact(outputs, reference())


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
        return ops.prefill_plan(
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
    source, destination, valid, mapping = ops.prefill_plan(
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
