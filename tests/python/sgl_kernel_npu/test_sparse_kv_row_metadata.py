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


def test_ragged_plan_and_duplicate_mapping_replay():
    torch.npu.set_device(0)
    table = torch.arange(80, device="npu", dtype=torch.int32).reshape(5, 16)
    requests = torch.tensor([1, -1, 2], device="npu", dtype=torch.int32)
    lengths = torch.tensor([9, 0, 6], device="npu", dtype=torch.int32)
    source, destination, valid = ops.prefill_plan(
        requests, lengths, table, 15, 9, 5, 16
    )
    exact(
        (source, destination, valid),
        (
            torch.cat((table[1, :9], table[2, :6])).long(),
            torch.arange(15, device="npu"),
            torch.ones(15, device="npu", dtype=torch.bool),
        ),
    )
    metadata = torch.tensor([2, 4, 6, 0, 9, 9, 7, 0, 4, 2, 2, 6], device="npu")
    locations = torch.tensor([23, 24, -1, -1, 36, 37], device="npu")
    current_valid = torch.ones(6, device="npu", dtype=torch.bool)

    def candidate():
        return ops.prefill_current_mapping(
            source, locations, current_valid, metadata, 80, 15, 3
        )

    def reference(duplicate=False):
        dst = torch.tensor([7, 8, 0, 0, 13, 14], device="npu")
        live = torch.tensor(
            [not duplicate, True, False, False, True, not duplicate], device="npu"
        )
        host = torch.ones(15, device="npu", dtype=torch.bool)
        host[dst[live]] = False
        return torch.arange(6, device="npu"), dst, live, host

    exact(candidate(), reference())
    graph, outputs = capture(candidate)
    locations[5] = locations[0]
    graph.replay()
    torch.npu.synchronize()
    exact(outputs, reference(duplicate=True))
