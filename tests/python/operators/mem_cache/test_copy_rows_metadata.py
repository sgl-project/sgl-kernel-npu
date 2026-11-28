"""Numerical and graph replay checks for copy rows metadata."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.operators.mem_cache.copy_rows_metadata import copy_rows_metadata


@pytest.fixture(autouse=True)
def cpu_reference_device():
    with torch.device("cpu"):
        yield


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
        return copy_rows_metadata(
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
