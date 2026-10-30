"""Verify reordered pages, lengths, padding and caller-owned outputs."""

import pytest

torch = pytest.importorskip("torch")


def test_fia_prep():
    from sgl_kernel_npu.operators.attention import fia_prep

    ti = torch.tensor(
        [[0, 2, 1, -1], [1, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
    )
    lengths = torch.tensor([35, 18, 0], dtype=torch.int32)
    requests = torch.tensor([0, 1, 0], dtype=torch.int32)
    pages = torch.arange(128, dtype=torch.int32).reshape(2, 64)
    args = (ti, lengths, requests, pages, 16)
    expected = fia_prep(*args, impl="torch_native")
    table = torch.empty((3, 4), dtype=torch.int32)
    out_lengths = torch.empty((3,), dtype=torch.int32)
    actual = fia_prep(*args, table, out_lengths, impl="triton")
    assert actual[0] is table and actual[1] is out_lengths
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
