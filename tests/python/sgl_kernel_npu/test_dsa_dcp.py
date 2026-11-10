import pytest
import sgl_kernel_npu  # noqa: F401
import torch


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("page_size", [1, 128])
def test_dcp_sparse_index_remap(dcp_size, page_size):
    op = getattr(torch.ops.npu, "sgl_dcp_sparse_index_remap", None)
    if op is None:
        pytest.skip("DSA DCP remap requires the A2/A3 kernel build")
    generator = torch.Generator().manual_seed(123)
    indices = torch.randint(
        -1, 131072, (6, 2048), generator=generator, dtype=torch.int32
    )
    indices[0].fill_(-1)
    indices[1, ::2] = -1
    indices[2] = torch.arange(2048).flip(0)
    device_indices = indices.npu()

    for rank in range(dcp_size):
        expected = torch.full_like(indices, -1)
        for i, row in enumerate(indices):
            pages = row // page_size
            keep = (row >= 0) & (pages % dcp_size == rank)
            remapped = (pages[keep] // dcp_size) * page_size + row[keep] % page_size
            expected[i, : remapped.numel()] = remapped
        assert torch.equal(
            op(device_indices, dcp_size, rank, page_size).cpu(), expected
        )


@pytest.mark.parametrize("dcp_size,page_size,query_len", [(4, 128, 4), (2, 64, 1)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_dcp_verify_page_tables(dcp_size, page_size, query_len, index_dtype):
    op = getattr(torch.ops.npu, "sgl_dcp_verify_page_tables", None)
    if op is None:
        pytest.skip("DSA DCP page tables require the A2/A3 kernel build")
    cols = 257
    slots = torch.arange(5 * cols * page_size, dtype=torch.int32).reshape(5, -1)
    slots[0].zero_()
    requests = torch.tensor([3, 0, 4, 3], dtype=index_dtype)
    local_cols = (cols + dcp_size - 1) // dcp_size + 1
    ordinary = torch.empty((4, cols), dtype=torch.int32, device="npu")
    local = torch.empty((4 * query_len, local_cols), dtype=torch.int32, device="npu")
    device_slots, device_requests = slots.npu(), requests.npu()
    # Reuse the same output buffers and shrink the active range to test tail clearing.
    for active in (0, 1, 127, 128, 129, cols, 5):
        op(
            device_slots,
            device_requests,
            ordinary,
            local,
            active,
            query_len,
            page_size,
            dcp_size,
        )
        expected = torch.zeros(ordinary.shape, dtype=torch.int32)
        expected[:, :active] = (
            slots[requests, : active * page_size : page_size] // page_size
        )
        expected_local = torch.zeros(local.shape, dtype=torch.int32)
        partial = (expected[:, :active:dcp_size] // dcp_size).repeat_interleave(
            query_len, dim=0
        )
        expected_local[:, : partial.size(1)] = partial
        assert torch.equal(ordinary.cpu(), expected)
        assert torch.equal(local.cpu(), expected_local)
