"""Check both aliasing and numerical effects of the in-place ScatterND API."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "coordinates",
    [[[6], [0], [3]], [[6, 2], [0, 4], [3, 1]]],
    ids=["slot-index", "two-dimensional-index"],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_scatter_nd_update_(coordinates, index_dtype):
    from sgl_kernel_npu.operators.tensor_ops import scatter_nd_update_

    destination = torch.zeros(7, 5, 17, dtype=torch.float16)
    indices = torch.tensor(coordinates, dtype=index_dtype)
    updates = torch.randn(
        (len(coordinates), *destination.shape[indices.shape[-1] :]),
        dtype=destination.dtype,
    )
    expected = destination.cpu()
    expected[tuple(indices.cpu().long().unbind(-1))] = updates.cpu()

    updated = scatter_nd_update_(destination, indices, updates, impl="torch_npu")

    assert updated.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
