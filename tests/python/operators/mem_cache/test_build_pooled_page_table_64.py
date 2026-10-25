import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1,), id="rank1-width1"),
        pytest.param((3,), id="rank1-width3"),
        pytest.param((4,), id="rank1-width4"),
        pytest.param((5,), id="rank1-width5"),
        pytest.param((65,), id="rank1-width65"),
        pytest.param((2, 3), id="rank2"),
        pytest.param((2, 3, 4), id="rank3"),
        pytest.param((0, 5), id="empty-leading-dim"),
        pytest.param((2, 0, 5), id="rank3-empty-leading-dim"),
    ],
)
def test_build_pooled_page_table_64(shape):
    from sgl_kernel_npu.operators.mem_cache import build_pooled_page_table_64

    torch.manual_seed(42)
    pool_size = 4
    page_table_64 = torch.randint(0, 100000, shape, dtype=torch.int64)
    original = page_table_64.clone()

    # Independent slicing reference: select the first token page in each group.
    expected = original[..., ::pool_size]
    actual = build_pooled_page_table_64(page_table_64, pool_size=pool_size)

    assert_accuracy(actual, expected)
    assert_accuracy(page_table_64, original)
