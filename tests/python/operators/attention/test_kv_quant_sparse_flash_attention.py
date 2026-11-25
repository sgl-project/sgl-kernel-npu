"""Packed MLA sparse attention accuracy and performance on arch35."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

_PAGE_SIZE = 128
_TOPK = 2048
_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0.02, rtol=0.03),
    "torch.float32": AccuracyStandard(atol=0.02, rtol=0.02),
}
perf_shape = [(128, 4096, (2048,))]


def generate_data(heads, kv_len, selected_counts, device="npu"):
    generator = torch.Generator(device="cpu").manual_seed(929)
    tokens = len(selected_counts)
    pages = (kv_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    page_shape = (pages, _PAGE_SIZE, 1)

    def randn(shape):
        return torch.randn(shape, generator=generator, device="cpu")

    query = (randn((tokens, heads, 512)) * 0.5).bfloat16()
    query_rope = (randn((tokens, heads, 64)) * 0.5).bfloat16()
    latent = randn((*page_shape, 512)).to(torch.float8_e4m3fn)
    rope = (randn((*page_shape, 64)) * 0.4).bfloat16()
    scales = torch.rand((*page_shape, 4), generator=generator, device="cpu") * 0.4 + 0.1
    packed_kv = torch.cat(
        (latent.view(torch.uint8), rope.view(torch.uint8), scales.view(torch.uint8)),
        dim=-1,
    )
    page_ids = torch.arange(pages, device="cpu", dtype=torch.int32).roll(1)
    indices = torch.full((tokens, 1, _TOPK), -1, dtype=torch.int32, device="cpu")
    for row, count in enumerate(selected_counts):
        selection = torch.randperm(kv_len, generator=generator, device="cpu")[:count]
        indices[row, 0, :count] = selection.sort().values

    return {
        "query": query.to(device),
        "query_rope": query_rope.to(device),
        "packed_kv": packed_kv.to(device),
        "sparse_indices": indices.to(device),
        "block_table": page_ids.unsqueeze(0).to(device),
        "actual_seq_lengths_query": torch.tensor(
            [tokens], dtype=torch.int32, device=device
        ),
        "actual_seq_lengths_kv": torch.tensor(
            [kv_len], dtype=torch.int32, device=device
        ),
        "scale_value": 1.0 / 24,
    }


@pytest.mark.parametrize(
    "heads,kv_len,selected_counts",
    [
        # Representative GLM-5.2 DSA decode dimensions with a full 2048-key selection.
        pytest.param(*perf_shape[0], id="glm52-dcp-decode-topk2048-split-g"),
        # The three Q rows share a page table, including a cross-tile and an empty row.
        pytest.param(32, 260, (3, 129, 0), id="shared-page-table-short-tail-empty"),
    ],
)
def test_kv_quant_sparse_flash_attention(heads, kv_len, selected_counts):
    from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

    data = generate_data(heads, kv_len, selected_counts)
    executor = check_accuracy(
        kv_quant_sparse_flash_attention, kwargs=data, standard=_STANDARD
    )
    check_performance(
        executor, enable_perf=(heads, kv_len, selected_counts) in perf_shape
    )


@pytest.mark.extra_checks
def test_kv_quant_sparse_flash_attention_rejects_invalid_record_width():
    from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

    require_test_impl(kv_quant_sparse_flash_attention, "ascendc")
    data = generate_data(32, 1, (1,))
    data["packed_kv"] = data["packed_kv"][..., :-1].contiguous()
    with pytest.raises(RuntimeError, match="packed KV must be uint8"):
        kv_quant_sparse_flash_attention(**data, impl="ascendc")
