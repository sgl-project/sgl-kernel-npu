"""Packed MLA sparse attention accuracy and performance on Ascend."""

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


def _hardware_kv_dtype():
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    return torch.uint8 if get_hardware().generation == Generation.ARCH35 else torch.int8


def _require_kv_dtype(kv_dtype):
    if kv_dtype != _hardware_kv_dtype():
        pytest.skip(f"{kv_dtype} packed KV is unsupported on this architecture")


def generate_data(
    heads, kv_len, selected_counts, device="npu", *, kv_dtype=torch.uint8
):
    generator = torch.Generator(device="cpu").manual_seed(929)
    tokens = len(selected_counts)
    pages = (kv_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    page_shape = (pages, _PAGE_SIZE, 1)

    def randn(shape):
        return torch.randn(shape, generator=generator, device="cpu")

    query = (randn((tokens, heads, 512)) * 0.5).bfloat16()
    query_rope = (randn((tokens, heads, 64)) * 0.5).bfloat16()
    latent = randn((*page_shape, 512))
    if kv_dtype == torch.int8:
        latent = (latent * 32).round().clamp(-128, 127).to(torch.int8)
    else:
        latent = latent.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = (randn((*page_shape, 64)) * 0.4).bfloat16()
    scales = torch.rand((*page_shape, 4), generator=generator, device="cpu") * 0.4 + 0.1
    if kv_dtype == torch.int8:
        scales /= 32
    packed_kv = torch.cat(
        (latent, rope.view(kv_dtype), scales.view(kv_dtype)),
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


@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.int8], ids=["e4m3", "int8"])
@pytest.mark.parametrize(
    "heads,kv_len,selected_counts",
    [
        # Representative GLM-5.2 DSA decode dimensions with a full 2048-key selection.
        pytest.param(*perf_shape[0], id="glm52-dcp-decode-topk2048-split-g"),
        # The three Q rows share a page table, including a cross-tile and an empty row.
        pytest.param(32, 260, (3, 129, 0), id="shared-page-table-short-tail-empty"),
    ],
)
def test_kv_quant_sparse_flash_attention(heads, kv_len, selected_counts, kv_dtype):
    from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

    _require_kv_dtype(kv_dtype)
    data = generate_data(heads, kv_len, selected_counts, kv_dtype=kv_dtype)
    executor = check_accuracy(
        kv_quant_sparse_flash_attention, kwargs=data, standard=_STANDARD
    )
    check_performance(
        executor,
        enable_perf=kv_dtype == torch.uint8
        and (heads, kv_len, selected_counts) in perf_shape,
    )


@pytest.mark.extra_checks
def test_kv_quant_sparse_flash_attention_rejects_invalid_record_width():
    from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

    require_test_impl(kv_quant_sparse_flash_attention, "ascendc")
    data = generate_data(32, 1, (1,), kv_dtype=_hardware_kv_dtype())
    data["packed_kv"] = data["packed_kv"][..., :-1].contiguous()
    with pytest.raises(RuntimeError, match="packed KV must be uint8"):
        kv_quant_sparse_flash_attention(**data, impl="ascendc")


def _rope0_inputs(kv_dtype):
    padded = generate_data(8, 260, (2, 129, 0), kv_dtype=kv_dtype)
    padded["query_rope"].zero_()
    packed = padded["packed_kv"]
    rope = packed[..., 512:640]
    rope.zero_()
    compact = dict(
        padded,
        query_rope=padded["query_rope"][..., :0],
        packed_kv=torch.cat((packed[..., :512], packed[..., 640:]), dim=-1),
    )
    return padded, compact


@pytest.mark.parametrize("return_softmax_lse", [True, False])
@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.int8], ids=["e4m3", "int8"])
def test_kv_quant_sparse_flash_attention_rope0(return_softmax_lse, kv_dtype):
    from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

    require_test_impl(kv_quant_sparse_flash_attention, "ascendc")
    _require_kv_dtype(kv_dtype)
    padded, compact = _rope0_inputs(kv_dtype)
    options = {"return_softmax_lse": return_softmax_lse}
    padded_output = kv_quant_sparse_flash_attention(**padded, **options)
    compact_output = kv_quant_sparse_flash_attention(**compact, **options)
    torch.testing.assert_close(compact_output, padded_output, rtol=0, atol=0)
    if not return_softmax_lse:
        assert compact_output[1].numel() == compact_output[2].numel() == 0
    check_accuracy(
        kv_quant_sparse_flash_attention,
        kwargs=compact | options,
        standard=_STANDARD,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.int8], ids=["e4m3", "int8"])
def test_kv_quant_sparse_flash_attention_rope0_graph_replay(kv_dtype):
    from sgl_kernel_npu.operators.attention import kv_quant_sparse_flash_attention

    require_test_impl(kv_quant_sparse_flash_attention, "ascendc")
    _require_kv_dtype(kv_dtype)
    _, compact = _rope0_inputs(kv_dtype)

    def run():
        return kv_quant_sparse_flash_attention(**compact, return_softmax_lse=False)[0]

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    compact["sparse_indices"][0, 0, 0] = 128
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output, run(), rtol=0, atol=0)
