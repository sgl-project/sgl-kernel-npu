"""Validate the legacy TorchNPU LightningIndexer in the paged DSA layout."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "heads,context_length,page_size,topk",
    [pytest.param(64, 4096, 128, 2048, id="deepseek-v32-dsa-paged-decode")],
)
def test_lightning_indexer_paged_decode(heads, context_length, page_size, topk):
    import torch_npu
    from sgl_kernel_npu.operators.indexer import lightning_indexer
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    # The vendor lists A2/A3 and 950DT, rather than the entire ARCH35 family.
    # This case covers the A2/A3 DSA workload without assuming a 950 SKU.
    # https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer.md
    if get_hardware().generation is not Generation.ARCH22:
        pytest.skip("This legacy LightningIndexer test targets Atlas A2/A3")
    if not callable(getattr(torch_npu, "npu_lightning_indexer", None)):
        pytest.skip("Installed TorchNPU lacks npu_lightning_indexer")

    generator = torch.Generator(device="cpu").manual_seed(7)
    dtype = torch.bfloat16
    head_dim = 128
    pages = context_length // page_size
    query = (
        torch.randn((1, heads, head_dim), generator=generator, device="cpu") * 0.01
    ).to(dtype)
    query[..., 0] = 1
    key = (
        torch.randn((pages, page_size, 1, head_dim), generator=generator, device="cpu")
        * 0.1
    ).to(dtype)
    physical_keys = key.flatten(0, 1)
    physical_keys[..., 0] = -2
    # Separate the chosen keys from the rest at the top-k boundary even
    # after BF16 rounding. Tie order inside the selected set is immaterial.
    positive = torch.randperm(context_length, generator=generator, device="cpu")[:topk]
    physical_keys[positive, :, 0] = 2
    weights = torch.linspace(0.5, 1.5, heads, device="cpu").to(dtype).unsqueeze(0)
    page_order = torch.randperm(pages, generator=generator, device="cpu")

    logical_keys = key[page_order].reshape(context_length, head_dim)
    similarities = (query[0].float() @ logical_keys.float().T).relu()
    scores = (similarities * weights[0, :, None].float()).sum(dim=0)
    expected = scores.topk(topk).indices.sort().values

    indices, values = lightning_indexer(
        query.npu(),
        key.npu(),
        weights.npu(),
        block_table=page_order.to(torch.int32).unsqueeze(0).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([context_length], dtype=torch.int32),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=3,
        impl="torch_npu",
    )
    assert indices.shape == (1, 1, topk)
    assert indices.dtype == torch.int32
    assert values.numel() == 0
    torch.testing.assert_close(indices.cpu().flatten().long().sort().values, expected)
