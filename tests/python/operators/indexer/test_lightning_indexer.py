"""Validate the legacy TorchNPU LightningIndexer in the paged DSA layout."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_data(heads, context_length, page_size, topk):
    generator = torch.Generator(device="cpu").manual_seed(7)
    dtype = torch.bfloat16
    head_dim = 128
    pages = (context_length + page_size - 1) // page_size
    query = (
        torch.randn((1, heads, head_dim), generator=generator, device="cpu") * 0.01
    ).to(dtype)
    query[:, : heads // 2, :2] = torch.tensor([1, 0], dtype=dtype)
    query[:, heads // 2 :, :2] = torch.tensor([0, 1], dtype=dtype)
    key = (
        torch.randn(
            (pages + 1, page_size, 1, head_dim), generator=generator, device="cpu"
        )
        * 0.1
    ).to(dtype)
    physical_keys = key.flatten(0, 1)
    page_order = torch.randperm(pages, generator=generator, device="cpu") + 1
    logical = torch.randperm(context_length, generator=generator, device="cpu")[:topk]
    physical = page_order[logical // page_size] * page_size + logical % page_size
    # The chosen set wins with the model's head weights. Uniform head
    # weights would rank the other keys higher.
    physical_keys[..., :2] = torch.tensor([3, -1], dtype=dtype)
    physical_keys[physical, :, :2] = torch.tensor([-1, 2], dtype=dtype)
    weights = torch.full((1, heads), 1.5, dtype=dtype)
    weights[:, : heads // 2] = 0.25

    return (query.npu(), key.npu(), weights.npu()), dict(
        block_table=page_order.to(torch.int32).unsqueeze(0).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_key=torch.tensor([context_length], dtype=torch.int32).npu(),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=3,
    )


@pytest.mark.parametrize(
    "heads, context_length, page_size, topk",
    [pytest.param(64, 4093, 128, 2048, id="deepseek-v32-tp8-bf16-dsa-paged-decode")],
)
def test_lightning_indexer_paged_decode(heads, context_length, page_size, topk):
    import torch_npu
    from sgl_kernel_npu.operators.indexer import lightning_indexer
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    # The legacy binding supports A2/A3 and the 950DT SKU.
    if (
        get_hardware().generation is Generation.ARCH35
        and not torch.npu.get_device_name().upper().startswith("ASCEND950DT")
    ):
        pytest.skip("Legacy LightningIndexer requires a 950DT on arch35")
    if not callable(getattr(torch_npu, "npu_lightning_indexer", None)):
        pytest.skip("Installed TorchNPU lacks npu_lightning_indexer")

    args, kwargs = generate_data(heads, context_length, page_size, topk)
    executor = check_accuracy(
        lightning_indexer,
        args,
        kwargs=kwargs,
        result_transform=lambda result: (
            result[0].cpu().sort().values,
            result[1].cpu(),
        ),
    )
    check_performance(executor)
