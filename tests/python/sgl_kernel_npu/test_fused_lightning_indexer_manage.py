"""LIM metadata correctness and replay using only the normal kernel wheel."""

import pytest
import torch

pytest.importorskip("torch_npu")
import sgl_kernel_npu  # noqa: F401
from sgl_kernel_npu.sparsity_driven_kv_offload import fused_lightning_indexer_manage
from utils import require_npu_op

pytestmark = require_npu_op("fused_lightning_indexer_manage")


def make_case(batch=1, heads=32, dtype=torch.bfloat16):
    torch.npu.set_device(0)
    device = "npu:0"
    capacity, prefix, source_capacity = 4096, 8192, 8320
    query = torch.zeros(batch, heads, 128, dtype=dtype, device=device)
    query[..., 0] = 1
    weights = torch.full((batch, heads), 1 / heads, dtype=dtype, device=device)
    key = torch.zeros(source_capacity // 128, 128, 1, 128, dtype=dtype, device=device)
    table = torch.arange(
        source_capacity // 128, dtype=torch.int32, device=device
    ).repeat(batch, 1)
    vector = lambda value: torch.full((batch,), value, dtype=torch.int32, device=device)
    pool = torch.full((batch, source_capacity), -1, dtype=torch.int32, device=device)
    ids = torch.empty(batch, 1, 2048, dtype=torch.int32, device=device)
    misses = torch.empty(batch, 32768, dtype=torch.int32, device=device)
    args = [
        weights,
        torch.ones(batch, heads, dtype=torch.float32, device=device),
        query,
        torch.ones(key.shape[:3], dtype=torch.float32, device=device),
        key,
        table,
        torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        vector(prefix + 1),
        vector(prefix),
        vector(capacity),
        vector(-2),
        torch.arange(batch, dtype=torch.int32, device=device),
        pool,
        ids,
        torch.empty_like(ids),
        vector(0),
        misses,
        torch.empty_like(misses),
        vector(0),
    ]
    return args, torch.full((batch, capacity), -1, dtype=torch.int32)


def set_hotset(args, offset):
    # Exactly K scores are positive, with a wide gap at the TopK boundary.
    # Their internal order is immaterial to the metadata/payload contract.
    keys = args[4].view(-1, 128)
    keys[:, 0] = -8
    keys[offset : offset + 2048, 0] = 8


def consume_plan(args, resident, offset):
    torch.npu.synchronize()
    ids, slots = args[13].cpu()[:, 0], args[14].cpu()[:, 0]
    src, dst, counts = args[16].cpu(), args[17].cpu(), args[18].cpu()
    expected = torch.arange(offset, offset + 2048, dtype=torch.int32)
    for row in range(ids.shape[0]):
        count = int(counts[row])
        assert 0 <= count <= resident.shape[1]
        assert torch.all(
            (dst[row, :count] >= 0) & (dst[row, :count] < resident.shape[1])
        )
        assert dst[row, :count].unique().numel() == count
        resident[row, dst[row, :count].long()] = src[row, :count]
        torch.testing.assert_close(ids[row].sort().values, expected, rtol=0, atol=0)
        assert torch.all((slots[row] >= 0) & (slots[row] < resident.shape[1]))
        # The actual miss plan must reconstruct every returned source/slot pair.
        torch.testing.assert_close(
            resident[row, slots[row].long()], ids[row], rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("batch", [1, 4])
def test_cold_warm_miss_and_request_reuse(dtype, heads, batch):
    args, resident = make_case(batch, heads, dtype)
    set_hotset(args, 0)
    fused_lightning_indexer_manage(*args)
    consume_plan(args, resident, 0)
    args[10].fill_(-1)
    set_hotset(args, 4096)
    fused_lightning_indexer_manage(*args)
    consume_plan(args, resident, 4096)
    # Reuse the request row while retaining its previous encoded pool contents.
    args[10].fill_(-2)
    resident.fill_(-1)
    set_hotset(args, 2048)
    fused_lightning_indexer_manage(*args)
    consume_plan(args, resident, 2048)


def test_graph_replay_reads_current_state_and_keys():
    args, resident = make_case()
    set_hotset(args, 0)
    fused_lightning_indexer_manage(*args)  # warm the persistent tiling cache
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fused_lightning_indexer_manage(*args)
    args[12].fill_(-1)
    args[10].fill_(-2)
    graph.replay()
    consume_plan(args, resident, 0)
    args[10].fill_(-1)
    set_hotset(args, 4096)
    graph.replay()
    consume_plan(args, resident, 4096)


def test_non_offload_short_row_and_invalid_dtype():
    args, _ = make_case()
    args[7].fill_(384)
    args[10].fill_(-3)
    fused_lightning_indexer_manage(*args)
    torch.npu.synchronize()
    ids, slots = args[13].cpu().view(-1), args[14].cpu().view(-1)
    valid = ids >= 0
    torch.testing.assert_close(
        ids[valid].sort().values, torch.arange(384, dtype=torch.int32)
    )
    torch.testing.assert_close(slots, ids)
    assert args[15].item() == args[18].item() == 0
    args[2] = args[2].float()
    with pytest.raises(RuntimeError, match="fp16 or bf16"):
        fused_lightning_indexer_manage(*args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
