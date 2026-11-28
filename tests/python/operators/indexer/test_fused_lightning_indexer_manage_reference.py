"""Validate resident-plan semantics of the eager PyTorch LIM backend."""

import torch


def test_torch_native_cold_fill_eviction_refresh_and_short_reset():
    from sgl_kernel_npu.operators.indexer import fused_lightning_indexer_manage

    from tests.python.utils.common import require_test_impl

    require_test_impl(fused_lightning_indexer_manage, "torch_native")
    with torch.device("cpu"):
        width, length, capacity = 4, 8317, 8192
        pages = (length + 127) // 128
        query = torch.zeros(width, 32, 128, dtype=torch.bfloat16)
        query[..., 0] = 1
        weights = torch.ones(width, 32, dtype=query.dtype)
        keys = torch.zeros(pages, 128, 1, 128, dtype=query.dtype)
        key_values = keys.view(-1, 128)[:, 0]
        key_values[:2048] = 2
        key_values[length - width : length] = 16
        block_table = torch.arange(pages, dtype=torch.int32).view(1, -1)
        pool = torch.full((1, pages * 128), -1, dtype=torch.int32)
        topk_sources = torch.empty(width, 2048, dtype=torch.int32)
        topk_slots = torch.empty_like(topk_sources)
        topk_misses = torch.empty(width, dtype=torch.int32)
        miss_sources = torch.empty(1, 32768, dtype=torch.int32)
        miss_slots = torch.empty_like(miss_sources)
        miss_count = torch.empty(1, dtype=torch.int32)
        query_end = torch.tensor([width], dtype=torch.int32)
        actual = torch.tensor([length], dtype=torch.int32)
        budget = torch.tensor([capacity], dtype=torch.int32)
        state = torch.tensor([-2], dtype=torch.int32)
        entry = torch.zeros(1, dtype=torch.int32)

        def run():
            fused_lightning_indexer_manage(
                weights,
                query,
                keys,
                block_table,
                query_end,
                actual,
                actual,
                budget,
                state,
                entry,
                pool,
                topk_sources,
                topk_slots,
                topk_misses,
                miss_sources,
                miss_slots,
                miss_count,
                full_visible=True,
                impl="torch_native",
            )

        run()
        assert miss_count.item() == capacity
        assert (pool >= 0).sum().item() == capacity
        assert topk_misses.tolist() == [2048] * width
        for route, sources in enumerate(topk_sources):
            newest = length - width + route
            assert newest in sources
            assert sources.max() <= newest
        torch.testing.assert_close(topk_slots, pool[0, topk_sources.long()])

        new_source = length - width - 1
        assert pool[0, new_source] < 0
        key_values[new_source] = 32
        state.fill_(-1)
        run()
        rewritten = list(range(length - width, length))
        assert miss_count.item() == 1 + width
        assert miss_sources[0, : miss_count.item()].tolist() == [new_source, *rewritten]
        assert topk_misses.tolist() == [1] * width
        assert (topk_sources[:, 0] == new_source).all()
        torch.testing.assert_close(
            miss_slots[0, : miss_count.item()],
            pool[0, miss_sources[0, : miss_count.item()].long()],
        )
        assert len(pool[pool >= 0].unique()) == capacity

        actual.fill_(8)
        budget.zero_()
        state.fill_(-3)
        run()
        assert miss_count.item() == 0
        assert topk_misses.tolist() == [0] * width
        assert (topk_sources == topk_slots).all()
        assert (topk_sources >= 0).sum(dim=1).tolist() == [5, 6, 7, 8]
        assert pool[0, -1] == pages * 128 - 1
