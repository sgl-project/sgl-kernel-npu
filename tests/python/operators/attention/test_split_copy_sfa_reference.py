"""Check the eager split CopySFA backend against explicit KV payloads."""

import pytest
import torch


@pytest.mark.parametrize(
    "mode",
    ["legacy", "full_visible", "resident_refresh"],
    ids=["legacy", "full", "refresh"],
)
def test_torch_native_copies_host_rows_and_applies_causal_attention(mode):
    from sgl_kernel_npu.operators.attention import (
        fused_scatter_copy_sparse_flash_attention_split as copy_sfa,
    )

    from tests.python.utils.common import require_test_impl

    require_test_impl(copy_sfa, "torch_native")
    with torch.device("cpu"):
        dtype = torch.bfloat16
        query = torch.zeros((2, 8, 512), dtype=dtype)
        query_rope = torch.zeros((2, 8, 64), dtype=dtype)
        ckv = torch.zeros((17, 128, 1, 512), dtype=dtype)
        rope = torch.zeros((17, 128, 1, 64), dtype=dtype)
        hbm_pages = torch.arange(17, dtype=torch.int32).view(1, -1)
        host_pages = torch.tensor([[1, 0]], dtype=torch.int32)
        host_ckv = torch.zeros((256, 512), dtype=dtype)
        host_rope = torch.zeros((256, 64), dtype=dtype)
        host_ckv[133].fill_(12)
        host_rope[133].fill_(5)
        ckv.view(-1, 512)[2048:2051].fill_(4)
        selected = torch.arange(2048, dtype=torch.int32).repeat(2, 1)
        sources = selected.clone()
        sources[:, 0] = 5
        miss_sources = torch.tensor([[5]], dtype=torch.int32)
        miss_slots = torch.zeros_like(miss_sources)
        staging = {}
        if mode == "resident_refresh":
            # Source 2049 does not have a host page in this fixture. Its
            # retained slot must be refreshed from the current query window.
            miss_sources = torch.tensor([[5, 2049]], dtype=torch.int32)
            miss_slots = torch.tensor([[0, 1]], dtype=torch.int32)
            sources[:, 1] = 2049
            staging = {
                "current": torch.full((2, 1, 576), 7, dtype=dtype),
                "positions": torch.tensor([2049, 2050], dtype=torch.int64),
                "valid": torch.tensor([True, True]),
                "request_slots": torch.tensor([1], dtype=torch.int32),
                "req_to_token": torch.zeros((2, 2176), dtype=torch.int32),
                "resident_end": torch.tensor([-1, 128], dtype=torch.int32),
                "resident_refresh_tail": True,
            }
        attention = torch.empty_like(query)
        assert (
            copy_sfa(
                query,
                query_rope,
                ckv,
                rope,
                torch.tensor([2], dtype=torch.int32),
                torch.tensor([2051], dtype=torch.int32),
                torch.tensor([2048], dtype=torch.int32),
                sources,
                selected,
                torch.ones(2, dtype=torch.int32),
                miss_sources,
                miss_slots,
                torch.tensor([miss_sources.size(1)], dtype=torch.int32),
                hbm_pages,
                host_pages,
                0,
                attention,
                1.0,
                full_visible=mode != "legacy",
                dram_ckv=host_ckv,
                dram_rope=host_rope,
                impl="torch_native",
                **staging,
            )
            is None
        )
        torch.testing.assert_close(ckv[0, 0, 0], host_ckv[133], atol=0, rtol=0)
        torch.testing.assert_close(rope[0, 0, 0], host_rope[133], atol=0, rtol=0)
        if mode == "resident_refresh":
            assert staging["resident_end"].tolist() == [-1, -1]
            assert (ckv[0, 1] == 7).all() and (rope[0, 1] == 7).all()
            expected = (19 / 2048,) * 2
        elif mode == "full_visible":
            expected = (12 / 2048,) * 2
        else:
            expected = (20 / 2050, 24 / 2051)
        for route, value in enumerate(expected):
            torch.testing.assert_close(
                attention[route],
                torch.full_like(attention[route], value),
                atol=0,
                rtol=0,
            )


def test_torch_native_restores_short_history_and_stages_current_in_split_cache():
    from sgl_kernel_npu.operators.attention import (
        fused_scatter_copy_sparse_flash_attention_split as copy_sfa,
    )

    from tests.python.utils.common import require_test_impl

    require_test_impl(copy_sfa, "torch_native")
    with torch.device("cpu"):
        dtype = torch.float16
        query = torch.zeros((2, 8, 512), dtype=dtype)
        query_rope = torch.zeros((2, 8, 64), dtype=dtype)
        ckv = torch.zeros((2, 128, 1, 512), dtype=dtype)
        rope = torch.zeros((2, 128, 1, 64), dtype=dtype)
        hbm_pages = torch.tensor([[1, 0]], dtype=torch.int32)
        host_ckv = torch.full((256, 512), 9, dtype=dtype)
        host_rope = torch.full((256, 64), 6, dtype=dtype)
        current = torch.stack(
            (torch.full((1, 576), 2, dtype=dtype), torch.full((1, 576), 5, dtype=dtype))
        )
        host_ckv[19].fill_(3)
        request_map = torch.zeros((2, 256), dtype=torch.int32)
        request_map[1, 126] = 19
        resident_end = torch.tensor([-1, 126], dtype=torch.int32)
        empty_topk = torch.full((2, 2048), -1, dtype=torch.int32)
        empty_plan = torch.full((1, 1), -1, dtype=torch.int32)
        attention = torch.empty_like(query)
        copy_sfa(
            query,
            query_rope,
            ckv,
            rope,
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([129], dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            empty_topk,
            empty_topk,
            torch.zeros(2, dtype=torch.int32),
            empty_plan,
            empty_plan,
            torch.zeros(1, dtype=torch.int32),
            hbm_pages,
            torch.tensor([[0, 1]], dtype=torch.int32),
            0,
            attention,
            1.0,
            full_visible=True,
            current=current,
            positions=torch.tensor([127, 128]),
            valid=torch.tensor([True, True]),
            request_slots=torch.tensor([1], dtype=torch.int32),
            req_to_token=request_map,
            resident_end=resident_end,
            dram_ckv=host_ckv,
            dram_rope=host_rope,
            impl="torch_native",
        )
        assert resident_end.tolist() == [-1, 129]
        torch.testing.assert_close(ckv[1, 126, 0], host_ckv[19])
        torch.testing.assert_close(rope[1, 126, 0], host_rope[19])
        torch.testing.assert_close(ckv[1, 127, 0], current[0, 0, :512])
        torch.testing.assert_close(rope[0, 0, 0], current[1, 0, 512:])
        torch.testing.assert_close(attention[0], torch.full_like(attention[0], 5 / 128))
        torch.testing.assert_close(
            attention[1], torch.full_like(attention[1], 10 / 129)
        )
