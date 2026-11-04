"""A5-only eager/replay parity for the three AICPU scheduling operators."""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.npu
@pytest.mark.parametrize("kind", ["mqsmla", "qli", "qsli"])
def test_metadata_replay_reads_updated_lengths(kind):
    from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata
    from sgl_kernel_npu.operators.indexer import (
        quant_lightning_indexer_metadata,
        quant_sparse_lightning_indexer_metadata,
    )

    device = torch.device("npu", torch.npu.current_device())
    cu_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    lengths = torch.tensor([128, 256], dtype=torch.int32, device=device)
    candidates = torch.tensor([[4], [8]], dtype=torch.int32, device=device)
    window = torch.tensor([[64], [128]], dtype=torch.int32, device=device)

    def invoke():
        if kind == "mqsmla":
            return mixed_quant_sparse_flash_mla_metadata(
                ori_topk_length=window,
                cmp_topk_length=lengths.view(2, 1),
                cu_seqlens_q=cu_q,
                num_heads_q=64,
                num_heads_kv=1,
                head_dim=512,
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=True,
            )
        common = dict(
            cu_seqlens_q=cu_q,
            seqused_k=lengths,
            batch_size=2,
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            max_seqlen_q=1,
            max_seqlen_k=4096,
            topk=512,
            mask_mode=3,
            cmp_ratio=1,
            layout_q="TND",
            layout_k="PA_BBND",
        )
        if kind == "qli":
            return quant_lightning_indexer_metadata(**common)
        return quant_sparse_lightning_indexer_metadata(
            candidate_block_length=candidates,
            quant_mode=1,
            candidate_block_size=8,
            **common,
        )

    # Compile and allocate once before capture; this test does not time warmup.
    baseline = invoke().clone()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = invoke()
    pointer = captured.data_ptr()
    changed = False
    # Cross a scheduling boundary; short lengths alone can share one metadata
    # plan and would let a stale captured result appear correct.
    for value in (64, 4096, 192):
        lengths.fill_(value)
        candidates.fill_(value // 8)
        window.fill_(min(value, 128))
        expected = invoke()
        graph.replay()
        torch.npu.synchronize()
        assert captured.data_ptr() == pointer
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        changed |= not torch.equal(expected, baseline)
    assert changed, "Test lengths must produce different eager scheduling plans"
