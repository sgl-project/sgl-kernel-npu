"""On-device production/reference checks for attention operators."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def test_a5_cache_writer_and_mixed_attention(impl_pair):
    from sgl_kernel_npu.operators.attention import (
        mixed_quant_sparse_flash_mla,
        mixed_quant_sparse_flash_mla_metadata,
    )
    from sgl_kernel_npu.operators.kvcacheio import (
        get_npu_kv_quant_path,
        kv_compress_epilog,
    )
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH35:
        pytest.skip("mixed-quant reader/writer requires A5")
    torch.manual_seed(13)
    x = torch.randn(128, 512, device="npu", dtype=torch.bfloat16)
    q = torch.randn(1, 64, 512, device="npu", dtype=torch.bfloat16) * 0.1
    slots = torch.arange(128, dtype=torch.int32, device="npu")
    slots[-1] = -1
    common = dict(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        seqused_q=torch.ones(1, dtype=torch.int32, device="npu"),
        seqused_ori_kv=torch.tensor([127], dtype=torch.int32, device="npu"),
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        quant_mode=1,
        rope_head_dim=64,
    )
    caches, outputs = [], []
    path = get_npu_kv_quant_path()
    for selected in impl_pair:
        cache = torch.zeros(
            1, 128, 1, path.packed_dim(448, 64), dtype=torch.float8_e4m3fn, device="npu"
        )
        kv_compress_epilog(
            cache,
            x,
            slots,
            quant_group_size=64,
            quant_mode=path.writer_quant_mode,
            round_scale=True,
            x_scale=1.0,
            impl=selected,
        )
        caches.append(cache)
        metadata = mixed_quant_sparse_flash_mla_metadata(
            batch_size=1,
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            has_ori_kv=True,
            has_cmp_kv=False,
            impl=selected,
            **common,
        )
        out, _ = mixed_quant_sparse_flash_mla(
            q,
            ori_kv=cache,
            ori_block_table=torch.zeros(1, 1, device="npu", dtype=torch.int32),
            sinks=torch.zeros(64, device="npu"),
            metadata=metadata,
            softmax_scale=512**-0.5,
            impl=selected,
            **common,
        )
        outputs.append(out)
    # Check semantic row regions; alignment padding is not part of the contract.
    semantic_bytes = 576 + 7 * path.scale_itemsize
    torch.testing.assert_close(
        caches[0].view(torch.uint8)[..., :semantic_bytes],
        caches[1].view(torch.uint8)[..., :semantic_bytes],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(*outputs, atol=0.03, rtol=0.03)
