"""On-device released-CANN/reference comparisons for DSV4 migrated operators.

Requires the built sgl-kernel-npu wheel and CANN 9.2 transformer Python package.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def test_mhc_pre_post(impl_pair):
    from sgl_kernel_npu.operators.dsv4 import mhc_post, mhc_pre

    impl, reference = impl_pair
    torch.manual_seed(11)
    x = torch.randn(16, 4, 4096, device="npu", dtype=torch.bfloat16)
    fn = torch.randn(24, 4 * 4096, device="npu") * 0.01
    scale = torch.ones(3, device="npu")
    base = torch.zeros(24, device="npu")
    actual = mhc_pre(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl=impl)
    expected = mhc_pre(x, fn, scale, base, 4, 20, 1e-6, 1e-6, impl=reference)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0.03, rtol=0.03)
    y, post, comb = expected
    a = mhc_post(y, x, post, comb, impl=impl)
    e = mhc_post(y, x, post, comb, impl=reference)
    torch.testing.assert_close(a, e, atol=0.06, rtol=0.03)


def test_partial_rope(impl_pair):
    from sgl_kernel_npu.operators.dsv4 import inplace_partial_rotary_mul

    impl, reference = impl_pair
    x = torch.randn(16, 1, 64, 512, device="npu", dtype=torch.bfloat16)
    angles = torch.randn(16, 1, 1, 32, device="npu").repeat_interleave(2, -1)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    a, e = x.clone(), x.clone()
    kwargs = dict(rotary_mode="interleave", partial_slice=[448, 512])
    inplace_partial_rotary_mul(a, cos, sin, impl=impl, **kwargs)
    inplace_partial_rotary_mul(e, cos, sin, impl=reference, **kwargs)
    torch.testing.assert_close(a, e, atol=0.03, rtol=0.03)


def test_bf16_attention_metadata_and_compute(impl_pair):
    from sgl_kernel_npu.operators.dsv4 import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )

    impl, reference = impl_pair
    torch.manual_seed(12)
    q = torch.randn(2, 64, 512, device="npu", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(2, 128, 1, 512, device="npu", dtype=torch.bfloat16)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
    lengths = torch.tensor([64, 128], dtype=torch.int32, device="npu")
    common = dict(
        cu_seqlens_q=cu,
        seqused_q=cu[1:] - cu[:-1],
        seqused_ori_kv=lengths,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    outputs = []
    for selected in impl_pair:
        metadata = sparse_flash_mla_metadata(
            batch_size=2,
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            has_ori_kv=True,
            has_cmp_kv=False,
            impl=selected,
            **common,
        )
        out, _ = sparse_flash_mla(
            q,
            ori_kv=kv,
            ori_block_table=torch.tensor([[0], [1]], device="npu", dtype=torch.int32),
            sinks=torch.zeros(64, device="npu"),
            metadata=metadata,
            softmax_scale=512**-0.5,
            impl=selected,
            **common,
        )
        outputs.append(out)
    torch.testing.assert_close(*outputs, atol=0.03, rtol=0.03)


def test_a5_cache_writer_and_mixed_attention(impl_pair):
    from sgl_kernel_npu.operators.dsv4 import (
        kv_compress_epilog,
        mixed_quant_sparse_flash_mla,
        mixed_quant_sparse_flash_mla_metadata,
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
    for selected in impl_pair:
        cache = torch.zeros(1, 128, 1, 608, dtype=torch.float8_e4m3fn, device="npu")
        kv_compress_epilog(
            cache,
            x,
            slots,
            quant_group_size=64,
            quant_mode="fp8_bf16",
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
    torch.testing.assert_close(
        caches[0].view(torch.uint8)[..., :590],
        caches[1].view(torch.uint8)[..., :590],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(*outputs, atol=0.03, rtol=0.03)


def test_indexer_cache(impl_pair):
    from sgl_kernel_npu.operators.dsv4 import indexer_quant_cache

    x = torch.randn(8, 128, device="npu", dtype=torch.bfloat16)
    slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], device="npu", dtype=torch.int32)
    outputs = []
    for selected in impl_pair:
        keys = torch.zeros(1, 128, 1, 128, device="npu", dtype=torch.float8_e4m3fn)
        scales = torch.zeros(1, 128, 1, 1, device="npu", dtype=torch.float32)
        indexer_quant_cache(keys, scales, x, slots, quant_mode="fp8", impl=selected)
        outputs.append((keys, scales))
    torch.testing.assert_close(
        outputs[0][0].float(), outputs[1][0].float(), atol=16, rtol=0.06
    )
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=1e-5, rtol=0.03)
