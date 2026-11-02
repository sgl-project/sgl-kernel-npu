"""On-device production/reference checks for attention operators."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def test_bf16_attention_metadata_and_compute(impl_pair):
    from sgl_kernel_npu.operators.attention import (
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
