import math

import pytest
import torch
import torch_npu  # noqa: F401

import sgl_kernel_npu.attentions as attentions


def test_laser_attn_public_api_and_schema():
    assert callable(attentions.laser_attn)
    assert hasattr(torch.ops.npu, "laser_attn")

    schema = torch.ops.npu.laser_attn.default._schema
    assert schema.name == "npu::laser_attn"
    assert [argument.name for argument in schema.arguments] == [
        "query",
        "key",
        "value",
        "atten_mask",
        "alibi_mask",
        "drop_mask",
        "scale_value",
        "head_num",
        "input_layout",
        "keep_prob",
        "pre_tokens",
        "next_tokens",
        "is_highPrecision",
    ]
    assert len(schema.returns) == 2


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Laser Attention requires an Ascend NPU",
)
@pytest.mark.parametrize("num_heads,num_key_value_heads", [(2, 2), (2, 1)])
def test_laser_attn_fp16_bnsd_matches_torch_reference(
    num_heads, num_key_value_heads
):
    batch_size, seq_len, head_dim = 1, 2048, 128
    query_shape = (batch_size, num_heads, seq_len, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(0)
    query = torch.randn(
        (batch_size, seq_len, num_heads, head_dim),
        device="npu",
        dtype=torch.float16,
    ).transpose(1, 2)
    key = torch.randn(
        (batch_size, seq_len, num_key_value_heads, head_dim),
        device="npu",
        dtype=torch.float16,
    ).transpose(1, 2)
    value = torch.randn(
        (batch_size, seq_len, num_key_value_heads, head_dim),
        device="npu",
        dtype=torch.float16,
    ).transpose(1, 2)
    assert not query.is_contiguous()

    result = attentions.laser_attn(
        query=query,
        key=key,
        value=value,
        atten_mask=None,
        alibi_mask=None,
        drop_mask=None,
        scale_value=scale,
        head_num=num_heads,
        input_layout="BNSD",
        keep_prob=1.0,
        pre_tokens=2**31 - 1,
        next_tokens=1,
        is_highPrecision=True,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    softmax_log_max_sum, attention_out = result
    assert softmax_log_max_sum.shape == (batch_size, num_heads, seq_len)
    assert softmax_log_max_sum.dtype == torch.float32
    assert attention_out.shape == query_shape
    assert attention_out.dtype == torch.float32

    head_group_size = num_heads // num_key_value_heads
    expanded_key = key.repeat_interleave(head_group_size, dim=1)
    expanded_value = value.repeat_interleave(head_group_size, dim=1)
    scores = torch.matmul(query.float(), expanded_key.float().transpose(-2, -1)) * scale
    expected = torch.matmul(torch.softmax(scores, dim=-1), expanded_value.float())
    torch.npu.synchronize()

    torch.testing.assert_close(attention_out, expected, rtol=1e-2, atol=1e-2)
