from typing import Optional

import torch


def laser_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
    alibi_mask: Optional[torch.Tensor] = None,
    drop_mask: Optional[torch.Tensor] = None,
    scale_value: float = 1.0,
    head_num: int = 2,
    input_layout: str = "BNSD",
    keep_prob: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 1,
    is_highPrecision: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Laser Attention on FP16 BNSD inputs.

    The current kernel requires head dimension 128 and sequence lengths that
    are multiples of 128. Both returned tensors use FP32.
    """
    return _attention_op("laser_attn")(
        query,
        key,
        value,
        atten_mask,
        alibi_mask,
        drop_mask,
        scale_value,
        head_num,
        input_layout,
        keep_prob,
        pre_tokens,
        next_tokens,
        is_highPrecision,
    )


def _attention_op(name: str):
    op = getattr(torch.ops.npu, name, None)
    if op is None:
        raise RuntimeError(
            f"{name} was not built for this SoC; these attention kernels require Ascend A2/A3"
        )
    return op


def ada_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_mask: torch.Tensor,
    sparse_count_table: torch.Tensor,
    input_layout: str = "BNSD",
    sparse_size: int = 128,
    num_heads: int = 1,
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    causal: bool = True,
    inner_precise: int = 1,
    pre_tokens: int = 214748647,
    next_tokens: int = 0,
    actual_seq_lengths: Optional[list[int]] = None,
    actual_seq_lengths_kv: Optional[list[int]] = None,
) -> torch.Tensor:
    """Run Ada block sparse attention on A2/A3 FP16 or BF16 inputs.

    Layouts: BNSD, BSND, BSH. Masks use int8 and counts use int32;
    ``sparse_block_estimate`` produces both tensors in the required format.
    """
    return _attention_op("ada_block_sparse_attention")(
        query,
        key,
        value,
        sparse_mask,
        sparse_count_table,
        input_layout,
        sparse_size,
        num_heads,
        num_key_value_heads,
        scale_value,
        causal,
        inner_precise,
        pre_tokens,
        next_tokens,
        actual_seq_lengths,
        actual_seq_lengths_kv,
    )


def sparse_block_estimate(
    query: torch.Tensor,
    key: torch.Tensor,
    actual_seq_lengths: Optional[list[int]] = None,
    actual_seq_lengths_kv: Optional[list[int]] = None,
    input_layout: str = "BNSD",
    stride: int = 8,
    sparse_size: int = 128,
    num_heads: int = 1,
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    threshold: float = 1.0,
    causal: bool = True,
    keep_sink: bool = True,
    keep_recent: bool = True,
    row_sparse: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the int8 block mask and int32 block counts for Ada attention.

    Supports A2/A3 FP16 and BF16 inputs in BNSD, BSND or BSH layout.
    """
    return _attention_op("sparse_block_estimate")(
        query,
        key,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        input_layout,
        stride,
        sparse_size,
        num_heads,
        num_key_value_heads,
        scale_value,
        threshold,
        causal,
        keep_sink,
        keep_recent,
        row_sparse,
    )


__all__ = ["laser_attn", "ada_block_sparse_attention", "sparse_block_estimate"]
