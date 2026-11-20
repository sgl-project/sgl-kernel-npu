# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2

"""Direct A5/950 EQBSA dispatch on caller-prepared tensors."""

from __future__ import annotations

import torch

from ._mindiesd import load_eqbsa

_eqbsa = load_eqbsa()


def eagle_quant_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_sparse_mask: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    q_input_layout: str = "BNSD",
    kv_input_layout: str = "BNSD",
    num_key_value_heads: int = 1,
    scale_value: float = 1.0,
    inner_precise: int = 4,
    actual_seq_lengths: list[int] | None = None,
    actual_seq_lengths_kv: list[int] | None = None,
    softmax_lse_flag: int = 0,
    query_scale: torch.Tensor | None = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    query_dtype: torch.dtype | None = None,
    key_dtype: torch.dtype | None = None,
    value_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (attention_out, softmax_lse) from the official raw operator.

    The caller owns quantization, sparse block selection, tensor layout and
    input validation, including shapes, dtypes and quantization scales.
    The A5/950 MIX path has been validated with D=128.
    Do not read softmax_lse unless softmax_lse_flag requests its computation.
    """
    return _eqbsa(
        query,
        key,
        value,
        block_sparse_mask=block_sparse_mask,
        block_shape=[128, 128] if block_shape is None else block_shape,
        q_input_layout=q_input_layout,
        kv_input_layout=kv_input_layout,
        num_key_value_heads=num_key_value_heads,
        scale_value=scale_value,
        inner_precise=inner_precise,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        softmax_lse_flag=softmax_lse_flag,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        query_dtype=query_dtype,
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        output_dtype=output_dtype,
    )
