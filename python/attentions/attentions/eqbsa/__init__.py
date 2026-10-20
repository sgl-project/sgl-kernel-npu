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

    The caller owns quantization, sparse block selection and tensor layout.
    A5/950 validation is restricted to D=128; the MIX kernel has a known D=64
    accuracy issue. INT8 inputs require FP32 scales. BNSD scale shapes and
    binary mask block counts are checked against the kernel's memory layout.
    TND scale packing and INT32 sparse tables remain the caller's responsibility.
    Other argument validation belongs to the official operator.
    Do not read softmax_lse unless softmax_lse_flag requests its computation.
    """
    if query.shape[-1] != 128:
        raise ValueError("A5/950 EQBSA currently supports only head dimension D=128.")
    # The INT8 kernel reads all three scales; the raw schema marks them optional.
    if query.dtype == torch.int8:
        for name, tensor, scale, layout in (
            ("query", query, query_scale, q_input_layout),
            ("key", key, key_scale, kv_input_layout),
            ("value", value, value_scale, kv_input_layout),
        ):
            if scale is None:
                raise ValueError(
                    "INT8 EQBSA requires query_scale, key_scale and value_scale."
                )
            if scale.dtype != torch.float32:
                raise ValueError(f"{name}_scale must have dtype torch.float32.")
            if layout == "BNSD" and tensor.ndim == 4:
                # Q/K scales use 64-token blocks, independently of block_shape.
                size = (
                    tensor.shape[-1]
                    if name == "value"
                    else (tensor.shape[-2] + 63) // 64
                )
                expected = (*tensor.shape[:2], size)
                if scale.shape != expected:
                    raise ValueError(
                        f"{name}_scale must have shape {expected} for BNSD."
                    )
    block_shape = [128, 128] if block_shape is None else block_shape
    if (
        q_input_layout == kv_input_layout == "BNSD"
        and query.ndim == key.ndim == 4
        and block_sparse_mask is not None
        and block_sparse_mask.ndim == 4
        and block_sparse_mask.dtype in (torch.int8, torch.bool)
        and len(block_shape) == 2
        and block_shape[0] > 0
        and block_shape[1] > 0
    ):
        # Leave rank, dtype and block_shape errors to the official operator.
        expected_blocks = (
            (query.shape[2] + block_shape[0] - 1) // block_shape[0],
            (key.shape[2] + block_shape[1] - 1) // block_shape[1],
        )
        if block_sparse_mask.shape[2:] != expected_blocks:
            raise ValueError(
                f"block_sparse_mask must have block counts {expected_blocks}."
            )
    return _eqbsa(
        query,
        key,
        value,
        block_sparse_mask=block_sparse_mask,
        block_shape=block_shape,
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
