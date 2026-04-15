# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch_npu
from ...utils.exception import ParametersInvalid


@dataclass
class AttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    max_seqlen_q: int = None
    max_seqlen_k: int = None
    dropout_p: float = 0.0
    window_size: int = None
    softcap: float = None
    alibi_slopes: torch.Tensor = None
    deterministic: bool = None
    return_attn_probs: bool = None
    block_table: torch.Tensor = None


def validate_varlen_attention_inputs(inputs: AttentionInputs):
    unsupported = {
        'max_seqlen_q': inputs.max_seqlen_q,
        'max_seqlen_k': inputs.max_seqlen_k,
        'window_size': inputs.window_size,
        'softcap': inputs.softcap,
        'alibi_slopes': inputs.alibi_slopes,
        'deterministic': inputs.deterministic,
        'return_attn_probs': inputs.return_attn_probs,
        'block_table': inputs.block_table,
    }
    provided = [name for name, val in unsupported.items() if val is not None]
    if provided:
        msg = ", ".join(f"{k}={v}" for k, v in unsupported.items() if v is not None)
        raise ParametersInvalid(f"Unsupported parameters in varlen attention: {msg}")

    if not math.isclose(inputs.dropout_p, 0.0, abs_tol=1e-6):
        raise ParametersInvalid(f"dropout_p should be set to 0.0 during evaluation, but got {inputs.dropout_p}")

    for name, tensor in [('q', inputs.q), ('k', inputs.k), ('v', inputs.v)]:
        if tensor.dim() != 3:
            raise ParametersInvalid(f"Expected {name} to be 3D (T, N, D), but got {tensor.dim()}D tensor.")


def attention_forward_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: list[torch.Tensor],
        cu_seqlens_k: list[torch.Tensor],
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[int] = None,
        softcap: Optional[float] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: Optional[bool] = None,
        return_attn_probs: Optional[bool] = None,
        block_table: Optional[torch.Tensor] = None,
):
    """
    Attention forward function for npu. Input layout must be 'BSND'.
    Args:
        q ('torch.Tensor'):
            The input query of attention calculation formula.
        k ('torch.Tensor'):
            The input key of attention calculation formula.
        v ('torch.Tensor'):
            The input value of attention calculation formula.
        cu_seqlens_q ('list[torch.Tensor]'):
            (batch_size + 1,), dtype torch.int32. The cumulative sequence length for q.
        cu_seqlens_k ('list[torch.Tensor]'):
            (batch_size + 1,), dtype torch.int32. The cumulative sequence length for k.
        max_seqlen_q ('int', *optional*, defaults to `None`):
            Reserved for future use (planned for regularization).
        max_seqlen_k ('int', *optional*, defaults to `None`):
            Reserved for future use (planned for regularization).
        dropout_p ('float'):
            Dropout probability.
        softmax_scale ('float', *optional*, defaults to `None`):
            The input scale of attention calculation formula.
        causal ('bool', defaults to False):
            Reserved for future use (planned for regularization).
        window_size ('int', *optional*, defaults to `None`):
            Reserved for future use (planned for regularization).
        softcap: ('float', *optional*, defaults to `None`). 
            Reserved for future use (planned for regularization).
        alibi_slopes ('torch.Tensor', *optional*, defaults to `None`):
            Reserved for future use (planned for regularization).
        deterministic ('bool', *optional*, defaults to `None`):
            Reserved for future use (planned for regularization).
        return_attn_probs ('bool', *optional*, defaults to `None`): 
            Reserved for future use (planned for regularization).
        block_table ('torch.Tensor', *optional*, defaults to `None`):
            Reserved for future use (planned for regularization).
    Return:
        (total, nheads, headdim).
    """
    inputs = AttentionInputs(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=block_table
    )
    validate_varlen_attention_inputs(inputs)

    _, num_heads, head_dim = q.shape

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    if causal:
        atten_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(q.device)
        sparse_mode = 3
    else:
        atten_mask = None
        sparse_mode = 0

    return torch_npu.npu_fusion_attention(q, k, v, num_heads, pse=None, padding_mask=None,
                                          atten_mask=atten_mask, scale=softmax_scale, keep_prob=1 - dropout_p,
                                          input_layout="TND",
                                          actual_seq_qlen=cu_seqlens_q[1:],
                                          actual_seq_kvlen=cu_seqlens_k[1:],
                                          sparse_mode=sparse_mode,
                                          )[0]
