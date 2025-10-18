# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
import torchair
from torch import Generator, SymInt, contiguous_format, inf, strided
from torch.types import (
    Device,
    Number,
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
)
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import (
    declare_supported,
    register_fx_node_ge_converter,
)
from torchair._ge_concrete_graph.supported_declaration import (
    BOOL,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    Support,
    _TypedTensor,
)
from torchair.ge import attr
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec


# 为自定义算子注册converter，用于torch.compile 场景成图
# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_sparse_flash_attention.default)
def convert_npu_sparse_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    sparse_block_size: int,
    *,
    block_table: Optional[Tensor] = None,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 3,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "SparseFlashAttention",
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "sparse_indices": sparse_indices,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "query_rope": query_rope,
            "key_rope": key_rope,
        },
        attrs={
            "scale_value": attr.Float(scale_value),
            "sparse_block_size": attr.Int(sparse_block_size),
            "layout_query": attr.Str(layout_query),
            "layout_kv": attr.Str(layout_kv),
            "sparse_mode": attr.Int(sparse_mode),
        },
        outputs=["attention_out"],
    )
