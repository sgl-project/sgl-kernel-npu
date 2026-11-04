# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Tuple

import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error

# isort: split
from ..op_builder.builder import AS_LIBRARY, OpBuilder

# isort: split
from .comm_context import CommContextManager

# dispatch_quant_mode=0: BF16 activations and BF16 weights, no weight scales.
_DISPATCH_QUANT_MODE_NO_QUANT = 0


class _MegaMoeOpBuilder(OpBuilder):
    def __init__(self):
        super(_MegaMoeOpBuilder, self).__init__("npu_zerc_moe")

    def sources(self):
        return ["ops/csrc/mega_moe.cpp"]

    def schema(self) -> str:
        return (
            "npu_zerc_moe(Tensor context, Tensor x, Tensor topk_ids, Tensor topk_weights, "
            "Tensor[] weight1, Tensor[] weight2, int moe_expert_num, int ep_world_size, int ccl_buffer_size, *, "
            "Tensor[]? weight_scales1=None, Tensor[]? weight_scales2=None, Tensor? x_active_mask=None, "
            "Tensor? scales=None, int? max_recv_token_num=0, int? dispatch_quant_mode=0, "
            'int? combine_quant_mode=0, str? comm_alg="", int? global_bs=0, '
            "int? dispatch_quant_out_dtype=28, int? weight1_type=28, int? weight2_type=28, "
            "Tensor? timer=None) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mega_moe_meta(
            context,
            x,
            topk_ids,
            topk_weights,
            weight1,
            weight2,
            moe_expert_num,
            ep_world_size,
            ccl_buffer_size,
            weight_scales1=None,
            weight_scales2=None,
            x_active_mask=None,
            scales=None,
            max_recv_token_num=0,
            dispatch_quant_mode=0,
            combine_quant_mode=0,
            comm_alg="",
            global_bs=0,
            dispatch_quant_out_dtype=28,
            weight1_type=28,
            weight2_type=28,
            timer=None,
        ):
            torch._check(
                ep_world_size != 0,
                lambda: (
                    f"ep_rank_id should not be 0, " f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            bs = x.size(0)
            h = x.size(1)
            local_moe_expert_num = moe_expert_num // ep_world_size
            y = x.new_empty(tuple([bs, h]), dtype=x.dtype)
            expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int32)
            return (y, expert_token_nums)


_mega_moe_op_builder = _MegaMoeOpBuilder()
_op_module = _mega_moe_op_builder.load()


@impl(AS_LIBRARY, _mega_moe_op_builder.name, "PrivateUse1")
def _npu_mega_moe(
    context,
    x,
    topk_ids,
    topk_weights,
    weight1,
    weight2,
    moe_expert_num,
    ep_world_size,
    ccl_buffer_size,
    weight_scales1=None,
    weight_scales2=None,
    x_active_mask=None,
    scales=None,
    max_recv_token_num=0,
    dispatch_quant_mode=0,
    combine_quant_mode=0,
    comm_alg="",
    global_bs=0,
    dispatch_quant_out_dtype=28,
    weight1_type=28,
    weight2_type=28,
    timer=None,
    shared_weight1=None,
    shared_weight2=None,
    shared_weight_scales1=None,
    shared_weight_scales2=None,
):
    return _op_module.npu_zerc_moe(
        context,
        x,
        topk_ids,
        topk_weights,
        weight1,
        weight2,
        moe_expert_num,
        ep_world_size,
        ccl_buffer_size,
        weight_scales1,
        weight_scales2,
        x_active_mask,
        scales,
        max_recv_token_num,
        dispatch_quant_mode,
        combine_quant_mode,
        comm_alg,
        global_bs,
        dispatch_quant_out_dtype,
        weight1_type,
        weight2_type,
        timer,
        shared_weight1,
        shared_weight2,
        shared_weight_scales1,
        shared_weight_scales2,
    )


class SymmBuffer:
    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        max_recv_token_num: int = 0,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: int = 28,
        combine_quant_mode: int = 0,
        comm_alg: str = "",
    ):
        # Metadata
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank_id, init_comm=False
        )
        self.ep_world_size = torch.distributed.get_world_size(group)
        required_ccl_buffer_size = _get_mega_moe_custom_ccl_buffer_size_bytes(
            self.ep_world_size,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
            combine_quant_mode,
            comm_alg,
        )
        # CANN 9.1 channel 协议要求由算子注册自己持有的设备内存，不能再把
        # ProcessGroup 的默认 HCCL buffer 直接填入 channel descriptor。
        self._ctx_manager = CommContextManager(
            self.group_name,
            self.ep_world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            customCclBufferSize=required_ccl_buffer_size,
        )
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size
        # Channel 模式会精确回填 custom buffer 大小；KFC 模式仍使用
        # ProcessGroup 的默认 HCCL buffer，因此这里只做容量下限校验。
        if self.ccl_buffer_size < required_ccl_buffer_size:
            raise RuntimeError(
                "MegaMoe MC2 CCL buffer is smaller than required: "
                f"actual={self.ccl_buffer_size}, required={required_ccl_buffer_size}."
            )
        self.num_experts = num_experts
        self.max_recv_token_num = max_recv_token_num
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.dispatch_quant_mode = dispatch_quant_mode
        self.dispatch_quant_out_dtype = dispatch_quant_out_dtype
        self.combine_quant_mode = combine_quant_mode
        self.comm_alg = comm_alg
        # 当前定制 kernel 的属性名仍是 global_bs，值应为全通信域上限。
        self.global_bs = num_max_tokens_per_rank * self.ep_world_size

    def destroy(self):
        self._ctx_manager.destroy()


def npu_get_mega_moe_ccl_buffer_size(
    ep_world_size: int,
    moe_expert_num: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: int = 28,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
) -> int:
    def inline_align(val, align):
        return (val + align - 1) // align * align

    # 保留公开接口原有的 MB 语义，兼容仍使用该值配置 HCCL_BUFFSIZE 的调用方。
    local_moe_expert_num = moe_expert_num // ep_world_size
    peermem_data_offset = 60 * 1024
    y_out_dtype_size = 2
    mb_conversion = 1024 * 1024
    dispatch_token_per_expert = (
        ep_world_size * inline_align(ep_world_size * local_moe_expert_num, 128) * 4
    )
    if dispatch_quant_mode == _DISPATCH_QUANT_MODE_NO_QUANT:
        # BF16 no-quant dispatches the unquantized BF16 rows without scales.
        dispatch_row_bytes = hidden * 2
    else:
        dispatch_row_bytes = hidden + hidden // 32
    quant_token_scales = inline_align(
        num_max_tokens_per_rank * num_topk * dispatch_row_bytes, 512
    )
    combine_out = inline_align(
        num_max_tokens_per_rank * hidden * num_topk * y_out_dtype_size, 512
    )
    ccl_buffer_size = (
        peermem_data_offset
        + dispatch_token_per_expert
        + quant_token_scales
        + combine_out
    )
    ccl_buffer_size = (
        inline_align(inline_align(ccl_buffer_size, mb_conversion) // mb_conversion, 2)
        // 2
    )

    return ccl_buffer_size


def _get_mega_moe_custom_ccl_buffer_size_bytes(
    ep_world_size: int,
    moe_expert_num: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: int = 28,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
) -> int:
    del dispatch_quant_out_dtype, combine_quant_mode, comm_alg

    def inline_align(val, align):
        return (val + align - 1) // align * align

    if ep_world_size < 2:
        raise ValueError(f"ep_world_size must be >= 2, got {ep_world_size}")
    if moe_expert_num <= 0 or moe_expert_num % ep_world_size != 0:
        raise ValueError(
            f"moe_expert_num({moe_expert_num}) must be positive and divisible by "
            f"ep_world_size({ep_world_size})"
        )
    if num_max_tokens_per_rank <= 0:
        raise ValueError(
            "num_max_tokens_per_rank must be the maximum local BS and greater than 0 "
            "when creating an Ascend950 MC2 registered buffer"
        )
    if num_topk <= 0 or hidden <= 0:
        raise ValueError(
            f"num_topk and hidden must be positive, got {num_topk}, {hidden}"
        )

    local_moe_expert_num = moe_expert_num // ep_world_size
    # 与 mc2/mega_moe/op_host/op_tiling/mega_moe_tiling.cpp 的
    # leastCclBufferSize 以及 op_kernel/mega_moe_base.h::PeermemInfo 严格对齐。
    peermem_data_offset = 60 * 1024
    dispatch_token_per_expert = (
        ep_world_size * inline_align(ep_world_size * local_moe_expert_num, 128) * 4
    )
    expert_cumsum = inline_align(ep_world_size * local_moe_expert_num, 128) * 4
    expanded_rows = (
        inline_align(ep_world_size * num_max_tokens_per_rank * num_topk, 128) * 4
    )
    # host tiling 当前按 ALIGN_128 * sizeof(float) 校验，比 kernel 的
    # ALIGN_256 * sizeof(bfloat16_t) 更保守，这里必须满足 host 上限。
    expert_probs = (
        inline_align(ep_world_size * num_max_tokens_per_rank * num_topk, 128) * 4
    )
    if dispatch_quant_mode == _DISPATCH_QUANT_MODE_NO_QUANT:
        # kernel GetA0RowStrideBf16: H 个 BF16 元素补齐到 32B, 无 scale 段。
        dispatch_a0 = num_max_tokens_per_rank * inline_align(hidden * 2, 32)
    else:
        host_a0_row_bytes = hidden + hidden // 32
        # kernel GetA0RowStride: Align256(k) + Align32(CeilDiv(k,64)*2) —— 数据段补齐到256B
        # (quant ComputeFp8Data按256B粒度写回, 尾段补0), scale段补齐到32B。
        kernel_a0_row_bytes = inline_align(hidden, 256) + inline_align(
            (hidden + 63) // 64 * 2, 32
        )
        dispatch_a0 = num_max_tokens_per_rank * max(
            host_a0_row_bytes, kernel_a0_row_bytes
        )
    combine_d = inline_align(num_max_tokens_per_rank * ep_world_size * hidden * 2, 128)
    required_bytes = (
        peermem_data_offset
        + dispatch_token_per_expert
        + expert_cumsum
        + expanded_rows
        + expert_probs
        + dispatch_a0
        + combine_d
    )
    # 页级别向上取整，避免后续 tiling 边界或内存注册粒度造成越界。
    return inline_align(required_bytes, 2 * 1024 * 1024)


def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    max_recv_token_num: int = 0,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: int = 28,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
) -> SymmBuffer:

    return SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        max_recv_token_num,
        dispatch_quant_mode,
        dispatch_quant_out_dtype,
        combine_quant_mode,
        comm_alg,
    )


def mega_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: List[torch.Tensor],
    l2_weights: List[torch.Tensor],
    sym_buffer: SymmBuffer,
    scales: Optional[torch.Tensor] = None,
    l1_weights_sf: Optional[List[torch.Tensor]] = None,
    l2_weights_sf: Optional[List[torch.Tensor]] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    weight1_type: int = 28,
    weight2_type: int = 28,
    timer: Optional[torch.Tensor] = None,
    # 共享专家：不参与路由的本地专家，对本卡全部 token 本地计算，无需 Dispatch。
    # 传 None 表示未启用，此时 host tiling 保持既有布局（逐字节一致）。
    # l1/l2_weights_shared 的 dtype 需与 l1_weights/l2_weights 一致。
    l1_weights_shared: Optional[List[torch.Tensor]] = None,
    l2_weights_shared: Optional[List[torch.Tensor]] = None,
    l1_weights_shared_sf: Optional[List[torch.Tensor]] = None,
    l2_weights_shared_sf: Optional[List[torch.Tensor]] = None,
):

    return _npu_mega_moe(
        sym_buffer.context,
        x,
        topk_ids,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer.num_experts,
        sym_buffer.ep_world_size,
        sym_buffer.ccl_buffer_size,
        l1_weights_sf,
        l2_weights_sf,
        x_active_mask,
        scales,
        sym_buffer.max_recv_token_num,
        sym_buffer.dispatch_quant_mode,
        sym_buffer.combine_quant_mode,
        sym_buffer.comm_alg,
        sym_buffer.global_bs,
        sym_buffer.dispatch_quant_out_dtype,
        weight1_type,
        weight2_type,
        timer,
        l1_weights_shared,
        l2_weights_shared,
        l1_weights_shared_sf,
        l2_weights_shared_sf,
    )
