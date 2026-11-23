# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge

    # isort: split
    from torchair.ge._ge_graph import Tensor, TensorSpec

    # isort: split
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support

    # isort: split
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

    # isort: split
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDef,
        TensorDescriptor,
    )

    # isort: split
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        TensorType,
        auto_convert_to_tensor,
        compat_as_bytes,
        compat_as_bytes_list,
        get_default_ge_graph,
        get_invalid_desc,
        next_unique_name,
        trans_to_list_list_float,
        trans_to_list_list_int,
    )

    # isort: split
    from torchair._ge_concrete_graph.compat_ir import IrDef, ge_op
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, True, True, True, True, False, False],
        [False, False, False, False, False, False, False, False, True, True],
    )
    def ZercMoe(
        context: Tensor,
        x: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        weight1: List[Tensor],
        weight2: List[Tensor],
        weight_scales1: List[Tensor],
        weight_scales2: List[Tensor],
        x_active_mask: Optional[Tensor],
        scales: Optional[Tensor],
        *,
        moe_expert_num: int,
        ep_world_size: int,
        ccl_buffer_size: int,
        max_recv_token_num: int = 0,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: int = 28,
        combine_quant_mode: int = 0,
        comm_alg: str = "",
        global_bs: int = 0,
        shared_expert_num: int = 0,
        activation_clamp: float = -1.0,
        dependencies=[],
        node_name=None
    ):
        """REG_OP(ZercMoe)\n
        .INPUT(context, TensorType({DT_INT32}))\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,\n
            DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .INPUT(topk_ids, TensorType({DT_INT32}))\n
        .INPUT(topk_weights, TensorType({DT_FLOAT, DT_BF16}))\n
        .DYNAMIC_INPUT(weight1, TensorType({DT_BF16, DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,\n
            DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .DYNAMIC_INPUT(weight2, TensorType({DT_BF16, DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,\n
            DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))\n
        .DYNAMIC_INPUT(weight_scales1, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .DYNAMIC_INPUT(weight_scales2, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OPTIONAL_INPUT(x_active_mask, TensorType({DT_INT8}))\n
        .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0, DT_INT64}))\n
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(expert_token_nums, TensorType({DT_INT32}))\n
        .REQUIRED_ATTR(moe_expert_num, Int)\n
        .REQUIRED_ATTR(ep_world_size, Int)\n
        .REQUIRED_ATTR(ccl_buffer_size, Int)\n
        .ATTR(max_recv_token_num, Int, 0)\n
        .ATTR(dispatch_quant_mode, Int, 0)\n
        .ATTR(dispatch_quant_out_dtype, Int, 28)\n
        .ATTR(combine_quant_mode, Int, 0)\n
        .ATTR(comm_alg, String, "")\n
        .ATTR(global_bs, Int, 0)\n        .ATTR(shared_expert_num, Int, 0)\n
        .ATTR(activation_clamp, Float, -1.0)\n
        """
        # process inputs
        inputs = {
            "context": context,
            "x": x,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "weight1": weight1,
            "weight2": weight2,
            "weight_scales1": weight_scales1,
            "weight_scales2": weight_scales2,
            "x_active_mask": x_active_mask,
            "scales": scales,
        }

        # process attrs
        attrs = {
            "moe_expert_num": attr.Int(moe_expert_num),
            "ep_world_size": attr.Int(ep_world_size),
            "ccl_buffer_size": attr.Int(ccl_buffer_size),
            "max_recv_token_num": attr.Int(max_recv_token_num),
            "dispatch_quant_mode": attr.Int(dispatch_quant_mode),
            "dispatch_quant_out_dtype": attr.Int(dispatch_quant_out_dtype),
            "combine_quant_mode": attr.Int(combine_quant_mode),
            "comm_alg": attr.Str(comm_alg),
            "global_bs": attr.Int(global_bs),
            "shared_expert_num": attr.Int(shared_expert_num),
            "activation_clamp": attr.Float(activation_clamp),
        }

        # process outputs
        outputs = [
            "y",
            "expert_token_nums",
        ]

        return ge_op(
            op_type="ZercMoe",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("ZercMoe")
            .input("context", "DT_INT32")
            .input(
                "x",
                "DT_BF16, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, "
                "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
            )
            .input("topk_ids", "DT_INT32")
            .input("topk_weights", "DT_FLOAT, DT_BF16")
            .dynamic_input(
                "weight1",
                "DT_BF16, DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, "
                "DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
            )
            .dynamic_input(
                "weight2",
                "DT_BF16, DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, "
                "DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2",
            )
            .dynamic_input("weight_scales1", "DT_FLOAT, DT_FLOAT8_E8M0")
            .dynamic_input("weight_scales2", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("x_active_mask", "DT_INT8")
            .optional_input("scales", "DT_FLOAT, DT_FLOAT8_E8M0, DT_INT64")
            .required_attr("moe_expert_num", attr.Int)
            .required_attr("ep_world_size", attr.Int)
            .required_attr("ccl_buffer_size", attr.Int)
            .attr("max_recv_token_num", attr.Int(0))
            .attr("dispatch_quant_mode", attr.Int(0))
            .attr("dispatch_quant_out_dtype", attr.Int(28))
            .attr("combine_quant_mode", attr.Int(0))
            .attr("comm_alg", attr.Str(""))
            .attr("global_bs", attr.Int(0))
            .attr("shared_expert_num", attr.Int(0))
            .attr("activation_clamp", attr.Float(-1.0))
            .output("y", "DT_BF16, DT_FLOAT16")
            .output("expert_token_nums", "DT_INT32"),
        )

    @register_fx_node_ge_converter(torch.ops.zerc_moe.npu_zerc_moe.default)
    def convert_npu_mega_moe(
        context: Tensor,
        x: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        weight1: List[Tensor],
        weight2: List[Tensor],
        moe_expert_num: int,
        ep_world_size: int,
        ccl_buffer_size: int,
        *,
        weight_scales1: Optional[List[Tensor]] = None,
        weight_scales2: Optional[List[Tensor]] = None,
        x_active_mask: Optional[Tensor] = None,
        scales: Optional[Tensor] = None,
        max_recv_token_num: int = 0,
        dispatch_quant_mode: int = 0,
        combine_quant_mode: int = 0,
        comm_alg: str = "",
        global_bs: int = 0,
        shared_expert_num: int = 0,
        dispatch_quant_out_dtype: int = 28,
        weight1_type: int = 28,
        weight2_type: int = 28,
        timer: Optional[Tensor] = None,
        activation_clamp: float = -1.0,
        meta_outputs: TensorSpec = None
    ):

        if scales is not None:
            raise ValueError(
                "MegaMoe scales is not supported; this reserved optional slot is used internally by Timer V2"
            )

        if activation_clamp == float("inf"):
            activation_clamp = -1.0

        return ZercMoe(
            context=context,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            weight1=weight1,
            weight2=weight2,
            weight_scales1=weight_scales1,
            weight_scales2=weight_scales2,
            x_active_mask=x_active_mask,
            scales=timer,
            moe_expert_num=moe_expert_num,
            ep_world_size=ep_world_size,
            ccl_buffer_size=ccl_buffer_size,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=dispatch_quant_mode,
            combine_quant_mode=combine_quant_mode,
            comm_alg=comm_alg,
            global_bs=global_bs,
            shared_expert_num=shared_expert_num,
            activation_clamp=activation_clamp,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )
