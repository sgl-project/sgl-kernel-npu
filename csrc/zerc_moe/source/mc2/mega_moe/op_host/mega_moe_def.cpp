/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class ZercMoe : public OpDef
{
public:
    explicit ZercMoe(const char *name) : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("topk_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("topk_weights")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // DT_BF16 weights select the no-quant path (dispatch_quant_mode=0) and
        // carry no weight scales.
        this->Input("weight1")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight2")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight_scales1")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_scales2")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x_active_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scales")
            .ParamType(OPTIONAL)
            // MegaMoe does not currently support the legacy dispatch-scales input.
            // Timer V2 reuses this reserved optional slot to keep the release ACLNN
            // and kernel ABI identical to the ZeRC/compute-only commits.
            .DataTypeList({ge::DT_FLOAT8_E8M0, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataTypeList({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Output("expert_token_nums").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Attr("moe_expert_num").AttrType(REQUIRED).Int();
        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();
        this->Attr("max_recv_token_num").AttrType(OPTIONAL).Int(0);
        this->Attr("dispatch_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("dispatch_quant_out_dtype").AttrType(OPTIONAL).Int(static_cast<int>(ge::DT_UNDEFINED));
        this->Attr("combine_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_alg").AttrType(OPTIONAL).String("");
        this->Attr("global_bs").AttrType(OPTIONAL).Int(0);
        // Shared experts ride as extra trailing entries of weight1/weight2 (and
        // their scale lists): entry 0 is the routed block, entry 1 the shared
        // block. This keeps the prototype at the pre-shared-expert input count
        // (per-DYNAMIC-input launch cost scales with input slots).
        this->Attr("shared_expert_num").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(ZercMoe);

}  // namespace ops
