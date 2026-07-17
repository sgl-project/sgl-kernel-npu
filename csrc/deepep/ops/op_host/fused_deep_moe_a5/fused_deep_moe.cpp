/**
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe operator definition file
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe operator definition file
 */
#if defined(__DAV_C310__)
#include "register/op_def_registry.h"

namespace ops {
namespace {
// 16 variants: quant weight (4) x activation x (2) x weight layout (2).
//   quant: FP8 E4M3, FP8 E5M2, FP4 E2M1, FP4 E1M2
//   x:     BF16, FP16
//   layout: ND, FRACTAL_NZ
#define FUSED_DEEP_MOE_ACT_DTYPES                                                                                      \
    ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,             \
        ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,         \
        ge::DT_FLOAT16, ge::DT_FLOAT16

#define FUSED_DEEP_MOE_GMM_WEIGHT_DTYPES                                                                               \
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,     \
        ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1,         \
        ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2,         \
        ge::DT_FLOAT4_E1M2

#define FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS                                                                              \
    ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, \
        ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,                   \
        ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ
}  // namespace

class FusedDeepMoe : public OpDef {
public:
    explicit FusedDeepMoe(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({FUSED_DEEP_MOE_ACT_DTYPES})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gmm1_weight")
            .ParamType(DYNAMIC)
            .DataType({FUSED_DEEP_MOE_GMM_WEIGHT_DTYPES})
            .Format({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .UnknownShapeFormat({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .AutoContiguous();
        this->Input("gmm1_weight_scale")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gmm2_weight")
            .ParamType(DYNAMIC)
            .DataType({FUSED_DEEP_MOE_GMM_WEIGHT_DTYPES})
            .Format({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .UnknownShapeFormat({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .AutoContiguous();
        this->Input("gmm2_weight_scale")
            .ParamType(DYNAMIC)
            .DataTypeList({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_scales")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("share_gmm1_weight")
            .ParamType(OPTIONAL)
            .DataType({FUSED_DEEP_MOE_GMM_WEIGHT_DTYPES})
            .Format({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .UnknownShapeFormat({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .AutoContiguous();
        this->Input("share_gmm1_weight_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("share_gmm2_weight")
            .ParamType(OPTIONAL)
            .DataType({FUSED_DEEP_MOE_GMM_WEIGHT_DTYPES})
            .Format({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .UnknownShapeFormat({FUSED_DEEP_MOE_GMM_WEIGHT_FORMATS})
            .AutoContiguous();
        this->Input("share_gmm2_weight_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_smooth_scales")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("share_smooth_scales")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x_active_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({FUSED_DEEP_MOE_ACT_DTYPES})
            .FormatList({ge::FORMAT_ND});
        this->Output("share_output")
            .ParamType(REQUIRED)
            .DataType({FUSED_DEEP_MOE_ACT_DTYPES})
            .FormatList({ge::FORMAT_ND});
        this->Output("expert_token_nums")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});

        this->Attr("group_ep").String();
        this->Attr("ep_rank_size").Int();
        this->Attr("ep_rank_id").Int();
        this->Attr("moe_expert_num").Int();
        this->Attr("quant_mode").Int();
        this->Attr("global_bs").Int();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        this->MC2().HcclGroup({"group_ep"});
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(FusedDeepMoe);
}  // namespace ops
#endif  // defined(__DAV_C310__)

