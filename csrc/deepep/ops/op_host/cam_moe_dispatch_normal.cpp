#include "register/op_def_registry.h"

namespace ops {
class CamMoeDispatchNormal : public OpDef
{
public:
    explicit CamMoeDispatchNormal(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("topk_idx")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("send_offset")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("send_tokenIdx")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("recv_offset")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("recv_count")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("expert_global_offset")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("srcrank_in_expert_offset")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("r_in_srcrank_offset")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("recv_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E1M2})
            .FormatList({ge::FORMAT_ND});

        this->Output("x_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND});
        this->Output("assist_info_for_combine")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Output("dispatch_wait_recv_cost_stats")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Attr("group_ep").AttrType(REQUIRED).String();
        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("ep_rank_id").AttrType(REQUIRED).Int();
        this->Attr("group_tp").AttrType(OPTIONAL).String("");
        this->Attr("tp_world_size").AttrType(OPTIONAL).Int(0);
        this->Attr("tp_rank_id").AttrType(OPTIONAL).Int(0);
        this->Attr("moe_expert_num").AttrType(REQUIRED).Int();
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("real_max_bs").AttrType(OPTIONAL).Int(0);
        this->Attr("global_bs").AttrType(OPTIONAL).Int(0);
        this->Attr("round").AttrType(OPTIONAL).Int(4);
        this->Attr("per_round_tokens").AttrType(OPTIONAL).Int(1024);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
#ifdef __DAV_C310__
        this->AICore().AddConfig("ascend950", aicore_config);
#endif
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->MC2().HcclGroup({"group_ep", "group_tp"});
    }
};

OP_ADD(CamMoeDispatchNormal);

}  // namespace ops
