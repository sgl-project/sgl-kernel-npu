#include "register/op_def_registry.h"

namespace ops {
class NotifyDispatch : public OpDef
{
public:
    explicit NotifyDispatch(const char *name) : OpDef(name)
    {
        this->Input("sendData")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tokenPerExpertData")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sendDataOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("recvData")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("recvCount")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("recvOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("expertGlobalOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("srcrankInExpertOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rInSrcrankOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("totalRecvTokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("maxBs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("recvTokensPerExpert")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("sendCount").Int();
        this->Attr("num_tokens").Int();
        this->Attr("comm_group").String();
        this->Attr("rank_size").Int();
        this->Attr("rank_id").Int();
        this->Attr("local_rank_size").Int();
        this->Attr("local_rank_id").Int();
        this->Attr("round").Int();  // 低7个，ATTR_ROUND_INDEX
        this->Attr("perRoundTokens").Int();

        OpAICoreConfig aicore_config_base;
        aicore_config_base.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        OpAICoreConfig aicore_config_A2 = aicore_config_base;
        aicore_config_A2.ExtendCfgInfo("jitCompile.flag", "static_false");

        OpAICoreConfig aicore_config = aicore_config_base;
        aicore_config.ExtendCfgInfo("jitCompile.flag", "static_true");

        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config_A2);
        this->MC2().HcclGroup("comm_group");
    }
};

OP_ADD(NotifyDispatch);
}  // namespace ops
