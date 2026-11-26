// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"

namespace ops {

class SglQsaPrefillRuns : public OpDef
{
public:
    explicit SglQsaPrefillRuns(const char *name) : OpDef(name)
    {
        for (const auto *input : {"q", "k", "v"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const auto *input : {"slots", "runs", "counts"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale").AttrType(OPTIONAL).Float(0.0625f);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglQsaPrefillRuns);

}  // namespace ops
