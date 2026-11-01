// BSD 3-Clause License
// Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

// Split from sgl-kernel-npu csrc/sampler/apply_token_bitmask/op_host/apply_token_bitmask.cpp.
// Interface metadata and computation are preserved.

#include "register/op_def_registry.h"
#include "sgl_operator_config.h"

namespace ops {

class SglApplyTokenBitMask : public OpDef
{
public:
    explicit SglApplyTokenBitMask(const char *name) : OpDef(name)
    {
        // These are the working tensors prepared by the existing caller.
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bitmask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // A same-name output asks the generator for an in-place Ref parameter.
        this->Output("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglApplyTokenBitMask);

}  // namespace ops
