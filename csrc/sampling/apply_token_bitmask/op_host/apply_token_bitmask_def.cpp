#include "register/op_def_registry.h"
#include "apply_token_bitmask_tiling.h"
#include "sgl_operator_config.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    if (!input || !output) return ge::GRAPH_FAILED;
    *output = *input;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

class SglApplyTokenBitMask : public OpDef {
public:
    explicit SglApplyTokenBitMask(const char* name) : OpDef(name)
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
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().AddConfig(SGL_CANN_SOC_CONFIG);
    }
};

OP_ADD(SglApplyTokenBitMask);

}  // namespace ops
