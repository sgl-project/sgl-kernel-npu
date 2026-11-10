// Native CANN contract for the independent mamba operator.
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include <initializer_list>

namespace ops {
class SglCausalConv1d : public OpDef
{
public:
    explicit SglCausalConv1d(const char *name) : OpDef(name)
    {
        // The same two dtype combinations apply to x, weight, state, bias and y.
        for (const auto *input : {"x", "weight", "conv_states"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const auto *input : {"query_start_loc", "cache_indices", "has_initial_state"}) {
            this->Input(input)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("num_accepted_tokens")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // Required same-name Input/Output generates a single ACLNN Ref argument.
        this->Output("conv_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("activation_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("pad_slot_id").AttrType(OPTIONAL).Int(-1);
        this->Attr("run_mode").AttrType(OPTIONAL).Int(0);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglCausalConv1d);
}  // namespace ops
