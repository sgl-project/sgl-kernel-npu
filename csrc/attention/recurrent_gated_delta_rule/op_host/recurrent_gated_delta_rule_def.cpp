// Native CANN interface. Python's optional working pool is normalized by the adapter.
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"

namespace ops {
class SglRecurrentGatedDeltaRule : public OpDef
{
public:
    explicit SglRecurrentGatedDeltaRule(const char *name) : OpDef(name)
    {
        for (const auto *input : {"mix_qkv", "state", "beta"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const auto *input : {"actual_seq_lengths", "ssm_state_indices"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Input("recurrent_seed")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        for (const auto *input : {"cache_indices", "num_accepted_tokens"}) {
            this->Input(input)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const auto *input : {"g", "gk"}) {
            this->Input(input)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // Same-name required ports generate one ACLNN Ref parameter for state.
        this->Output("state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scale").AttrType(REQUIRED).Float(1.0f);
        this->Attr("nk").AttrType(REQUIRED).Int();
        this->Attr("nv").AttrType(REQUIRED).Int();
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};
OP_ADD(SglRecurrentGatedDeltaRule);
}  // namespace ops
