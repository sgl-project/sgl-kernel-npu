// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

#include <cstdint>
#include <initializer_list>

#include "register/op_def_registry.h"
#include "sgl_operator_config.h"

namespace ops {

class SglSparseAttentionScore : public OpDef
{
public:
    explicit SglSparseAttentionScore(const char *name) : OpDef(name)
    {
        for (const auto *input : {"query", "key", "value"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        for (const auto *input : {"select_idx", "block_table"}) {
            this->Input(input)
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("select_num_idx")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const auto *input : {"actual_seq_lengths", "actual_seq_lengths_kv"}) {
            this->Input(input)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        // Retain the legacy scale ports. The Torch adapter describes the pointer
        // type selected by the old dispatch without converting input storage.
        for (const auto *input : {"q_dequant_scale", "k_dequant_scale", "v_dequant_scale"}) {
            this->Input(input)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(static_cast<int64_t>(1));
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("block_size").AttrType(OPTIONAL).Int(static_cast<int64_t>(128));
        this->Attr("top_k").AttrType(OPTIONAL).Int(static_cast<int64_t>(16));
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(static_cast<int64_t>(0));
        for (const auto *config : SGL_CANN_CONFIGS)
            this->AICore().AddConfig(config);
    }
};

OP_ADD(SglSparseAttentionScore);
}  // namespace ops
