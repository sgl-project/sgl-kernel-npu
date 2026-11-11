/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
// OpDef for the migrated sparse attention with shared KV. The legacy
// ge_helper definition is preserved in csrc/sparse_attn_sharedkv.

#include "register/op_def_registry.h"
#include "sgl_operator_config.h"

namespace ops {

class SglSparseAttnSharedkv : public OpDef
{
public:
    explicit SglSparseAttnSharedkv(const char *name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const auto *input : {"ori_kv", "cmp_kv"}) {
            this->Input(input)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        for (const auto *input :
             {"ori_sparse_indices", "cmp_sparse_indices", "ori_block_table", "cmp_block_table", "cu_seqlens_q",
              "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_q", "seqused_kv"}) {
            this->Input(input)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("softmax_scale").AttrType(REQUIRED).Float(1.0f);
        this->Attr("cmp_ratio").AttrType(REQUIRED).Int(static_cast<int64_t>(1));
        this->Attr("ori_mask_mode").AttrType(REQUIRED).Int(static_cast<int64_t>(4));
        this->Attr("cmp_mask_mode").AttrType(REQUIRED).Int(static_cast<int64_t>(3));
        this->Attr("ori_kv_stride").AttrType(REQUIRED).Int(static_cast<int64_t>(0));
        this->Attr("cmp_kv_stride").AttrType(REQUIRED).Int(static_cast<int64_t>(0));
        this->Attr("ori_win_left").AttrType(OPTIONAL).Int(static_cast<int64_t>(127));
        this->Attr("ori_win_right").AttrType(OPTIONAL).Int(static_cast<int64_t>(0));
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_ND");
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglSparseAttnSharedkv);

}  // namespace ops
