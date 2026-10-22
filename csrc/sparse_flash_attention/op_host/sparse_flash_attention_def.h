// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_DEF_H
#define SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_DEF_H

#include <cstdint>

#include "ge_helper.h"

namespace sglang::SFAHost {
using namespace ge_helper;

class SparseFlashAttention : public OpDef
{
public:
    explicit SparseFlashAttention(const char *name) : OpDef(name)
    {
        Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("query_rope").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("key_rope").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Output("attention_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Output("softmax_max").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Output("softmax_sum").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Attr("scale_value").AttrType(REQUIRED).Float(1.0F);
        Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        Attr("attention_mode").AttrType(OPTIONAL).Int(2);
        Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
    }
};

}  // namespace sglang::SFAHost
#endif
