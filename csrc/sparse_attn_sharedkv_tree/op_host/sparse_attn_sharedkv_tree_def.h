/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SPARSE_ATTN_SHAREDKV_TREE_DEF_H
#define SPARSE_ATTN_SHAREDKV_TREE_DEF_H

#include "ge_helper.h"

namespace sglang {
namespace SparseAttnSharedkvTreeHost {
using namespace ge_helper;

class SparseAttnSharedkvTree : public OpDef {
public:
    explicit SparseAttnSharedkvTree(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("ori_kv").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("cmp_kv").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("ori_sparse_indices").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("cmp_sparse_indices").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("ori_block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("cmp_block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("cu_seqlens_q").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("cu_seqlens_ori_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("cu_seqlens_cmp_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("seqused_q").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("seqused_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("sinks").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Input("metadata").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("cmp_tree_kv").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("tree_ancestors").ParamType(OPTIONAL).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("tree_prefix").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Output("attn_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});

        this->Attr("softmax_scale").AttrType(REQUIRED).Float(1.0f);
        this->Attr("cmp_ratio").AttrType(REQUIRED).Int(1);
        this->Attr("ori_mask_mode").AttrType(REQUIRED).Int(4);
        this->Attr("cmp_mask_mode").AttrType(REQUIRED).Int(3);
        this->Attr("ori_kv_stride").AttrType(REQUIRED).Int(0);
        this->Attr("cmp_kv_stride").AttrType(REQUIRED).Int(0);
        this->Attr("ori_win_left").AttrType(OPTIONAL).Int(127);
        this->Attr("ori_win_right").AttrType(OPTIONAL).Int(0);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_ND");
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
    }
};
}  // namespace SparseAttnSharedkvTreeHost
}  // namespace sglang

#endif  // SPARSE_ATTN_SHAREDKV_TREE_DEF_H
