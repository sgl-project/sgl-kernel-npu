/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_DEF_H
#define SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_DEF_H

#include "ge_helper.h"

namespace sglang::KvQuantSASHost {
using namespace ge_helper;

class KvQuantSparseAttnSharedkv : public OpDef
{
public:
    explicit KvQuantSparseAttnSharedkv(const char *name) : OpDef(name)
    {
        Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("ori_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_ND});
        Input("cmp_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_ND});
        Input("ori_sparse_indices").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("cmp_sparse_indices").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("ori_block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("cmp_block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("cu_seqlens_q").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("cu_seqlens_ori_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("cu_seqlens_cmp_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("seqused_q").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("seqused_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("sinks").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Input("metadata").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Output("attn_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Output("softmax_lse").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});

        // Keep this order in sync with the source parser's ATTR_*_INDEX values.
        Attr("kv_quant_mode").AttrType(REQUIRED).Int(int64_t{1});
        Attr("tile_size").AttrType(OPTIONAL).Int(int64_t{64});
        Attr("rope_head_dim").AttrType(OPTIONAL).Int(int64_t{64});
        Attr("softmax_scale").AttrType(REQUIRED).Float(1.0F);
        Attr("cmp_ratio").AttrType(REQUIRED).Int(int64_t{1});
        Attr("ori_mask_mode").AttrType(REQUIRED).Int(int64_t{4});
        Attr("cmp_mask_mode").AttrType(REQUIRED).Int(int64_t{3});
        Attr("ori_win_left").AttrType(OPTIONAL).Int(int64_t{127});
        Attr("ori_win_right").AttrType(OPTIONAL).Int(int64_t{0});
        Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        Attr("layout_kv").AttrType(OPTIONAL).String("PA_ND");
        Attr("ori_kv_stride0").AttrType(OPTIONAL).Int(int64_t{0});
        Attr("cmp_kv_stride0").AttrType(OPTIONAL).Int(int64_t{0});
        Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
    }
};

}  // namespace sglang::KvQuantSASHost
#endif
