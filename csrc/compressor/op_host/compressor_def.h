/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file compressor_def.h
 * \brief ge_helper-based operator definition for the Compressor op.
 */
#ifndef COMPRESSOR_DEF_H
#define COMPRESSOR_DEF_H

#include "ge_helper.h"

namespace sglang {
namespace CompressorHost {
using namespace ge_helper;

class Compressor : public OpDef
{
public:
    explicit Compressor(const char *name) : OpDef(name)
    {
        // ---- 12 个输入（前 8 必选，后 4 可选）----
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wgate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("state_cache").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Input("ape")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("norm_weight")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rope_sin")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rope_cos")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("state_block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("start_pos")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        // ---- 输出：cmp_kv 为新输出；state_cache 为 in-place 原地写回（host 里单独处理）----
        this->Output("cmp_kv").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});

        // ---- 7 个 attr ----
        this->Attr("rope_head_dim").AttrType(REQUIRED).Int(64);
        this->Attr("cmp_ratio").AttrType(REQUIRED).Int(4);
        this->Attr("coff").AttrType(OPTIONAL).Int(1);
        this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("rotary_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("cache_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("state_cache_stride_dim0").AttrType(OPTIONAL).Int(0);
    }
};

}  // namespace CompressorHost
}  // namespace sglang

#endif  // COMPRESSOR_DEF_H
