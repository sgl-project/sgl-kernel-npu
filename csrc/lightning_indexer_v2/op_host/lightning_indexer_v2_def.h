/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_v2_def.h
 * \brief
 */
#ifndef SGL_LI_V2_DEF_H
#define SGL_LI_V2_DEF_H
#include <cstdint>
#include "ge_helper.h"

namespace sglang {
namespace LIV2Host {
using namespace ge_helper;

// 与上游 op_host/lightning_indexer_def.cpp 对应。
// 上游 weights 的 dtype 列表为 {BF16, FP16, FLOAT, FLOAT}，靠框架按四元组选档；
// ge_helper::OpDef 只能按 query 的 dtype 选出一个下标，因此这里 weights 仍按
// {BF16, FP16} 声明，fp32 weights 由 launcher 调用
// TilingContext::OverrideInputDataType() 修正（见 lightning_indexer_v2.cpp）。
class LightningIndexerV2 : public OpDef
{
public:
    explicit LightningIndexerV2(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_key").AttrType(OPTIONAL).String("BSND");
        this->Attr("sparse_count").AttrType(OPTIONAL).Int(2048);  // 2048:默认值，筛选前2048
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);      // 3:默认值，只计算下三角
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int64(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int64(INT64_MAX);
        this->Attr("return_values").AttrType(OPTIONAL).Bool(false);
    }
};
}  // namespace LIV2Host
}  // namespace sglang
#endif  // SGL_LI_V2_DEF_H
