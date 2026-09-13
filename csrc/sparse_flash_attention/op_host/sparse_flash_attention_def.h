/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_def.h
 * \brief Operator interface, transcribed from upstream's op_host/sparse_flash_attention_def.cpp.
 *
 * Upstream declares the op as an ops::OpDef and hands it to the CANN op packer with OP_ADD(). Here it
 * is a ge_helper::OpDef instead: the same input/output/attr table, but consumed by
 * OpDef::SetToContext() to build the fake tiling context the vendored tiling parses. Nothing is
 * registered with GE, so this creates no custom vendor-package op and cannot shadow CANN's own
 * SparseFlashAttention. The upstream OpAICoreConfig / AICore().AddConfig() block has no counterpart
 * and is dropped -- it only tells the packer which SOCs to emit binaries for.
 *
 * Two things must hold for SetToContext() to work, and both are inherited from upstream unchanged:
 *
 *   - Every DataType list is the same length and *positionally parallel*: SetToContext() locates the
 *     query dtype in input 0's list and reuses that one index for every other input and output. So
 *     position 0 is the fp16 case throughout and position 1 the bf16 case, which is why the int32 and
 *     float lists below repeat their single dtype twice rather than listing it once.
 *   - Attr order fixes the attr indices the tiling reads (SCALE_VALUE_ATTR_INDEX = 0 ...
 *     RETURN_SOFTMAX_LSE_ATTR_INDEX = 8 in sparse_flash_attention_tiling.h). Do not reorder.
 *
 * One deliberate difference from upstream: the five attrs the tiling reads as int64_t
 * (GetAttrPointer<int64_t>) are declared with Int64() rather than Int(). ge_helper's Int() stores a
 * 32-bit int and GetAttrPointer type-checks with typeid, so Int() would fail the check at run time.
 * CANN's own OpDef::Attr::Int() is 64-bit, so upstream's `.Int(...)` and this `.Int64(...)` mean the
 * same thing.
 */
#ifndef SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_DEF_H
#define SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_DEF_H

#include <climits>
#include <cstdint>

#include "ge_helper.h"

namespace sglang {
namespace SFAHost {
using namespace ge_helper;

class SparseFlashAttention : public OpDef
{
public:
    explicit SparseFlashAttention(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_rope")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0F);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int64(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int64(3);  // 3:默认值，只计算下三角
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int64(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int64(INT64_MAX);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int64(2);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
    }
};

}  // namespace SFAHost
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_DEF_H
