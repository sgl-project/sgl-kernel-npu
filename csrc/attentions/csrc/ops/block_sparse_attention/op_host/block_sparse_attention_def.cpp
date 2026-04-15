/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "register/op_def_registry.h"


namespace ops {
class BlockSparseAttention : public OpDef {
public:
    explicit BlockSparseAttention(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,
                ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("pse_shift")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_INT8,
                ge::DT_INT8, ge::DT_INT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_BOOL, ge::DT_INT8,
                ge::DT_UINT8, ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_INT8,
                ge::DT_INT8, ge::DT_INT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_BOOL, ge::DT_INT8,
                ge::DT_UINT8, ge::DT_BOOL, ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("actual_seq_lengths")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("deq_scale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("quant_scale1")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("deq_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("quant_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("quant_offset2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("sparse_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8, ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND});
        this->Input("sparse_count_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16,
                ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16,
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16,
                ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8,
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND});
        this->Attr("num_heads")
            .AttrType(REQUIRED)
            .Int(1);
        this->Attr("scale_value")
            .AttrType(OPTIONAL)
            .Float(1.0);
        this->Attr("pre_tokens")
            .AttrType(OPTIONAL)
            .Int(214748647); // default 214748647
        this->Attr("next_tokens")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("input_layout")
            .AttrType(OPTIONAL)
            .String("BSH");
        this->Attr("num_key_value_heads")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("sparse_mode")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("inner_precise")
            .AttrType(OPTIONAL)
            .Int(1);
        this->Attr("sparse_size")
            .AttrType(OPTIONAL)
            .Int(1);
        this->Attr("causal")
            .AttrType(OPTIONAL)
            .Bool(false);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")   // set value of aclnn support
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false"); // set jit compile flag
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BlockSparseAttention);
}
