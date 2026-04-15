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
class SparseBlockEstimate : public OpDef {
public:
    explicit SparseBlockEstimate(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
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
        this->Output("sparse_mask")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND});
        this->Output("sparse_count_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Attr("input_layout").AttrType(OPTIONAL).String("BNSD");
        this->Attr("stride").AttrType(OPTIONAL).Int(8);         // stride 大小，默认为8
        this->Attr("sparse_size").AttrType(OPTIONAL).Int(128);  // sparse 块大小，默认为128
        this->Attr("num_heads").AttrType(OPTIONAL).Int(1);
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(1);
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);  // 缩放因子
        this->Attr("threshold").AttrType(OPTIONAL).Float(1.0);
        this->Attr("causal").AttrType(OPTIONAL).Bool(false);
        this->Attr("keep_sink").AttrType(OPTIONAL).Bool(true);
        this->Attr("keep_recent").AttrType(OPTIONAL).Bool(true);
        this->Attr("row_sparse").AttrType(OPTIONAL).Float(1.0);  // ROW_SPARSE 强制稀疏率，当设置大于等于 1
                                                                 // 时不生效，0~1之间生效。 0.4 时表示强制保留 top-40%
                                                                 // (即稀疏率60%)

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SparseBlockEstimate);
}  // namespace ops