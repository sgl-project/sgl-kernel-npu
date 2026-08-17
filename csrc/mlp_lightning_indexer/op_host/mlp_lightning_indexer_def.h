#ifndef MLP_LIGHTNING_INDEXER_DEF_H
#define MLP_LIGHTNING_INDEXER_DEF_H

#include <cstdint>

#include "ge_helper.h"

namespace sglang::MlpLIHost {
using namespace ge_helper;

constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;
constexpr uint32_t ATTR_BLOCK_LEN_INDEX = 3;
constexpr uint32_t ATTR_Q_BLOCK_LEN_INDEX = 4;
constexpr uint32_t ATTR_INIT_NUM_INDEX = 5;
constexpr uint32_t ATTR_LOCAL_NUM_INDEX = 6;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 7;
constexpr uint32_t ATTR_PRE_TOKENS_INDEX = 8;
constexpr uint32_t ATTR_NEXT_TOKENS_INDEX = 9;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 10;

class MlpLightningIndexer : public OpDef {
public:
    explicit MlpLightningIndexer(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Input("cur_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("cur_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("init_tensor").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("local_tensor").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_key").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_count").AttrType(OPTIONAL).Int(2048);
        this->Attr("kv_block_len").AttrType(OPTIONAL).Int(1);
        this->Attr("q_block_len").AttrType(OPTIONAL).Int(1);
        this->Attr("init_num").AttrType(OPTIONAL).Int(0);
        this->Attr("local_num").AttrType(OPTIONAL).Int(0);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(0);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(0);
        this->Attr("return_value").AttrType(OPTIONAL).Int(0);

        SetAttrAny("pre_tokens", static_cast<int64_t>(INT64_MAX));
        SetAttrAny("next_tokens", static_cast<int64_t>(INT64_MAX));
        SetAttrAny("return_value", false);
    }
};

}  // namespace sglang::MlpLIHost

#endif  // MLP_LIGHTNING_INDEXER_DEF_H
