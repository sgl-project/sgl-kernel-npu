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

#include <string>
#include <cinttypes>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "la_preprocess_tiling.h"

using namespace std;

namespace optiling {

namespace {
constexpr int OUTPUTINDEX0 = 0;
constexpr int OUTPUTINDEX1 = 1;
constexpr int OUTPUTINDEX2 = 2;
constexpr int DEFAULTALIGNLEN = 256;
constexpr int SEQ_LEN_DIM = 2;
constexpr int HEAD_DIM_DIM = 3;
constexpr int INPUT_HEAD_NUM_DIM = 2;
constexpr int INPUT_HEAD_DIM_DIM = 3;
}


static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    const gert::Shape *qShape = context->GetInputShape(0);
    const gert::Shape *kShape = context->GetInputShape(1);
    const gert::Shape *vShape = context->GetInputShape(2);
    gert::Shape *outQShape = context->GetOutputShape(0);
    gert::Shape *outKShape = context->GetOutputShape(1);
    gert::Shape *outVShape = context->GetOutputShape(2);

    if (qShape == nullptr || kShape == nullptr || vShape == nullptr ||
        outQShape == nullptr || outKShape == nullptr || outVShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto alignLen = *context->GetAttrs()->GetAttrPointer<int32_t>(0);

    // 输出形状: [batch, head_num, padded_seq_len, head_dim]
    outQShape->SetDimNum(qShape->GetDimNum());
    outQShape->SetDim(0, qShape->GetDim(0));  // batch
    outQShape->SetDim(1, qShape->GetDim(2));  // head_num (从第2维移到第1维)
    int32_t qPadDim = (qShape->GetDim(1) + alignLen - 1) / alignLen * alignLen;  // padded seq_len
    outQShape->SetDim(SEQ_LEN_DIM, qPadDim);
    outQShape->SetDim(HEAD_DIM_DIM, qShape->GetDim(INPUT_HEAD_DIM_DIM));  // head_dim

    outKShape->SetDimNum(kShape->GetDimNum());
    outKShape->SetDim(0, kShape->GetDim(0));
    outKShape->SetDim(1, kShape->GetDim(INPUT_HEAD_NUM_DIM));
    int32_t kPadDim = (kShape->GetDim(1) + alignLen - 1) / alignLen * alignLen;
    outKShape->SetDim(SEQ_LEN_DIM, kPadDim);
    outKShape->SetDim(HEAD_DIM_DIM, kShape->GetDim(INPUT_HEAD_DIM_DIM));

    outVShape->SetDimNum(vShape->GetDimNum());
    outVShape->SetDim(0, vShape->GetDim(0));
    outVShape->SetDim(1, vShape->GetDim(INPUT_HEAD_NUM_DIM));
    int32_t vPadDim = (vShape->GetDim(1) + alignLen - 1) / alignLen * alignLen;
    outVShape->SetDim(SEQ_LEN_DIM, vPadDim);
    outVShape->SetDim(HEAD_DIM_DIM, vShape->GetDim(INPUT_HEAD_DIM_DIM));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUTINDEX0, ge::DT_FLOAT16);
    context->SetOutputDataType(OUTPUTINDEX1, ge::DT_FLOAT16);
    context->SetOutputDataType(OUTPUTINDEX2, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LaPreprocessTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* qShape = context->GetInputShape(0);
    const gert::StorageShape* kShape = context->GetInputShape(1);
    const gert::StorageShape* vShape = context->GetInputShape(2);
    
    if (qShape == nullptr || kShape == nullptr || vShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(0));
    uint32_t qSeqLen = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(1));
    uint32_t kSeqLen = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(1));
    uint32_t vSeqLen = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(1));
    uint32_t headNum = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(2));
    uint32_t headDim = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(3));

    if (context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto alignLen = *(context->GetAttrs()->GetAttrPointer<int32_t>(0));
    if (context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dataType = context->GetInputDesc(0)->GetDataType();

    uint32_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = 1;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivecNum = platformInfo.GetCoreNumAiv();

    LaPreprocessTiling tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_qSeqLen(qSeqLen);
    tiling.set_kSeqLen(kSeqLen);
    tiling.set_vSeqLen(vSeqLen);
    tiling.set_headNum(headNum);
    tiling.set_headDim(headDim);
    tiling.set_alignLen(alignLen);
    tiling.set_ubSize(ubSize);

    context->SetBlockDim(aivecNum);
    context->SetTilingKey(tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class LaPreprocess : public OpDef {
public:
    explicit LaPreprocess(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("out_query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("align_len").AttrType(OPTIONAL).Int(optiling::DEFAULTALIGNLEN);

        this->SetInferShape(optiling::InferShape)
            .SetInferDataType(optiling::InferDataType);

        this->AICore().SetTiling(optiling::LaPreprocessTilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

// 算子注册
OP_ADD(LaPreprocess);

} // namespace ops
