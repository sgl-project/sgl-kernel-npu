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

#include "ascend_laser_attention_tiling.h"

#include <string>
#include <cinttypes>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


using namespace std;

namespace {
constexpr size_t CONST_ONES_SIZE = 16384 * 2;  // half
constexpr size_t CONST_ZERO_SIZE = 32 * 128 * 4; // float
constexpr size_t MAX_TOKEN = 2147483647;
const int ALIGNNUM = 16;
inline __attribute__((always_inline)) int32_t CeilDiv(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

inline __attribute__((always_inline)) int32_t Align(int32_t num, int32_t alignNum)
{
    if (alignNum != 0) {
        return (num + alignNum - 1) / alignNum * alignNum;
    } else {
        return num + alignNum - 1;
    }
}

}

namespace optiling {
class AscendLaserAttentionTiling {
public:
    AscendLaserAttentionTilingData tilingData;

    ge::graphStatus Tiling4LaserAttention(gert::TilingContext* context);

    ge::graphStatus DoTiling(gert::TilingContext* context);

    ge::graphStatus CheckTiling(gert::TilingContext* context);
};

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(0);
    if (queryShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *softmaxOut = context->GetOutputShape(0);
    if (softmaxOut == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t queryDimNum = static_cast<int32_t>(queryShape->GetDimNum());
    if (queryDimNum < 4) { // query dim num is 4
        return ge::GRAPH_FAILED;
    }
    softmaxOut->SetDimNum(queryDimNum - 1);
    softmaxOut->SetDim(0, queryShape->GetDim(0));
    softmaxOut->SetDim(1, queryShape->GetDim(1));
    softmaxOut->SetDim(2, queryShape->GetDim(2));    // index is 2

    gert::Shape *attnOut = context->GetOutputShape(1);
    if (attnOut == nullptr) {
        return ge::GRAPH_FAILED;
    }
    attnOut->SetDimNum(queryDimNum);
    attnOut->SetDim(0, queryShape->GetDim(0));
    attnOut->SetDim(1, queryShape->GetDim(1));
    attnOut->SetDim(2, queryShape->GetDim(2));    // index is 2
    attnOut->SetDim(3, queryShape->GetDim(3));    // index is 3

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType queryDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, queryDtype);
    context->SetOutputDataType(1, queryDtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendLaserAttentionTiling::DoTiling(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const auto aicNum = ascendcPlatform.GetCoreNumAic();

    auto col_size = tilingData.get_kSeqLength();
    if (tilingData.get_sparseMode() == 1) {
        col_size = tilingData.get_windowLen();
    }

    int32_t coreNumPerGroup = 1;
    int32_t factor = 2;
    if (col_size <= 8 * 1024 / factor) {    // value is 8 * 1024
        coreNumPerGroup = 1;
    } else if (col_size > 8 * 1024 / factor && col_size <= 16 * 1024 / factor) {    // value is 8、16、1024
        coreNumPerGroup = 2;    // 2 is coreNumPerGroup
    } else if (col_size > 16 * 1024 / factor && col_size <= 32 * 1024 / factor) {    // value is 16、32、1024
        coreNumPerGroup = 4;    // 4 is coreNumPerGroup
    } else {
        if (aicNum == 20) {    // 20 is aicNum
            coreNumPerGroup = 4;    // 4 is coreNumPerGroup
        } else {
            coreNumPerGroup = 8;    // 8 is coreNumPerGroup
        }
    }

    tilingData.set_coreNumPerGroup(coreNumPerGroup);
    tilingData.set_coreGroupNum(aicNum / coreNumPerGroup);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendLaserAttentionTiling::CheckTiling(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (tilingData.get_batchSize() == 0) {
        cout << context->GetNodeName() << "op [TilingData]: batchSize is 0. " << endl;
        return ge::GRAPH_FAILED;
    }

    if (tilingData.get_headNum() == 0) {
        cout << context->GetNodeName() << "op [TilingData]: headNum is 0. " << endl;
        return ge::GRAPH_FAILED;
    }

    if (tilingData.get_qSeqLength() == 0) {
        cout << context->GetNodeName() << "op [TilingData]: qSeqLength is 0. " << endl;
        return ge::GRAPH_FAILED;
    }

    if (tilingData.get_kSeqLength() == 0) {
        cout << context->GetNodeName() << "op [TilingData]: kSeqLength is 0. " << endl;
        return ge::GRAPH_FAILED;
    }

    if (tilingData.get_vSeqLength() == 0) {
        cout << context->GetNodeName() << "op [TilingData]: vSeqLength is 0. " << endl;
        return ge::GRAPH_FAILED;
    }

    // sparse场景下windowsLength需要是256的倍数
    if (tilingData.get_sparseMode() == 1 && (tilingData.get_windowLen() % 256 != 0)) {
        cout << context->GetNodeName() << "op [TilingData]: windowLen= " << tilingData.get_windowLen() << endl;
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AscendLaserAttentionTiling::Tiling4LaserAttention(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t inputNum = static_cast<int32_t>(context->GetComputeNodeInputNum());
    if (inputNum < 3) { // 3 is the number of inputs
        return ge::GRAPH_FAILED;
    }
    const auto queryShape = context->GetInputShape(0);
    const auto keyShape = context->GetInputShape(1);
    const auto valueShape = context->GetInputShape(2);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto pScale = context->GetAttrs()->GetAttrPointer<float>(0);
    const auto pHeadNum = context->GetAttrs()->GetAttrPointer<int32_t>(1);
    const auto pre_tokens = context->GetAttrs()->GetAttrPointer<int32_t>(4);
    const auto isHighPrecision = context->GetAttrs()->GetAttrPointer<bool>(6);
    if (pScale == nullptr || pHeadNum == nullptr || pre_tokens == nullptr || isHighPrecision == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto qkvShape = queryShape->GetStorageShape(); // [batchSize, headNum, seqSize, headDim]
    int32_t batchSize = qkvShape.GetDim(0);
    int32_t headNum = *pHeadNum;
    int32_t seqSize = qkvShape.GetDim(2);
    int32_t headDim = qkvShape.GetDim(3);
    int32_t qSeqLength = queryShape->GetStorageShape().GetDim(2);
    int32_t kSeqLength = keyShape->GetStorageShape().GetDim(2);
    int32_t vSeqLength = valueShape->GetStorageShape().GetDim(2);

    int32_t maskSeqLength = 0;

    tilingData.set_batchSize(batchSize);
    tilingData.set_headNum(headNum);
    tilingData.set_seqSize(seqSize);
    tilingData.set_headDim(headDim);

    tilingData.set_qSeqLength(qSeqLength);
    tilingData.set_kSeqLength(kSeqLength);
    tilingData.set_vSeqLength(vSeqLength);
    tilingData.set_maskSeqLength(maskSeqLength);
    tilingData.set_scale(*pScale);

    bool isTriangle = false;
    tilingData.set_isTriangle(isTriangle);
    int32_t attenType = 0;
    int32_t sparseMode = 0;
    int32_t headGroupSize = 1;
    int32_t windowLen = 0;
    int32_t queryDim = queryShape->GetStorageShape().GetDim(1);
    int32_t keyDim = keyShape->GetStorageShape().GetDim(1);
    if (keyDim <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (queryDim != keyDim) {
        attenType = 1;
        headGroupSize = queryDim / keyDim;
    }
    if (*pre_tokens == MAX_TOKEN) {
        windowLen = 0;
    } else {
        windowLen = *pre_tokens;
    }
    tilingData.set_attenType(attenType);
    tilingData.set_sparseMode(sparseMode);
    tilingData.set_headGroupSize(headGroupSize);
    tilingData.set_windowLen(windowLen);
    tilingData.set_isHighPrecision(*isHighPrecision);

    DoTiling(context);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AscendLaserAttentionTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    AscendLaserAttentionTiling tiling;
    tiling.Tiling4LaserAttention(context);

    // set block_dim & workspace
    const auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const auto aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum == 25) {    // 25 is aicNum
        context->SetBlockDim(24);    // 24 is blockdim
    } else {
        context->SetBlockDim(aicNum);
    }
    auto workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }

    size_t coreGrouNum = static_cast<size_t>(tiling.tilingData.get_coreGroupNum());
    size_t coreNumPerGroup = static_cast<size_t>(tiling.tilingData.get_coreNumPerGroup());
    size_t batchSize = static_cast<size_t>(tiling.tilingData.get_batchSize());
    size_t headNum = static_cast<size_t>(tiling.tilingData.get_headNum());
    size_t seqSize = static_cast<size_t>(tiling.tilingData.get_seqSize());
    size_t workspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());

    auto groupNum = coreGrouNum * coreNumPerGroup;
    auto rowSumSize = batchSize * headNum * seqSize * sizeof(float);
    workspaces[0] = groupNum * 128 * 128 * 32 * 2 * 4 +    // 128、32、2、4 is offset
                    groupNum * 256 * 128 * 8 * 2 * 4 * 2 +    // 256、128、8、2、4 is offset
                    rowSumSize + workspaceSize;

    tiling.tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.tilingData.GetDataSize());

    const auto inputData = context->GetInputTensor(0);
    if (inputData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = inputData->GetDataType();
    if (dtype == ge::DT_BF16) {
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(0); // FP16
    }

    return tiling.CheckTiling(context);
}
}

namespace ops {
    class AscendLaserAttention : public OpDef {
    public:
        explicit AscendLaserAttention(const char* name) : OpDef(name)
        {
            this->Input("query")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("key")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("value")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("atten_mask")
                    .ParamType(OPTIONAL)
                    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("alibi_mask")
                    .ParamType(OPTIONAL)
                    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("drop_mask")
                    .ParamType(OPTIONAL)
                    .DataType({ge::DT_UINT8, ge::DT_UINT8})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            // （qseqlen，1）
            this->Output("softmax_log_max_sum")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("attention_out")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("scale_value")
                    .AttrType(REQUIRED)
                    .Float();
            this->Attr("head_num")
                    .AttrType(REQUIRED)
                    .Int();
            this->Attr("input_layout")
                    .AttrType(REQUIRED)
                    .String();
            this->Attr("keep_prob")
                    .AttrType(OPTIONAL)
                    .Float(1.0);
            this->Attr("pre_tokens")
                    .AttrType(OPTIONAL)
                    .Int(MAX_TOKEN);    // 2147483647 is pre_tokens
            this->Attr("next_tokens")
                    .AttrType(OPTIONAL)
                    .Int(1);
            this->Attr("is_highPrecision")
                    .AttrType(OPTIONAL)
                    .Bool(true);

            this->SetInferShape(optiling::InferShape)
            .SetInferDataType(optiling::InferDtype);

            this->AICore().SetTiling(optiling::AscendLaserAttentionTilingFunc);
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(AscendLaserAttention);
} // namespace optiling
