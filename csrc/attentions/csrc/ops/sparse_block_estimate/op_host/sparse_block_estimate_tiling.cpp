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


#include <queue>
#include <vector>
#include <string>
// #include <iostream>  // 必须引入 std::cout/std::endl 头文件

#include <unordered_map>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "sparse_block_estimate_tiling.h"

#include <cstdint>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>

using std::string;

using namespace matmul_tiling;

namespace optiling {
constexpr uint32_t NUM_2 = 2;
int32_t SINGLE_CORE_MBASE = 128;
int32_t SINGLE_CORE_NBASE = 1024;

constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 2;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 3;
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;

constexpr uint32_t INPUT_LAYOUT = 0;
constexpr uint32_t STRIDE = 1;
constexpr uint32_t SPARSE_SIZE = 2;
constexpr uint32_t NUM_HEADS_ATTR = 3;
constexpr uint32_t NUM_KV_HEADS_ATTR = 4;
constexpr uint32_t SCALE_VALUE = 5;
constexpr uint32_t THRESHOLD = 6;
constexpr uint32_t CAUSAL = 7;
constexpr uint32_t KEEP_SINK = 8;
constexpr uint32_t KEEP_RECENT = 9;
constexpr uint32_t ATTR_ROW_SPARSE = 10;
constexpr uint32_t DIM0_INDEX = 0;
constexpr uint32_t DIM1_INDEX = 1;
constexpr uint32_t DIM2_INDEX = 2;
constexpr uint32_t DIM3_INDEX = 3;

uint64_t BASE_TILING_KEY = 1000000000000000000;  // 默认 TILING_KEY

void PromptFlashAttentionSplitNSNew(SparseBlockEstimateTilingData &tiling, uint32_t curCoreNum,
    std::vector<int64_t> &actualSeqLengths, std::vector<int64_t> &actualSeqLengthsKV, int64_t actualSharedPrefixLen,
    bool useBalanceTiling)
{
    SparseBlockEstimateSeqParams *seqParams = &tiling.sparseBlockEstimateSeqParams;

    uint32_t arrayLen = tiling.get_batchSize();  // batch size
    uint32_t sOuterSize = SINGLE_CORE_MBASE * tiling.get_stride();
    uint32_t sInnerSize = SINGLE_CORE_NBASE * tiling.get_stride();

    std::vector<uint32_t> accumSOuterTilingNums(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sInnerLoopTimes(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sOuterBlockNums(static_cast<size_t>(arrayLen), 0U);

    const size_t tilingElementArrayLen =
        (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    std::vector<uint32_t> coreSposEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSposStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidStart(tilingElementArrayLen, 0U);

    int64_t totalBlockWight = 0;
    int totalOuterBlockNum = 0;
    uint32_t preAccumSOuterNum = 0U;
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    uint32_t sInnerPrefixLoopTimes = (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
    bool isSOuterNoTail = true;
    bool isSInnerNoTail = true;
    bool causal = tiling.get_causal();
    for (uint32_t i = 0; i < arrayLen; i++) {
        int seqLen = actualSeqLengths[i];
        int subSeqInnerLen = actualSeqLengthsKV[i];
        sOuterBlockNums[i] = (seqLen + sOuterSize - 1) / sOuterSize;
        sInnerLoopTimes[i] = (subSeqInnerLen + sInnerSize - 1) / sInnerSize + sInnerPrefixLoopTimes;
        accumSOuterTilingNums[i] = (sOuterBlockNums[i] * tiling.get_headNumQ()) + preAccumSOuterNum;
        preAccumSOuterNum = accumSOuterTilingNums[i];

        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[i]);

        if (seqLen % sOuterSize != 0) {
            isSOuterNoTail = false;
        }
        if (subSeqInnerLen % sInnerSize != 0) {
            isSInnerNoTail = false;
        }
        totalOuterBlockNum += sOuterBlockNums[i];
        if (causal) {
            totalBlockWight += (static_cast<int64_t>(sOuterBlockNums[i]) + 1) *
                               static_cast<int64_t>(sOuterBlockNums[i]) / NUM_2;  // div 2
        } else {
            totalBlockWight += static_cast<int64_t>(sOuterBlockNums[i]) * static_cast<int64_t>(sInnerLoopTimes[i]);
        }
    }
    if ((!useBalanceTiling)) {
        accumSOuterTilingNums[0] = 0;
    }

    float coreWightTarget = (float(totalBlockWight * tiling.get_headNumQ()) / float(curCoreNum));

    int curWight = 0;
    int curCore = 0;
    coreSposStart[curCore] = 0;
    coreSidStart[curCore] = 0;
    coreNidStart[curCore] = 0;
    for (uint32_t i = 0; i < tiling.get_headNumQ(); i++) {
        for (uint32_t j = 0; j < arrayLen; j++) {
            for (uint32_t k = 0; k < sOuterBlockNums[j]; k++) {
                int64_t dif = int64_t(coreWightTarget * float(curCore + 1)) - curWight;
                int64_t curWightPlus;
                if (causal) {
                    curWightPlus = k + 1;
                } else {
                    curWightPlus = sInnerLoopTimes[j];
                }
                if ((curWightPlus - dif) > dif) {
                    if (k == 0) {
                        if (j == 0) {
                            coreNidEnd[curCore] = i;
                            coreSidEnd[curCore] = arrayLen;
                            coreSposEnd[curCore] = sOuterBlockNums[arrayLen - 1];
                        } else {
                            coreNidEnd[curCore] = i + 1;
                            coreSidEnd[curCore] = j;
                            coreSposEnd[curCore] = sOuterBlockNums[j - 1];
                        }
                    } else {
                        coreNidEnd[curCore] = i + 1;
                        coreSidEnd[curCore] = j + 1;
                        coreSposEnd[curCore] = k;
                    }
                    curCore += 1;
                    coreNidStart[curCore] = i;
                    coreSidStart[curCore] = j;
                    coreSposStart[curCore] = k;
                }
                curWight += curWightPlus;
            }
        }
    }

    coreNidEnd[curCore] = (tiling.get_headNumQ());
    coreSidEnd[curCore] = arrayLen;
    coreSposEnd[curCore] = sOuterBlockNums[arrayLen - 1];

    // Temporary reuse
    seqParams->set_coreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());

    uint32_t actualCoreNums = curCore + 1;
    tiling.set_actualCoreNums(actualCoreNums);
}

static ge::graphStatus SetDataShape(
    SparseBlockEstimateTilingData &tiling, gert::TilingContext *context, const string &layoutStr)
{
    uint32_t batchSize;
    uint32_t dim;
    uint32_t qs;
    uint32_t kvs;
    auto attrs = context->GetAttrs();
    uint32_t qHeadNum = *attrs->GetAttrPointer<int32_t>(NUM_HEADS_ATTR);
    uint32_t kvHeadNum = *attrs->GetAttrPointer<int32_t>(NUM_KV_HEADS_ATTR);
    const gert::StorageShape *QueryShape = context->GetInputShape(0);
    const gert::StorageShape *KeyShape = context->GetInputShape(1);

    const gert::Tensor *tempData = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    const gert::Tensor *tempDataKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    uint32_t actualLenDims = (tempData != nullptr && tempData->GetData<int64_t>() != nullptr)
                                 ? tempData->GetShapeSize()
                                 : 0;  // ! len of act_seq_list
    uint32_t actualLenDimsKV =
        (tempDataKV != nullptr && tempDataKV->GetData<int64_t>() != nullptr) ? tempDataKV->GetShapeSize() : 0;

    if (layoutStr == "BNSD") {
        batchSize = QueryShape->GetStorageShape().GetDim(DIM0_INDEX);
        dim = QueryShape->GetStorageShape().GetDim(DIM3_INDEX);
        qs = QueryShape->GetStorageShape().GetDim(DIM2_INDEX);
        kvs = KeyShape->GetStorageShape().GetDim(DIM2_INDEX);
    } else if (layoutStr == "BSND") {
        batchSize = QueryShape->GetStorageShape().GetDim(DIM0_INDEX);
        dim = QueryShape->GetStorageShape().GetDim(DIM3_INDEX);
        qs = QueryShape->GetStorageShape().GetDim(DIM1_INDEX);
        kvs = KeyShape->GetStorageShape().GetDim(DIM1_INDEX);
    } else if (layoutStr == "BSH") {
        batchSize = QueryShape->GetStorageShape().GetDim(0);
        qs = QueryShape->GetStorageShape().GetDim(DIM1_INDEX);
        kvs = KeyShape->GetStorageShape().GetDim(DIM1_INDEX);
        dim = QueryShape->GetStorageShape().GetDim(DIM2_INDEX) / qHeadNum;
    } else if (layoutStr == "TND") {
        if (actualLenDims == 0 || actualLenDimsKV == 0) {
            return ge::GRAPH_FAILED;
        }
        batchSize = actualLenDims;
        qs = QueryShape->GetStorageShape().GetDim(0);
        kvs = KeyShape->GetStorageShape().GetDim(0);
        dim = QueryShape->GetStorageShape().GetDim(DIM2_INDEX);
    } else {
        return ge::GRAPH_FAILED;
    }
    tiling.set_headNumQ(qHeadNum);
    tiling.set_headNumKV(kvHeadNum);
    tiling.set_seqLenQ(qs);
    tiling.set_seqLenK(kvs);
    tiling.set_dim(dim);
    tiling.set_batchSize(batchSize);
    tiling.set_actualSeqLengthsSize(actualLenDims);
    tiling.set_actualSeqLengthsKVSize(actualLenDimsKV);
    return ge::GRAPH_SUCCESS;
}

static void setTilingKey(gert::TilingContext *context, bool causal, ge::DataType qDataType, const string &layoutStr)
{
    uint64_t tilingKey = BASE_TILING_KEY;
    tilingKey += causal ? 1U : 0U;

    if (layoutStr == "BSND" || layoutStr == "BSH") {
        tilingKey += 10U;
    } else if (layoutStr == "TND") {
        tilingKey += 20U;
    }

    if (qDataType == ge::DT_BF16) {
        tilingKey += 100U;
    }

    auto ret = context->SetTilingKey(tilingKey);
}

ge::graphStatus SparseBlockEstimateTilingFunc(gert::TilingContext *context)
{
    auto db = 2;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto attrs = context->GetAttrs();
    uint32_t coresAic = ascendcPlatform.GetCoreNumAic();
    uint32_t coresAiv = ascendcPlatform.GetCoreNumAiv();
    bool inputIsNullPtr =
        (context->GetInputDesc(QUERY_INDEX) == nullptr) || (context->GetInputDesc(KEY_INDEX) == nullptr) ||
        (context->GetInputShape(QUERY_INDEX) == nullptr) || (context->GetInputShape(KEY_INDEX) == nullptr);
    if (inputIsNullPtr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    ge::DataType kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    if ((qDataType != ge::DT_BF16 && qDataType != ge::DT_FLOAT16) ||
        (kDataType != ge::DT_BF16 && kDataType != ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }

    SparseBlockEstimateTilingData tiling;

    uint32_t sparseSize = *attrs->GetAttrPointer<int32_t>(SPARSE_SIZE);
    const char *layout = attrs->GetAttrPointer<char>(INPUT_LAYOUT);

    std::string layoutStr = std::string(layout);
    if (SetDataShape(tiling, context, layoutStr) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_sparseSize(sparseSize);
    tiling.set_scaleFactor(*attrs->GetAttrPointer<float>(SCALE_VALUE));
    tiling.set_threshold(*attrs->GetAttrPointer<float>(THRESHOLD));
    bool causal = *attrs->GetAttrPointer<bool>(CAUSAL);
    tiling.set_causal(causal);
    tiling.set_setFirstCol(*attrs->GetAttrPointer<bool>(KEEP_SINK));
    tiling.set_setDiag(*attrs->GetAttrPointer<bool>(KEEP_RECENT));
    float rowSparse = *attrs->GetAttrPointer<float>(ATTR_ROW_SPARSE);
    tiling.set_rowSparse(rowSparse);

    int32_t stride = *attrs->GetAttrPointer<int32_t>(STRIDE);
    tiling.set_stride(stride);

    while (tiling.get_seqLenK() < stride * db * SINGLE_CORE_NBASE) {
        SINGLE_CORE_NBASE /= 2; // 2：数据减半
    }
    SINGLE_CORE_NBASE = std::max(SINGLE_CORE_NBASE, 512); // 512: 下限阈值

    auto tasks = tiling.get_headNumQ() * tiling.get_seqLenQ(); // 6038
    if (coresAiv == 0) {
        return ge::GRAPH_FAILED;
    }
    tasks = (tasks + coresAiv - 1) / coresAiv; // 151
    tasks = (tasks + sparseSize -1) / sparseSize; // 2
    SINGLE_CORE_MBASE = std::min(SINGLE_CORE_MBASE, int(tasks * sparseSize / stride));
    tiling.set_sOuterFactor(SINGLE_CORE_MBASE);
    tiling.set_sInnerFactor(SINGLE_CORE_NBASE);

    MatmulApiTiling cubeTiling(ascendcPlatform);
    auto mmInputType =
        qDataType == ge::DT_BF16 ? matmul_tiling::DataType::DT_BF16 : matmul_tiling::DataType::DT_FLOAT16;
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, mmInputType, false);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, mmInputType, true);
    cubeTiling.SetCType(TPosition::GM,
        CubeFormat::ND_ALIGN,
        matmul_tiling::DataType::DT_FLOAT);  // 不管input是fp16还是bf16 保持fp 32不动
    uint32_t seqLenQDivStride = (tiling.get_seqLenQ() + stride - 1) / stride;
    uint32_t seqLenKDivStride = (tiling.get_seqLenK() + stride - 1) / stride;
    auto dim = tiling.get_dim();
    cubeTiling.SetOrgShape(seqLenQDivStride, seqLenKDivStride, dim * stride);
    cubeTiling.SetShape(SINGLE_CORE_MBASE, SINGLE_CORE_NBASE, dim * stride);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    cubeTiling.SetFixSplit(std::min(SINGLE_CORE_MBASE, 128), std::min(SINGLE_CORE_NBASE, 128), 128); // 128: 上限阈值

    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {  // Get matmul tiling.
        return ge::GRAPH_FAILED;
    }
    tiling.cubeTilingData.set_dbL0C(2); // 2: db

    context->SetBlockDim(coresAic);
    tiling.set_coreNumAic(coresAic);

    // 分核
    bool useBalanceTiling = true;
    uint32_t gqa = tiling.get_headNumQ() / tiling.get_headNumKV();
    if (gqa != 1) {
        useBalanceTiling = false;
    }
    const gert::Tensor *tempData = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    const gert::Tensor *tempDataKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    size_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
    size_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
    std::vector<int64_t> actualSeqLengths;
    std::vector<int64_t> actualSeqLengthsKV;
    auto batchSize = tiling.get_batchSize();
    actualSeqLengths.resize(batchSize);
    actualSeqLengthsKV.resize(batchSize);
    for (size_t i = 0; i < batchSize; i++) {
        if (layoutStr != "TND") {
            if ((actualLenDims == 0) || (tempData->GetData<int64_t>() == nullptr)) {
                actualSeqLengths[i] = tiling.get_seqLenQ();
            } else {
                actualSeqLengths[i] = (actualLenDims > 1) ? static_cast<uint32_t>(tempData->GetData<int64_t>()[i])
                                                          : static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
            }
            if ((actualLenDimsKV == 0) ||
                (tempDataKV->GetData<int64_t>() == nullptr)) {  // The user did not input act_seq_kv
                actualSeqLengthsKV[i] = tiling.get_seqLenK();
            } else {
                actualSeqLengthsKV[i] = (actualLenDimsKV > 1)
                                            ? static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[i])
                                            : static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
            }
        } else {
            // TND
            actualSeqLengths[i] = (i == 0) ? tempData->GetData<int64_t>()[i]
                                           : tempData->GetData<int64_t>()[i] - tempData->GetData<int64_t>()[i - 1];
        }
    }
    int64_t actualSharedPrefixLen = 0;
    PromptFlashAttentionSplitNSNew(
        tiling, coresAiv, actualSeqLengths, actualSeqLengthsKV, actualSharedPrefixLen, useBalanceTiling);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    setTilingKey(context, causal, qDataType, layoutStr);

    size_t userWorkspaceSize =
        coresAiv * (SINGLE_CORE_MBASE * sizeof(float) * (SINGLE_CORE_NBASE + 32) * (db * 2 /* first reduce + qk */) +
                       SINGLE_CORE_MBASE * sizeof(half) * db * stride * tiling.get_dim());  // reorderq
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSparseBlockEstimate(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(SparseBlockEstimate)
    .Tiling(SparseBlockEstimateTilingFunc)
    .TilingParse<SparseBlockEstimateCompileInfo>(TilingPrepareForSparseBlockEstimate);

} // namespace optiling