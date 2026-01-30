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
 * \file apply_top_k_top_p_min_p_tiling.cpp
 * \brief
 */

#include "apply_top_k_top_p_min_p_tiling.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace sglang::ATKTPMPHost {

// --------------------------ApplyTopKTopPMinPTiling类成员函数定义-----------------------
ge::graphStatus ApplyTopKTopPMinPTiling::CheckDtype()
{
    TORCH_CHECK((tilingInfo_->opParamInfo.probs.dtype == ge::DT_FLOAT16) ||
                    (tilingInfo_->opParamInfo.probs.dtype == ge::DT_BF16) ||
                    (tilingInfo_->opParamInfo.probs.dtype == ge::DT_FLOAT),
                "The data types of probs, p and sampled_res must be float16, bfloat16 or float.");

    TORCH_CHECK(tilingInfo_->opParamInfo.probs.dtype == tilingInfo_->opParamInfo.p.dtype,
                "The data types of probs and p must be the same.");
    TORCH_CHECK(tilingInfo_->opParamInfo.probs.dtype == tilingInfo_->opParamInfo.sampledRes.dtype,
                "The data types of probs and sampled_res must be the same.");

    TORCH_CHECK(tilingInfo_->opParamInfo.k.dtype == ge::DT_INT32, "The data types of the input k must be int32.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyTopKTopPMinPTiling::CheckShape()
{
    TORCH_CHECK(tilingInfo_->opParamInfo.probs.shape.size() == DIM_NUM_TWO,
                "ApplyTopKTopPMinP: the dimNum of probs should be ", DIM_NUM_TWO, ", but now is ",
                tilingInfo_->opParamInfo.probs.shape.size(), ".");
    tilingData_.batchSize = tilingInfo_->opParamInfo.probs.shape[DIM_IDX_ZERO];
    tilingData_.vocabSize = tilingInfo_->opParamInfo.probs.shape[DIM_IDX_ONE];

    TORCH_CHECK(tilingInfo_->opParamInfo.k.shape.size() == DIM_NUM_ONE, "ApplyTopKTopPMinP: the dimNum of k should be ",
                DIM_NUM_ONE, ", but now is ", tilingInfo_->opParamInfo.k.shape.size(), ".");
    int64_t kSize = tilingInfo_->opParamInfo.k.shape[DIM_IDX_ZERO];
    TORCH_CHECK(kSize == tilingData_.batchSize, "ApplyTopKTopPMinP: the shape of k should be [", tilingData_.batchSize,
                "], but now is [", kSize, "].");

    TORCH_CHECK(tilingInfo_->opParamInfo.p.shape.size() == DIM_NUM_ONE, "ApplyTopKTopPMinP: the dimNum of p should be ",
                DIM_NUM_ONE, ", but now is ", tilingInfo_->opParamInfo.p.shape.size(), ".");
    int64_t pSize = tilingInfo_->opParamInfo.p.shape[DIM_IDX_ZERO];
    TORCH_CHECK(pSize == tilingData_.batchSize, "ApplyTopKTopPMinP: the shape of p should be [", tilingData_.batchSize,
                "], but now is [", pSize, "].");

    if (tilingInfo_->opParamInfo.minP.shape.size() != DIM_NUM_ZERO) {
        int64_t minPSize = tilingInfo_->opParamInfo.minP.shape[DIM_IDX_ZERO];
        TORCH_CHECK(minPSize == tilingData_.batchSize, ": the shape of p should be [", tilingData_.batchSize,
                    "], but now is [", minPSize, "].");
        tilingInfo_->needMinPSample = 1;
    }

    TORCH_CHECK(tilingInfo_->opParamInfo.sampledRes.shape.size() == DIM_NUM_TWO,
                "ApplyTopKTopPMinP: the dimNum of sampled_res should be ", DIM_NUM_TWO, ", but now is ",
                tilingInfo_->opParamInfo.sampledRes.shape.size(), ".");
    int64_t sampledResSize0 = tilingInfo_->opParamInfo.sampledRes.shape[DIM_IDX_ZERO];
    int64_t sampledResSize1 = tilingInfo_->opParamInfo.sampledRes.shape[DIM_IDX_ONE];
    TORCH_CHECK(sampledResSize0 == tilingData_.batchSize && sampledResSize1 == tilingData_.vocabSize,
                "ApplyTopKTopPMinP: the size of sampledRes should be [", tilingData_.batchSize, ", ",
                tilingData_.vocabSize, "], but now is [", sampledResSize0, ", ", sampledResSize1, "].");
    return ge::GRAPH_SUCCESS;
}

void ApplyTopKTopPMinPTiling::SplitTask()
{
    tilingData_.loopDataNum = tilingData_.ubSize / BYTES_B32 / LOCAL_TENSOR_NUM / BYTES_PER_REPEAT * BYTES_PER_REPEAT;
    tilingData_.coreNum = tilingData_.batchSize > tilingData_.coreNum ? tilingData_.coreNum : tilingData_.batchSize;
    tilingData_.batchPerCore = tilingData_.batchSize / std::max(tilingData_.coreNum, static_cast<int64_t>(1));
    tilingData_.batchTailCore = tilingData_.batchSize - tilingData_.batchPerCore * tilingData_.coreNum;
}

ge::graphStatus ApplyTopKTopPMinPTiling::DoTiling()
{
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    TORCH_CHECK(aivNum != 0 && aivNum != 0, "num of core obtained is 0");
    tilingData_.coreNum = static_cast<int64_t>(aivNum);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData_.ubSize = static_cast<int64_t>(ubSize) - SELECT_MODE_BYTES;

    auto socVersion = ascendcPlatform.GetSocVersion();
    TORCH_CHECK(socVersion == platform_ascendc::SocVersion::ASCEND910B ||
                    socVersion == platform_ascendc::SocVersion::ASCEND910_93,
                "soc version does not support ", (int32_t)socVersion);

    SplitTask();

    // -------------set workspacesize-----------------
    tilingInfo_->workspaceSize = static_cast<int64_t>(ascendcPlatform.GetLibApiWorkSpaceSize()) +
                                 tilingData_.batchSize * tilingData_.vocabSize * BYTES_B32;

    // -------------set tilingkey-----------------
    tilingData_.tilingKey =
        G_DTYPE_MAP.at(tilingInfo_->opParamInfo.probs.dtype) * COEF_TEN + tilingInfo_->needMinPSample;

    return ge::GRAPH_SUCCESS;
}

const ApplyTopKTopPMinPTilingData &ApplyTopKTopPMinPTiling::GetTilingData() const
{
    return tilingData_;
}
}  // namespace sglang::ATKTPMPHost
