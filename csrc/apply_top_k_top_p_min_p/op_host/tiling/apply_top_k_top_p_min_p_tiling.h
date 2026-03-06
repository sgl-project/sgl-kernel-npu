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
 * \file apply_top_k_top_p_min_p_tiling.h
 * \brief
 */

#ifndef APPLY_TOP_K_TOP_P_MIN_P_TILING_H_
#define APPLY_TOP_K_TOP_P_MIN_P_TILING_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "apply_top_k_top_p_min_p_tiling_data.h"
#include "ge_helper.h"

namespace sglang::ATKTPMPHost {
struct TensorParaInfo {
    ge::DataType dtype;
    c10::ArrayRef<int64_t> shape;
};

const std::map<ge::DataType, int64_t> G_DTYPE_MAP = {{ge::DT_FLOAT, 1}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 3}};
// ------------------算子原型索引常量定义----------------
// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
// Dim Num
constexpr uint32_t DIM_NUM_ZERO = 0;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;

constexpr int64_t COEF_TEN = 10;
constexpr int64_t BYTES_B32 = 4;
constexpr int64_t LOCAL_TENSOR_NUM = 4;
constexpr int64_t SELECT_MODE_BYTES = 8192;
constexpr int64_t BYTES_PER_REPEAT = 256;

// -----------算子Tiling入参结构体定义---------------
struct ApplyTopKTopPMinPParaInfo {
    TensorParaInfo probs = {ge::DT_FLOAT, c10::ArrayRef<int64_t>{}};
    TensorParaInfo k = {ge::DT_INT32, c10::ArrayRef<int64_t>{}};
    TensorParaInfo p = {ge::DT_FLOAT, c10::ArrayRef<int64_t>{}};
    TensorParaInfo minP = {ge::DT_FLOAT, c10::ArrayRef<int64_t>{}};
    TensorParaInfo sampledRes = {ge::DT_FLOAT, c10::ArrayRef<int64_t>{}};
};

// -----------算子Tiling入参信息类---------------
class ApplyTopKTopPMinPTilingInfo
{
public:
    ApplyTopKTopPMinPParaInfo opParamInfo;
    int64_t workspaceSize = 0;
    int64_t needMinPSample = 0;
};

// ---------------算子Tiling类---------------
class ApplyTopKTopPMinPTiling
{
public:
    explicit ApplyTopKTopPMinPTiling(ApplyTopKTopPMinPTilingInfo *tilingInfo) : tilingInfo_(tilingInfo) {};
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    void SplitTask();
    ge::graphStatus DoTiling();
    const ApplyTopKTopPMinPTilingData &GetTilingData() const;

private:
    ApplyTopKTopPMinPTilingInfo *tilingInfo_ = nullptr;
    ApplyTopKTopPMinPTilingData tilingData_;
};

}  // namespace sglang::ATKTPMPHost
#endif  // APPLY_TOP_K_TOP_P_MIN_P_TILING_H_
