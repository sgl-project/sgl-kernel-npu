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
 * \file lightning_indexer_v2_tiling_data.h
 * \brief Shared host/device tiling struct for the lightning_indexer_v2 op.
 */
#ifndef SGL_LI_V2_TILING_DATA_H
#define SGL_LI_V2_TILING_DATA_H
#include <cstdint>

namespace sglang {
namespace LIV2Host {

// -----------算子TilingData定义---------------
// 与上游 CANN 版本的 BEGIN_TILING_DATA_DEF(LITilingData) 字段一一对应。
// 这里不使用 #pragma pack: preTokens/nextTokens 为 64bit，设备侧按自然对齐读取。
// 64bit 字段前置，保证 host 与 device 侧布局一致且无隐式填充歧义。
struct LITilingData {
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    uint32_t bSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t gSize = 0U;
    uint32_t s1Size = 0U;
    uint32_t s2Size = 0U;
    uint32_t sparseCount = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t blockSize = 0U;
    uint32_t maxBlockNumPerBatch = 0U;
    uint32_t sparseMode = 0U;
    uint32_t returnValue = 0U;
    uint32_t tilingKey = 0U;
};

static_assert(sizeof(LITilingData) == 64, "LITilingData layout changed unexpectedly");

}  // namespace LIV2Host
}  // namespace sglang
#endif  // SGL_LI_V2_TILING_DATA_H
