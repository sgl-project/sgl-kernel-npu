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
 * \file kv_compress_epilog_tiling_data.h
 * \brief Plain tiling-data struct for kv_compress_epilog (A5-only).
 */

#ifndef KV_COMPRESS_EPILOG_TILING_DATA_H
#define KV_COMPRESS_EPILOG_TILING_DATA_H

#include <cstdint>

namespace sglang {

#pragma pack(push, 1)
struct KvCompressEpilogTilingData {
    int64_t bs;
    int64_t d;
    int64_t kvCacheCol;
    int64_t scaleCol;
    int64_t concatCol;
    int64_t padCol;
    int64_t quantMode;
    int64_t roundScale;
    int64_t perGroupSize;
    int64_t rowOfFormerBlock;
    int64_t rowOfTailBlock;
    int64_t rowLoopOfFormerBlock;
    int64_t rowLoopOfTailBlock;
    int64_t rowFactor;
    int64_t tailRowFactorOfFormerBlock;
    int64_t tailRowFactorOfTailBlock;
    int64_t layout;
    int64_t blockSize;
    int64_t valuePerToken;
    int64_t scalePerToken;
    int64_t blockStride;
    int32_t dtype;       // bit0: slot_mapping int64, bit1: kv_compress_cache e4m3fn
    uint32_t tilingKey;  // single tiling config, always 0
};
#pragma pack(pop)

}  // namespace sglang

#endif  // KV_COMPRESS_EPILOG_TILING_DATA_H
