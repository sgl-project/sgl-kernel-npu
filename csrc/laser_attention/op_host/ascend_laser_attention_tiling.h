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

#ifndef ASCEND_LASER_ATTENTION_H_
#define ASCEND_LASER_ATTENTION_H_

#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(AscendLaserAttentionTilingData)
TILING_DATA_FIELD_DEF(int32_t, batchSize);       // B
TILING_DATA_FIELD_DEF(int32_t, headNum);         // N
TILING_DATA_FIELD_DEF(int32_t, seqSize);         // S
TILING_DATA_FIELD_DEF(int32_t, headDim);         // D
TILING_DATA_FIELD_DEF(int32_t, coreNumPerGroup); // Y
TILING_DATA_FIELD_DEF(int32_t, coreGroupNum);    // F

TILING_DATA_FIELD_DEF(int32_t, qSeqLength);      // qkv不等长
TILING_DATA_FIELD_DEF(int32_t, kSeqLength);      // qkv不等长
TILING_DATA_FIELD_DEF(int32_t, vSeqLength);      // qkv不等长
TILING_DATA_FIELD_DEF(int32_t, maskSeqLength);   // 预留
TILING_DATA_FIELD_DEF(float, scale);             // 预留
TILING_DATA_FIELD_DEF(float, keep_prob);         // 预留
TILING_DATA_FIELD_DEF(int32_t, pre_tokens);      // 预留
TILING_DATA_FIELD_DEF(int32_t, next_tokens);     // 预留

TILING_DATA_FIELD_DEF(bool, isTriangle);        // 是否倒三角
TILING_DATA_FIELD_DEF(int32_t, attenType);       // 0:MHA/1:GQA
TILING_DATA_FIELD_DEF(int32_t, sparseMode);      // 0:dense/1:sparse
TILING_DATA_FIELD_DEF(int32_t, headGroupSize);   // N/G
TILING_DATA_FIELD_DEF(int32_t, windowLen);       // sparse的滑动窗口
TILING_DATA_FIELD_DEF(bool, isHighPrecision);    // 高性能
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendLaserAttention, AscendLaserAttentionTilingData)
}  // namespace optiling
#endif // LASER_ATTENTION_H_
