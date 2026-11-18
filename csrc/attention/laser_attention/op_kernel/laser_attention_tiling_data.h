/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Provenance: csrc/attentions/csrc/ops/laser_attention/op_host/laser_attention_tiling.h
// The TilingData macros are replaced by one plain struct shared by Host and Kernel.
// Field names, order and types are unchanged, and the legacy get_/set_ accessors
// are kept so the copied Tiling logic needs no rewrite.

#ifndef LASER_ATTENTION_TILING_DATA_H
#define LASER_ATTENTION_TILING_DATA_H

#include <cstdint>

namespace optiling {

struct SglLaserAttentionTilingData {
    int32_t batchSize;        // B
    int32_t headNum;          // N
    int32_t seqSize;          // S
    int32_t headDim;          // D
    int32_t coreNumPerGroup;  // Y
    int32_t coreGroupNum;     // F

    int32_t qSeqLength;     // qkv unequal length
    int32_t kSeqLength;     // qkv unequal length
    int32_t vSeqLength;     // qkv unequal length
    int32_t maskSeqLength;  // reserved
    float scale;            // reserved
    float keep_prob;        // reserved
    int32_t pre_tokens;     // reserved
    int32_t next_tokens;    // reserved

    bool isTriangle;        // lower triangular
    int32_t attenType;      // 0: MHA, 1: GQA
    int32_t sparseMode;     // 0: dense, 1: sparse
    int32_t headGroupSize;  // N / G
    int32_t windowLen;      // sparse sliding window
    bool isHighPrecision;   // high precision path

    int32_t get_batchSize() const
    {
        return batchSize;
    }
    void set_batchSize(int32_t value)
    {
        batchSize = value;
    }
    int32_t get_headNum() const
    {
        return headNum;
    }
    void set_headNum(int32_t value)
    {
        headNum = value;
    }
    int32_t get_seqSize() const
    {
        return seqSize;
    }
    void set_seqSize(int32_t value)
    {
        seqSize = value;
    }
    int32_t get_headDim() const
    {
        return headDim;
    }
    void set_headDim(int32_t value)
    {
        headDim = value;
    }
    int32_t get_coreNumPerGroup() const
    {
        return coreNumPerGroup;
    }
    void set_coreNumPerGroup(int32_t value)
    {
        coreNumPerGroup = value;
    }
    int32_t get_coreGroupNum() const
    {
        return coreGroupNum;
    }
    void set_coreGroupNum(int32_t value)
    {
        coreGroupNum = value;
    }
    int32_t get_qSeqLength() const
    {
        return qSeqLength;
    }
    void set_qSeqLength(int32_t value)
    {
        qSeqLength = value;
    }
    int32_t get_kSeqLength() const
    {
        return kSeqLength;
    }
    void set_kSeqLength(int32_t value)
    {
        kSeqLength = value;
    }
    int32_t get_vSeqLength() const
    {
        return vSeqLength;
    }
    void set_vSeqLength(int32_t value)
    {
        vSeqLength = value;
    }
    int32_t get_maskSeqLength() const
    {
        return maskSeqLength;
    }
    void set_maskSeqLength(int32_t value)
    {
        maskSeqLength = value;
    }
    float get_scale() const
    {
        return scale;
    }
    void set_scale(float value)
    {
        scale = value;
    }
    float get_keep_prob() const
    {
        return keep_prob;
    }
    void set_keep_prob(float value)
    {
        keep_prob = value;
    }
    int32_t get_pre_tokens() const
    {
        return pre_tokens;
    }
    void set_pre_tokens(int32_t value)
    {
        pre_tokens = value;
    }
    int32_t get_next_tokens() const
    {
        return next_tokens;
    }
    void set_next_tokens(int32_t value)
    {
        next_tokens = value;
    }
    bool get_isTriangle() const
    {
        return isTriangle;
    }
    void set_isTriangle(bool value)
    {
        isTriangle = value;
    }
    int32_t get_attenType() const
    {
        return attenType;
    }
    void set_attenType(int32_t value)
    {
        attenType = value;
    }
    int32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(int32_t value)
    {
        sparseMode = value;
    }
    int32_t get_headGroupSize() const
    {
        return headGroupSize;
    }
    void set_headGroupSize(int32_t value)
    {
        headGroupSize = value;
    }
    int32_t get_windowLen() const
    {
        return windowLen;
    }
    void set_windowLen(int32_t value)
    {
        windowLen = value;
    }
    bool get_isHighPrecision() const
    {
        return isHighPrecision;
    }
    void set_isHighPrecision(bool value)
    {
        isHighPrecision = value;
    }
};

}  // namespace optiling

#endif  // LASER_ATTENTION_TILING_DATA_H
