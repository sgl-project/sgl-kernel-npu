/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_VF_H

#if !defined(__ASC_NPU_HOST__)

#include "kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace MicroAPI;

/*!
 * InverseAIVVF: forward-substitution inverse of the N x N lower-triangular (unit-diagonal) attn block.
 *   inv[0] = e_0; for i=1..N-1: inv[i] = e_i - sum_{j<i} attn[i,j] * inv[j]
 * attnUb/invResUb layout: [halfChunkSize rows, chunkSize cols]; subBlock uses column offset = `offset`,
 *   i.e. element (i, col) is at base + offset + i*chunkSize + col. Only first N columns per row are valid.
 * eiUb layout: identity matrix, row i = e_i, row stride = chunkSize (no column offset).
 *   eiUb must be pre-filled by the caller (unit-diagonal identity in the first N rows).
 */
template <uint32_t N>
__simd_vf__ inline void InverseAIVVFImpl(__ubuf__ float *attnUb, __ubuf__ float *invResUb, __ubuf__ float *eiUb,
                                         uint32_t offset, uint32_t chunkSize)
{
    uint32_t maskLen = N;
    MaskReg maskN = UpdateMask<float>(maskLen);

    RegTensor<float> inv0;
    LoadAlign(inv0, eiUb);
    StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset, inv0, maskN);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    RegTensor<uint32_t> idxReg;
    RegTensor<float> acc;
    for (uint16_t i = 1; i < static_cast<uint16_t>(N); ++i) {
        Duplicate(acc, static_cast<float>(0.0));
        RegTensor<float> li;
        LoadAlign(li, attnUb + offset + i * chunkSize);
        for (uint16_t j = 0; j < i; ++j) {
            Duplicate(idxReg, j);
            RegTensor<float> lijBrc;
            Gather(lijBrc, li, idxReg);
            RegTensor<float> invj;
            LoadAlign(invj, invResUb + offset + j * chunkSize);
            MulAddDst(acc, invj, lijBrc, maskN);
        }
        RegTensor<float> ei_i;
        LoadAlign(ei_i, eiUb + i * chunkSize);
        RegTensor<float> invi;
        Sub(invi, ei_i, acc, maskN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(invResUb + offset + i * chunkSize, invi, maskN);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <uint32_t N>
__aicore__ inline void InverseAIVVF(const LocalTensor<float> &attnUb, const LocalTensor<float> &invResUb,
                                    const LocalTensor<float> &eiUb, uint32_t offset, uint32_t chunkSize)
{
    __ubuf__ float *attn = reinterpret_cast<__ubuf__ float *>(attnUb.GetPhyAddr());
    __ubuf__ float *inv = reinterpret_cast<__ubuf__ float *>(invResUb.GetPhyAddr());
    __ubuf__ float *ei = reinterpret_cast<__ubuf__ float *>(eiUb.GetPhyAddr());
    InverseAIVVFImpl<N>(attn, inv, ei, offset, chunkSize);
}
}  // namespace ChunkGatedDeltaRule
#endif  // !__ASC_NPU_HOST__

#endif  // CHUNK_GATED_DELTA_RULE_STAGE1_VF_H
