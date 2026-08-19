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
 * \file kv_compress_epilog.cpp
 * \brief KV compress epilog kernel entry (A5-only).
 */

#include "kernel_operator.h"

// ascendc_library compiles kernel sources once more for host-bisheng, where
// CANN 9.1 does not define __NPU_ARCH__. The generated host stub is compiled
// separately; this MicroAPI-based implementation is device-only.
#if defined(__NPU_ARCH__)

#include "kv_compress_epilog.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void kv_compress_epilog(GM_ADDR kv_compress_cache, GM_ADDR x, GM_ADDR slot_mapping,
                                                         GM_ADDR kv_compress_cache_out, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    __gm__ uint8_t *userspace = workspace;
    if (userspace == nullptr) {
        return;
    }

    TPipe pipe;

    // Local copy of the tiling data. A struct-by-value copy from a __gm__ pointer is rejected
    // on A5 ("argument is in address space gm, but parameter must be in Local Memory") and the
    // GET_TILING_DATA_WITH_STRUCT macro cannot take a namespaced type name, so copy field by
    // field through a __gm__ pointer (scalar GM reads, as causal_conv1d does on A5). The
    // packed struct layout leaves every scalar naturally aligned, so no unaligned loads.
    const __gm__ sglang::KvCompressEpilogTilingData *__restrict__ tilingGm =
        reinterpret_cast<const __gm__ sglang::KvCompressEpilogTilingData *>(tiling);
    sglang::KvCompressEpilogTilingData tilingDataIn;
    tilingDataIn.bs = tilingGm->bs;
    tilingDataIn.d = tilingGm->d;
    tilingDataIn.kvCacheCol = tilingGm->kvCacheCol;
    tilingDataIn.scaleCol = tilingGm->scaleCol;
    tilingDataIn.concatCol = tilingGm->concatCol;
    tilingDataIn.padCol = tilingGm->padCol;
    tilingDataIn.quantMode = tilingGm->quantMode;
    tilingDataIn.roundScale = tilingGm->roundScale;
    tilingDataIn.perGroupSize = tilingGm->perGroupSize;
    tilingDataIn.rowOfFormerBlock = tilingGm->rowOfFormerBlock;
    tilingDataIn.rowOfTailBlock = tilingGm->rowOfTailBlock;
    tilingDataIn.rowLoopOfFormerBlock = tilingGm->rowLoopOfFormerBlock;
    tilingDataIn.rowLoopOfTailBlock = tilingGm->rowLoopOfTailBlock;
    tilingDataIn.rowFactor = tilingGm->rowFactor;
    tilingDataIn.tailRowFactorOfFormerBlock = tilingGm->tailRowFactorOfFormerBlock;
    tilingDataIn.tailRowFactorOfTailBlock = tilingGm->tailRowFactorOfTailBlock;
    tilingDataIn.layout = tilingGm->layout;
    tilingDataIn.blockSize = tilingGm->blockSize;
    tilingDataIn.valuePerToken = tilingGm->valuePerToken;
    tilingDataIn.scalePerToken = tilingGm->scalePerToken;
    tilingDataIn.blockStride = tilingGm->blockStride;
    tilingDataIn.dtype = tilingGm->dtype;
    tilingDataIn.tilingKey = tilingGm->tilingKey;
    const sglang::KvCompressEpilogTilingData *__restrict__ tilingData = &tilingDataIn;

    // Save and set overflow mode to saturation (0) for FP8 quantization
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);

    // Dispatch based on tiling key
    const uint32_t tilingKey = tilingData->tilingKey;
    const int32_t dtype = tilingData->dtype;
    if (tilingKey == 0) {
        if (dtype == 0) {  // slot_mapping int32, kv_compress_cache fp8_e5m2
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache, tilingData);
            op.Process();
        } else if (dtype == 1) {  // slot_mapping int64, kv_compress_cache fp8_e5m2
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int64_t, fp8_e5m2_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache, tilingData);
            op.Process();
        } else if (dtype == 2) {  // slot_mapping int32, kv_compress_cache fp8_e4m3fn
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache, tilingData);
            op.Process();
        } else {  // slot_mapping int64, kv_compress_cache fp8_e4m3fn
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int64_t, fp8_e4m3fn_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache, tilingData);
            op.Process();
        }
    }

    // Restore overflow mode
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#endif
