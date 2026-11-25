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

#include "kv_compress_epilog.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sgl_kv_compress_epilog(GM_ADDR kv_compress_cache, GM_ADDR x, GM_ADDR slot_mapping,
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

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::SglKvCompressEpilogTilingData);
    GET_TILING_DATA(tilingDataIn, tiling);
    const optiling::SglKvCompressEpilogTilingData *__restrict__ tilingData = &tilingDataIn;

    // Save and set overflow mode to saturation (0) for FP8 quantization
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);

    // Dispatch based on tiling key
    const uint32_t tilingKey = tilingData->tilingKey;
    const int32_t dtype = tilingData->dtype;
    if (tilingKey == 0) {
        if (dtype == 0) {  // slot_mapping int32, kv_compress_cache fp8_e5m2
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int32_t, fp8_e5m2_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache_out, tilingData);
            op.Process();
        } else if (dtype == 1) {  // slot_mapping int64, kv_compress_cache fp8_e5m2
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int64_t, fp8_e5m2_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache_out, tilingData);
            op.Process();
        } else if (dtype == 2) {  // slot_mapping int32, kv_compress_cache fp8_e4m3fn
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int32_t, fp8_e4m3fn_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache_out, tilingData);
            op.Process();
        } else {  // slot_mapping int64, kv_compress_cache fp8_e4m3fn
            KvCompressEpilogOps::KvCompressEpilogRegBase<bfloat16_t, int64_t, fp8_e4m3fn_t> op(&pipe);
            op.Init(x, slot_mapping, kv_compress_cache_out, tilingData);
            op.Process();
        }
    }

    // Restore overflow mode
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
