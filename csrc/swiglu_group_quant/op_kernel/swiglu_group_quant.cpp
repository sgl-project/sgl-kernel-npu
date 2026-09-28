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
 * \file swiglu_group_quant.cpp
 * \brief swiglu_group_quant kernel entry (A5-only).
 *
 * The three op classes and their shared helpers (swiglu_group_quant_base.h, *_perf.h,
 * swiglu_fp8_quant_per_token.h) live alongside this file. This entry does nothing but dispatch:
 *
 *   - Tiling arrives as a packed struct memcpy'd to GM and is copied field-by-field below.
 *     GET_TILING_DATA expands to a struct-by-value copy out of a __gm__ pointer, which A5 rejects
 *     ("argument is in address space gm, but parameter must be in Local Memory"), and
 *     GET_TILING_DATA_WITH_STRUCT cannot take a namespaced type name.
 *   - The tiling key is a plain field in that struct, compared directly.
 *   - x/y/scale dtypes are carried in the struct's `dtype` field and dispatched at runtime, since
 *     this repo builds one kernel rather than one binary per dtype combination.
 *
 * The dispatch covers every (tiling key, dtype) combination the host can produce. scale dtype is
 * implied by the quant mode for keys 1 and 2 (fp32 and e8m0 respectively), and is selected by
 * `ue8m0_scale` for the fp8 keys.
 */

#include "kernel_operator.h"

// ascendc_library compiles kernel sources once more for host-bisheng, where CANN 9.1 does not
// define __NPU_ARCH__. This MicroAPI-based implementation is device-only.
#if defined(__NPU_ARCH__)

#include "../op_host/tiling/swiglu_group_quant_tiling_data.h"

using sglang::SwigluGroupQuantTilingData;

#include "swiglu_group_quant_perf.h"
#include "swiglu_mx_quant_perf.h"
#include "swiglu_fp8_quant_per_token.h"

using namespace AscendC;

namespace {

// Shorthands so the dispatch below stays readable. Each expands to the "construct / Init / Process"
// sequence with the class's template parameters filled in.
#define SWIGLU_GROUP_QUANT_RUN_GROUP(XT, YT)                                        \
    do {                                                                            \
        SwigluGroupQuant::SwigluGroupQuantPerf<XT, YT, float> op;                   \
        op.Init(x, topkWeight, groupIndex, y, scale, userWs, &tilingDataIn, &pipe); \
        op.Process();                                                               \
    } while (0)

#define SWIGLU_GROUP_QUANT_RUN_MX(XT, YT)                                           \
    do {                                                                            \
        SwigluGroupQuant::SwigluMxQuantPerf<XT, YT, fp8_e8m0_t> op;                 \
        op.Init(x, topkWeight, groupIndex, y, scale, userWs, &tilingDataIn, &pipe); \
        op.Process();                                                               \
    } while (0)

#define SWIGLU_GROUP_QUANT_RUN_FP8(XT, YT, ST, OO)                                           \
    do {                                                                                     \
        SwigluGroupQuant::SwigluFp8QuantPerToken<XT, YT, ST, OO> op;                         \
        op.Init(x, topkWeight, groupIndex, y, scale, yOrigin, userWs, &tilingDataIn, &pipe); \
        op.Process();                                                                        \
    } while (0)

}  // namespace

extern "C" __global__ __aicore__ void swiglu_group_quant(GM_ADDR x, GM_ADDR topkWeight, GM_ADDR groupIndex, GM_ADDR y,
                                                         GM_ADDR scale, GM_ADDR yOrigin, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    // The op classes take a workspace argument but never read it, so the raw pointer is forwarded
    // as-is rather than through GetUserWorkspace (which kv_compress_epilog also avoids on A5).
    GM_ADDR userWs = workspace;
    if (userWs == nullptr) {
        return;
    }

    TPipe pipe;

    // Local copy of the tiling data; see the file comment for why this is field-by-field.
    const __gm__ SwigluGroupQuantTilingData *__restrict__ tilingGm =
        reinterpret_cast<const __gm__ SwigluGroupQuantTilingData *>(tiling);
    SwigluGroupQuantTilingData tilingDataIn;
    tilingDataIn.bs = tilingGm->bs;
    tilingDataIn.d = tilingGm->d;
    tilingDataIn.splitD = tilingGm->splitD;
    tilingDataIn.scaleCol = tilingGm->scaleCol;
    tilingDataIn.rowOfFormerBlock = tilingGm->rowOfFormerBlock;
    tilingDataIn.rowOfTailBlock = tilingGm->rowOfTailBlock;
    tilingDataIn.rowLoopOfFormerBlock = tilingGm->rowLoopOfFormerBlock;
    tilingDataIn.rowLoopOfTailBlock = tilingGm->rowLoopOfTailBlock;
    tilingDataIn.rowFactor = tilingGm->rowFactor;
    tilingDataIn.tailRowFactorOfFormerBlock = tilingGm->tailRowFactorOfFormerBlock;
    tilingDataIn.tailRowFactorOfTailBlock = tilingGm->tailRowFactorOfTailBlock;
    tilingDataIn.dLoop = tilingGm->dLoop;
    tilingDataIn.dFactor = tilingGm->dFactor;
    tilingDataIn.tailDFactor = tilingGm->tailDFactor;
    tilingDataIn.roundScale = tilingGm->roundScale;
    tilingDataIn.ue8m0Scale = tilingGm->ue8m0Scale;
    tilingDataIn.outputOrigin = tilingGm->outputOrigin;
    tilingDataIn.hasClampValue = tilingGm->hasClampValue;
    tilingDataIn.g = tilingGm->g;
    tilingDataIn.ubSize = tilingGm->ubSize;
    tilingDataIn.gLoop = tilingGm->gLoop;
    tilingDataIn.gFactor = tilingGm->gFactor;
    tilingDataIn.tailGFactor = tilingGm->tailGFactor;
    tilingDataIn.groupListType = tilingGm->groupListType;
    tilingDataIn.coreNum = tilingGm->coreNum;
    tilingDataIn.clampValue = tilingGm->clampValue;
    const int32_t dtype = tilingGm->dtype;
    const uint32_t tilingKey = tilingGm->tilingKey;

    const bool xBf16 = (dtype & sglang::SWIGLU_GROUP_QUANT_DTYPE_X_BF16) != 0;
    const bool yE5m2 = (dtype & sglang::SWIGLU_GROUP_QUANT_DTYPE_Y_E5M2) != 0;
    const bool scaleE8m0 = (dtype & sglang::SWIGLU_GROUP_QUANT_DTYPE_SCALE_E8M0) != 0;

    // The overflow mode is saved and restored around the whole dispatch; each class's Init
    // additionally forces it to saturation (0), which is what the fp8 casts need.
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();

    if (tilingKey == sglang::SWIGLU_GROUP_QUANT_TILING_KEY_GROUP_QUANT) {
        // Group (per-128-block) quant: scale is always fp32.
        if (xBf16) {
            if (yE5m2) {
                SWIGLU_GROUP_QUANT_RUN_GROUP(bfloat16_t, fp8_e5m2_t);
            } else {
                SWIGLU_GROUP_QUANT_RUN_GROUP(bfloat16_t, fp8_e4m3fn_t);
            }
        } else {
            if (yE5m2) {
                SWIGLU_GROUP_QUANT_RUN_GROUP(half, fp8_e5m2_t);
            } else {
                SWIGLU_GROUP_QUANT_RUN_GROUP(half, fp8_e4m3fn_t);
            }
        }
    } else if (tilingKey == sglang::SWIGLU_GROUP_QUANT_TILING_KEY_MX_QUANT) {
        // MX quant (per-32-group): scale is always e8m0.
        if (xBf16) {
            if (yE5m2) {
                SWIGLU_GROUP_QUANT_RUN_MX(bfloat16_t, fp8_e5m2_t);
            } else {
                SWIGLU_GROUP_QUANT_RUN_MX(bfloat16_t, fp8_e4m3fn_t);
            }
        } else {
            if (yE5m2) {
                SWIGLU_GROUP_QUANT_RUN_MX(half, fp8_e5m2_t);
            } else {
                SWIGLU_GROUP_QUANT_RUN_MX(half, fp8_e4m3fn_t);
            }
        }
    } else if (tilingKey == sglang::SWIGLU_GROUP_QUANT_TILING_KEY_FP8_QUANT) {
        // Per-token fp8 quant, without the swiglu result in the source dtype: scale is fp32 unless
        // the caller asked for ue8m0.
        if (xBf16) {
            if (yE5m2) {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e5m2_t, fp8_e8m0_t, false);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e5m2_t, float, false);
                }
            } else {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e4m3fn_t, fp8_e8m0_t, false);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e4m3fn_t, float, false);
                }
            }
        } else {
            if (yE5m2) {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e5m2_t, fp8_e8m0_t, false);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e5m2_t, float, false);
                }
            } else {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e4m3fn_t, fp8_e8m0_t, false);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e4m3fn_t, float, false);
                }
            }
        }
    } else if (tilingKey == sglang::SWIGLU_GROUP_QUANT_TILING_KEY_FP8_QUANT_YORIGIN) {
        // Same as above, plus the swiglu result written back in the source dtype.
        if (xBf16) {
            if (yE5m2) {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e5m2_t, fp8_e8m0_t, true);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e5m2_t, float, true);
                }
            } else {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e4m3fn_t, fp8_e8m0_t, true);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(bfloat16_t, fp8_e4m3fn_t, float, true);
                }
            }
        } else {
            if (yE5m2) {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e5m2_t, fp8_e8m0_t, true);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e5m2_t, float, true);
                }
            } else {
                if (scaleE8m0) {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e4m3fn_t, fp8_e8m0_t, true);
                } else {
                    SWIGLU_GROUP_QUANT_RUN_FP8(half, fp8_e4m3fn_t, float, true);
                }
            }
        }
    }

    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#endif
