// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// Migrated from csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp. The
// computation body is unchanged; only the entry point, the tiling hand-off and
// the user workspace base follow the independent CANN operator framework.

#include "kernel_operator.h"

#include "mla_preprocess_mix_fp16.hpp"
#include "mla_preprocess_mix_bf16.hpp"

#include "mla_preprocess_tiling_data.h"

extern "C" __global__ __aicore__ void sgl_mla_preprocess(
    GM_ADDR hiddenState, GM_ADDR gamma0, GM_ADDR beta0, GM_ADDR wdqkv, GM_ADDR descale0, GM_ADDR gamma1, GM_ADDR beta1,
    GM_ADDR wuq, GM_ADDR descale1, GM_ADDR gamma2, GM_ADDR cos, GM_ADDR sin, GM_ADDR wuk, GM_ADDR kvCache,
    GM_ADDR kvCacheRope, GM_ADDR slotMapping, GM_ADDR quantScale0, GM_ADDR quantOffset0, GM_ADDR bias0,
    GM_ADDR quantScale1, GM_ADDR quantOffset1, GM_ADDR bias1, GM_ADDR ctkvScale, GM_ADDR qnopeScale, GM_ADDR qOut0,
    GM_ADDR kvCacheOut0, GM_ADDR qOut1, GM_ADDR kvCacheOut1, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    PRELOAD(2);

    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_CUBE__
    SetPadding<uint64_t>((uint64_t)0);
    SetNdpara(1, 0, 0);
#endif

    // Preserve the legacy field-by-field hand-off: several tiling fields
    // (notably mm1/mm2 swizzleDirect) were intentionally left at their struct
    // defaults, so the computation must see the same values.
    REGISTER_TILING_DEFAULT(MlaTilingData);
    GET_TILING_DATA(tilingDataFull, tilingGM);
    MlaTilingData mlaTilingData;

    mlaTilingData.tilingKey = tilingDataFull.tilingKey;
    mlaTilingData.n = tilingDataFull.n;
    mlaTilingData.hiddenStateDim = tilingDataFull.hiddenStateDim;

    mlaTilingData.mm1.numBatch = tilingDataFull.mm1.numBatch;
    mlaTilingData.mm1.m = tilingDataFull.mm1.m;
    mlaTilingData.mm1.k = tilingDataFull.mm1.k;
    mlaTilingData.mm1.n = tilingDataFull.mm1.n;
    mlaTilingData.mm1.m0 = tilingDataFull.mm1.m0;
    mlaTilingData.mm1.k0 = tilingDataFull.mm1.k0;
    mlaTilingData.mm1.n0 = tilingDataFull.mm1.n0;
    mlaTilingData.mm1.mLoop = tilingDataFull.mm1.mLoop;
    mlaTilingData.mm1.kLoop = tilingDataFull.mm1.kLoop;
    mlaTilingData.mm1.nLoop = tilingDataFull.mm1.nLoop;
    mlaTilingData.mm1.coreLoop = tilingDataFull.mm1.coreLoop;
    mlaTilingData.mm1.swizzleCount = tilingDataFull.mm1.swizzleCount;
    mlaTilingData.mm1.enShuffleK = tilingDataFull.mm1.enShuffleK;
    mlaTilingData.mm1.blockDim = tilingDataFull.mm1.blockDim;
    mlaTilingData.mm1.enLoadAllAmat = tilingDataFull.mm1.enLoadAllAmat;
    mlaTilingData.mm1.b0matPingPongBufferLen = tilingDataFull.mm1.b0matPingPongBufferLen;

    mlaTilingData.mm2.numBatch = tilingDataFull.mm2.numBatch;
    mlaTilingData.mm2.m = tilingDataFull.mm2.m;
    mlaTilingData.mm2.k = tilingDataFull.mm2.k;
    mlaTilingData.mm2.n = tilingDataFull.mm2.n;
    mlaTilingData.mm2.m0 = tilingDataFull.mm2.m0;
    mlaTilingData.mm2.k0 = tilingDataFull.mm2.k0;
    mlaTilingData.mm2.n0 = tilingDataFull.mm2.n0;
    mlaTilingData.mm2.mLoop = tilingDataFull.mm2.mLoop;
    mlaTilingData.mm2.kLoop = tilingDataFull.mm2.kLoop;
    mlaTilingData.mm2.nLoop = tilingDataFull.mm2.nLoop;
    mlaTilingData.mm2.coreLoop = tilingDataFull.mm2.coreLoop;
    mlaTilingData.mm2.swizzleCount = tilingDataFull.mm2.swizzleCount;
    mlaTilingData.mm2.enShuffleK = tilingDataFull.mm2.enShuffleK;
    mlaTilingData.mm2.blockDim = tilingDataFull.mm2.blockDim;
    mlaTilingData.mm2.enLoadAllAmat = tilingDataFull.mm2.enLoadAllAmat;
    mlaTilingData.mm2.b0matPingPongBufferLen = tilingDataFull.mm2.b0matPingPongBufferLen;

    mlaTilingData.mm3.numBatch = tilingDataFull.mm3.numBatch;
    mlaTilingData.mm3.m = tilingDataFull.mm3.m;
    mlaTilingData.mm3.k = tilingDataFull.mm3.k;
    mlaTilingData.mm3.n = tilingDataFull.mm3.n;
    mlaTilingData.mm3.m0 = tilingDataFull.mm3.m0;
    mlaTilingData.mm3.k0 = tilingDataFull.mm3.k0;
    mlaTilingData.mm3.n0 = tilingDataFull.mm3.n0;
    mlaTilingData.mm3.mLoop = tilingDataFull.mm3.mLoop;
    mlaTilingData.mm3.kLoop = tilingDataFull.mm3.kLoop;
    mlaTilingData.mm3.nLoop = tilingDataFull.mm3.nLoop;
    mlaTilingData.mm3.coreLoop = tilingDataFull.mm3.coreLoop;
    mlaTilingData.mm3.swizzleCount = tilingDataFull.mm3.swizzleCount;
    mlaTilingData.mm3.enShuffleK = tilingDataFull.mm3.enShuffleK;
    mlaTilingData.mm3.blockDim = tilingDataFull.mm3.blockDim;

    mlaTilingData.perTaskNum = tilingDataFull.perTaskNum;
    mlaTilingData.resTaskNum = tilingDataFull.resTaskNum;
    mlaTilingData.numCore = tilingDataFull.numCore;

    mlaTilingData.rmsNumCore1 = tilingDataFull.rmsNumCore1;
    mlaTilingData.rmsNumCol1 = tilingDataFull.rmsNumCol1;
    mlaTilingData.rmsNumCore2 = tilingDataFull.rmsNumCore2;
    mlaTilingData.rmsNumCol2 = tilingDataFull.rmsNumCol2;

    mlaTilingData.hiddenSizeQ = tilingDataFull.hiddenSizeQ;
    mlaTilingData.headNumQ = tilingDataFull.headNumQ;
    mlaTilingData.headDim = tilingDataFull.headDim;
    mlaTilingData.concatSize = tilingDataFull.concatSize;
    mlaTilingData.rotaryCoeff = tilingDataFull.rotaryCoeff;
    mlaTilingData.ntokens = tilingDataFull.ntokens;
    mlaTilingData.realCore = tilingDataFull.realCore;
    mlaTilingData.nlCoreRun = tilingDataFull.nlCoreRun;
    mlaTilingData.lCoreRun = tilingDataFull.lCoreRun;
    mlaTilingData.maxNPerLoopForUb = tilingDataFull.maxNPerLoopForUb;
    mlaTilingData.preCoreLoopTime = tilingDataFull.preCoreLoopTime;
    mlaTilingData.preCoreLoopNLast = tilingDataFull.preCoreLoopNLast;
    mlaTilingData.lastCoreLoopTime = tilingDataFull.lastCoreLoopTime;
    mlaTilingData.lastCoreLoopNLast = tilingDataFull.lastCoreLoopNLast;

    mlaTilingData.esqFrontCore = tilingDataFull.esqFrontCore;
    mlaTilingData.esqTailCore = tilingDataFull.esqTailCore;
    mlaTilingData.esqFrontCoreBatch = tilingDataFull.esqFrontCoreBatch;
    mlaTilingData.esqTailCoreBatch = tilingDataFull.esqTailCoreBatch;
    mlaTilingData.esqHeadNum = tilingDataFull.esqHeadNum;
    mlaTilingData.esqColNum = tilingDataFull.esqColNum;
    mlaTilingData.esqUbHeadLoop = tilingDataFull.esqUbHeadLoop;
    mlaTilingData.esqHeadPerLoop = tilingDataFull.esqHeadPerLoop;
    mlaTilingData.esqHeadTail = tilingDataFull.esqHeadTail;
    mlaTilingData.esqColLoop = tilingDataFull.esqColLoop;
    mlaTilingData.esqColTail = tilingDataFull.esqColTail;

    mlaTilingData.s1Offset = tilingDataFull.s1Offset;
    mlaTilingData.s2Offset = tilingDataFull.s2Offset;
    mlaTilingData.s3Offset = tilingDataFull.s3Offset;
    mlaTilingData.s4Offset = tilingDataFull.s4Offset;
    mlaTilingData.s5Offset = tilingDataFull.s5Offset;

    // The framework reserves a system workspace before the user region.
    __gm__ uint8_t *userWorkspace = AscendC::GetUserWorkspace(workspaceGM);

    GM_ADDR s1 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s1Offset);
    GM_ADDR s2 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s2Offset);
    GM_ADDR s3 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s3Offset);
    GM_ADDR s4 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s4Offset);
    GM_ADDR s5 = userWorkspace + static_cast<uint64_t>(mlaTilingData.s5Offset);

    switch (mlaTilingData.tilingKey) {
        case KEY_FP16_CACHEMODE_0_QUANTMODE_0: {
            MLAPO_FP16::MLAOperation<CACHE_MODE_KVCACHE, DataFormat::NZ, DataFormat::NZ, DataFormat::ND> opFp16Cm0Qm0(
                mlaTilingData, tilingGM);
            opFp16Cm0Qm0.Init(hiddenState, gamma0, beta0, quantScale0, quantOffset0, wdqkv, bias0, gamma1, beta1,
                              quantScale1, quantOffset1, gamma2, sin, cos, sin, cos, kvCache, slotMapping, wuq, bias1,
                              wuk, descale0, descale1, ctkvScale, qnopeScale, qOut0, kvCacheOut0, qOut1, kvCacheOut1,
                              s1, s2, s3);
            if ASCEND_IS_AIC {
                opFp16Cm0Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opFp16Cm0Qm0.ProcessVector();
            }
            break;
        }
        case KEY_FP16_CACHEMODE_1_QUANTMODE_0: {
            MLAPO_FP16::MLAOperation<CACHE_MODE_KROPE_CTKV, DataFormat::NZ, DataFormat::NZ, DataFormat::ND>
                opFp16Cm1Qm0(mlaTilingData, tilingGM);
            opFp16Cm1Qm0.Init(hiddenState, gamma0, beta0, quantScale0, quantOffset0, wdqkv, bias0, gamma1, beta1,
                              quantScale1, quantOffset1, gamma2, sin, cos, sin, cos, kvCache, slotMapping, wuq, bias1,
                              wuk, descale0, descale1, ctkvScale, qnopeScale, qOut0, kvCacheOut0, qOut1, kvCacheOut1,
                              s1, s2, s3);
            if ASCEND_IS_AIC {
                opFp16Cm1Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opFp16Cm1Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_0_QUANTMODE_0: {
            MLAPO_BF16::MLAOperation<__bf16, 0, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm0Qm0(mlaTilingData, tilingGM);
            opBf16Cm0Qm0.Init(hiddenState, gamma0, beta0, quantScale0, quantOffset0, wdqkv, bias0, gamma1, beta1,
                              quantScale1, quantOffset1, gamma2, sin, cos, sin, cos, kvCache, slotMapping, wuq, bias1,
                              wuk, descale0, descale1, ctkvScale, qnopeScale, qOut0, kvCacheOut0, qOut1, kvCacheOut1,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm0Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm0Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_1_QUANTMODE_0: {
            MLAPO_BF16::MLAOperation<__bf16, 1, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm1Qm0(mlaTilingData, tilingGM);
            opBf16Cm1Qm0.Init(hiddenState, gamma0, beta0, quantScale0, quantOffset0, wdqkv, bias0, gamma1, beta1,
                              quantScale1, quantOffset1, gamma2, sin, cos, sin, cos, kvCache, slotMapping, wuq, bias1,
                              wuk, descale0, descale1, ctkvScale, qnopeScale, qOut0, kvCacheOut0, qOut1, kvCacheOut1,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm1Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm1Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_2_QUANTMODE_0: {
            MLAPO_BF16::MLAOperation<__bf16, 2, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm2Qm0(mlaTilingData, tilingGM);
            opBf16Cm2Qm0.Init(hiddenState, gamma0, beta0, quantScale0, quantOffset0, wdqkv, bias0, gamma1, beta1,
                              quantScale1, quantOffset1, gamma2, sin, cos, sin, cos, kvCache, slotMapping, wuq, bias1,
                              wuk, descale0, descale1, ctkvScale, qnopeScale, qOut0, kvCacheOut0, qOut1, kvCacheOut1,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm2Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm2Qm0.ProcessVector();
            }
            break;
        }
        case KEY_BF16_CACHEMODE_3_QUANTMODE_0: {
            MLAPO_BF16::MLAOperation<__bf16, 3, DataFormat::NZ, DataFormat::NZ, DataFormat::ND,
                                     QuantMode::PER_TENSOR_ASYMM_QUANT>
                opBf16Cm3Qm0(mlaTilingData, tilingGM);
            opBf16Cm3Qm0.Init(hiddenState, gamma0, beta0, quantScale0, quantOffset0, wdqkv, bias0, gamma1, beta1,
                              quantScale1, quantOffset1, gamma2, sin, cos, sin, cos, kvCache, slotMapping, wuq, bias1,
                              wuk, descale0, descale1, ctkvScale, qnopeScale, qOut0, kvCacheOut0, qOut1, kvCacheOut1,
                              s1, s2, s3, s4, s5);
            if ASCEND_IS_AIC {
                opBf16Cm3Qm0.ProcessCube();
            }
            if ASCEND_IS_AIV {
                opBf16Cm3Qm0.ProcessVector();
            }
            break;
        }
        default: {
            break;
        }
    }
    return;
}
