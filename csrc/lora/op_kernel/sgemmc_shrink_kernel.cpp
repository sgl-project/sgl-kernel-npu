/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/sgmv_shrink.cpp
 */

#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMC_SHRINK_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMC_SHRINK_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lora_common_kernel.h"
#include "common_tiling_kernel.h"

#include "../op_host/tiling/sgemmc_tiling_data.h"

template <typename scalar_t, typename inner_t>
class SGEMMCShrink
{
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using INNER_T = inner_t;
    using Y_T = scalar_t;

    using X_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::VECTOR, X_T, false>;
    using W_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, W_T, true>;
    using Y_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, INNER_T>;
    using BIAS_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T>;

    using MAT_TYPE = AscendC::Matmul<X_MAT_TYPE, W_MAT_TYPE, Y_MAT_TYPE, BIAS_MAT_TYPE, CFG_MDL>;

public:
    __aicore__ explicit SGEMMCShrink(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR loraScales, uint32_t loraScalesSize, GM_ADDR y, uint32_t batchSize,
                                uint32_t numBlocksPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank,
                                GM_ADDR workspace, TCubeTiling &tiling)
    {
        batchSize_ = batchSize;
        numBlocksPerCore_ = numBlocksPerCore;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        singleLoRAWeightLen_ = inputHiddenDim_ * maxLoRARank_;

        xInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        yOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(y));
        wInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        loraIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraIndices), loraIndicesSize);
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen), seqLenSize);
        loraRanksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraRanks), loraRanksSize);
        loraScalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(loraScales), loraScalesSize);

        REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), matmulObj, &tiling);
    }

    __aicore__ inline void Process()
    {
        int64_t blocks = AscendC::GetBlockNum();
        int64_t blockIdx = AscendC::GetBlockIdx();

        int64_t startIdx = blockIdx * numBlocksPerCore_;
        int64_t endIdx = startIdx + numBlocksPerCore_;

        AscendC::WaitPreTaskEnd();

        int64_t batchIdx = 0;
        int64_t requestBlock = 0;
        lora_common::BlockIterator blockIterator(seqLenGm_);
        requestBlock = blockIterator.GetBlockIdx(batchIdx);
        if (requestBlock < 0) {
            return;
        }

        int32_t reqLoRAIndex_ = loraIndicesGm_.GetValue(requestBlock);
        if (reqLoRAIndex_ < 0) {
            return;
        }

        int64_t reqLoRAWeightOffset_ = reqLoRAIndex_ * singleLoRAWeightLen_;
        int32_t reqLoRARank_ = loraRanksGm_.GetValue(reqLoRAIndex_);

        if (reqLoRARank_ == 0) {
            return;
        }

        matmulObj.SetTensorA(xInGm_);
        matmulObj.SetTensorB(wInGm_);
        matmulObj.template Iterate<false>();

        half loraScale = loraScalesGm_.GetValue(reqLoRAIndex_);
        INNER_T scalar = AscendC::ScalarCast<half, INNER_T, AscendC::RoundMode::CAST_ROUND>(loraScale);

        uint32_t baseM = this->tiling.baseM;
        uint32_t baseN = this->tiling.baseN;
        pipe_->InitBuffer(vectorCalcBuf, baseM * baseN * sizeof(INNER_T));
        pipe_->InitBuffer(vectorInQueue, 1, baseM * baseN * sizeof(INNER_T));
        pipe_->InitBuffer(vectorOutQueue, 1, baseM * baseN * sizeof(Y_T));

        AscendC::DataCopyParams copyParams = {
            (uint16_t)baseM, (uint16_t)(baseN * sizeof(Y_T) / AscendC::DEFAULT_C0_SIZE), (uint16_t)0,
            (uint16_t)((this->tiling.N - baseN) * sizeof(Y_T) / AscendC::DEFAULT_C0_SIZE)};
        uint32_t iterateTimes =
            AscendC::Ceil(this->tiling.singleCoreM, baseM) * AscendC::Ceil(this->tiling.singleCoreN, baseN);
        for (uint32_t i = 0; i < iterateTimes; ++i) {
            // compute
            auto cInLocal = vectorInQueue.AllocTensor<INNER_T>();
            matmulObj.template GetTensorC<false>(cInLocal);
            vectorInQueue.EnQue(cInLocal);
            // any vector operator
            auto src = vectorInQueue.DeQue<INNER_T>();
            auto dst = vectorOutQueue.AllocTensor<Y_T>();

            AscendC::LocalTensor<INNER_T> tmpTensor = vectorCalcBuf.Get<INNER_T>();
            AscendC::Muls(tmpTensor, src, scalar, baseM * baseN);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(dst, tmpTensor, AscendC::RoundMode::CAST_NONE, baseM * baseN);
            AscendC::PipeBarrier<PIPE_V>();
            vectorOutQueue.EnQue(dst);
            vectorInQueue.FreeTensor(src);
            // copy out
            auto cOutLocal = vectorOutQueue.DeQue<Y_T>();
            DataCopy(yOutGm_[i], cOutLocal, copyParams);
            vectorOutQueue.FreeTensor(cOutLocal);
        }
        matmulObj.End();
        AscendC::SetNextTaskStart();
    }

private:
    AscendC::TPipe *pipe_;
    MAT_TYPE matmulObj;

    AscendC::GlobalTensor<X_T> xInGm_;
    AscendC::GlobalTensor<W_T> wInGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    AscendC::GlobalTensor<int32_t> loraIndicesGm_;
    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> loraRanksGm_;
    AscendC::GlobalTensor<half> loraScalesGm_;

    AscendC::GlobalTensor<Y_T> workspaceGlobal;
    AscendC::LocalTensor<Y_T> vectorInLocal;

    TCubeTiling tiling;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> vectorInQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vectorOutQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> vectorCalcBuf;

    uint32_t batchSize_;
    uint32_t numBlocksPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    uint32_t singleLoRAWeightLen_;

    uint64_t reqLoRAWeightOffset_;
    int32_t reqLoRAIndex_;
    int32_t reqLoRARank_;
};

extern "C" __global__ __aicore__ void sgemmc_shrink(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices,
                                                    uint32_t loraIndicesSize, GM_ADDR seqLen, uint32_t seqLenSize,
                                                    GM_ADDR loraRanks, uint32_t loraRanksSize, GM_ADDR loraScales,
                                                    uint32_t loraScalesSize, GM_ADDR y, uint32_t batchSize,
                                                    uint32_t numBlocksPerCore, uint32_t inputHiddenDim,
                                                    uint32_t maxLoRARank, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    AscendC::TPipe pipe;
    sglang::npu_kernel::SGEMMCTilingData tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);

    if (tilingData.dataType == 1) {
        SGEMMCShrink<bfloat16_t, float> op(&pipe);
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, loraScales,
                loraScalesSize, y, batchSize, numBlocksPerCore, inputHiddenDim, maxLoRARank, workspace,
                tilingData.cubeTiling);
        op.Process();
    } else {
        SGEMMCShrink<half, float> op(&pipe);
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, loraScales,
                loraScalesSize, y, batchSize, numBlocksPerCore, inputHiddenDim, maxLoRARank, workspace,
                tilingData.cubeTiling);
        op.Process();
    }
}

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMC_SHRINK_H
