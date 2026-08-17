/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
 */

#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMC_EXPAND_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMC_EXPAND_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lora_common_kernel.h"
#include "common_tiling_kernel.h"

#include "../op_host/tiling/sgemmc_tiling_data.h"

template <typename scalar_t, typename inner_t>
class SGEMMCExpand
{
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using INNER_T = inner_t;
    using Y_T = scalar_t;

    using X_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::VECTOR, X_T, false>;
    using W_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, W_T, true>;
    using Y_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, INNER_T>;
    using BIAS_MAT_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    using MAT_TYPE = AscendC::Matmul<X_MAT_TYPE, W_MAT_TYPE, Y_MAT_TYPE, BIAS_MAT_TYPE, CFG_MDL>;

public:
    __aicore__ explicit SGEMMCExpand(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR sliceOffsets, uint32_t sliceOffsetsSize, GM_ADDR yIn, GM_ADDR yOut,
                                uint32_t batchSize, uint32_t maxLoRARank, uint32_t outputFullDim, GM_ADDR workspace,
                                TCubeTiling &tiling)
    {
        this->tiling = tiling;

        batchSize_ = batchSize;
        maxLoRARank_ = maxLoRARank;
        sliceCount_ = sliceOffsetsSize - 1;
        outputFullDim_ = outputFullDim;
        singleLoRAWeightLen_ = maxLoRARank_ * outputFullDim_;

        xInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        wInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        yInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(yIn));
        yOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(yOut));
        loraIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraIndices), loraIndicesSize);
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen), seqLenSize);
        loraRanksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraRanks), loraRanksSize);
        sliceOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sliceOffsets), sliceOffsetsSize);

        // The workspace buffer starts with the lib-api (system) region used via
        // GetSysWorkSpacePtr(); user scratch must begin after it, otherwise
        // per-block matmul staging corrupts the matmul library's own state.
        workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ INNER_T *>(AscendC::GetUserWorkspace(workspace)));
    }

    __aicore__ inline void Process()
    {
        int64_t blocks = AscendC::GetBlockNum();
        int64_t blockIdx = AscendC::GetBlockIdx();

        if ASCEND_IS_AIV {
            if (AscendC::GetSubBlockIdx() == 1) {
                return;
            }
            blockIdx /= AscendC::GetSubBlockNum();
        }

        int64_t tokenIdx = blockIdx / sliceCount_;
        int64_t sliceIdx = blockIdx % sliceCount_;

        lora_common::BlockIterator blockIterator(seqLenGm_);
        int64_t requestBlock = blockIterator.GetBlockIdx(tokenIdx);
        if (requestBlock < 0) {
            return;
        }

        reqLoRAIndex_ = loraIndicesGm_.GetValue(requestBlock);
        if (reqLoRAIndex_ < 0) {
            return;
        }

        reqLoRAWeightOffset_ = reqLoRAIndex_ * singleLoRAWeightLen_;
        reqLoRARank_ = loraRanksGm_.GetValue(reqLoRAIndex_);

        if (reqLoRARank_ == 0) {
            return;
        }

        int32_t beginSlice = sliceOffsetsGm_.GetValue(sliceIdx);
        int32_t endSlice = sliceOffsetsGm_.GetValue(sliceIdx + 1);
        int32_t slice = endSlice - beginSlice;
        // Async GetTensorC stages C tiles in the workspace padded to the
        // *base* granularity: even with org M=1 (singleCoreM=1) the staged
        // region is baseM rows tall. Confirmed empirically: tiling prints
        // singleCoreM=1 / baseM=16, and striding by singleCoreM*N left every
        // block's staging 16x too small, so neighbouring blocks overwrote
        // each other's [256-col] C tiles nondeterministically. Stride by
        // baseM * tiling.N (full staged width) instead. Must match the
        // host-side allocation in sgemmc_tiling.cpp.
        workspaceGlobal = workspaceGlobal[blockIdx * tiling.baseM * tiling.N];

        REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), matmulObj, &tiling);

        matmulObj.DisableBias();
        matmulObj.SetWorkspace(workspaceGlobal);
        matmulObj.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
        // Org M is 1 by host contract, so singleCoreM == 1: each block
        // computes a single [1, slice] row of C. (The async C staging in the
        // workspace is still padded to baseM rows; see the stride above.)
        matmulObj.SetSingleShape(tiling.singleCoreM, slice, reqLoRARank_);
        // A is [tokens, slices*max_rank]; each slice block is max_rank wide,
        // so the slice offset must use maxLoRARank_ (not reqLoRARank_, which
        // undercounts when rank < max_rank with slices > 1).
        matmulObj.SetTensorA(xInGm_[tokenIdx * sliceCount_ * maxLoRARank_ + sliceIdx * maxLoRARank_], false);
        matmulObj.SetTensorB(wInGm_[reqLoRAWeightOffset_ + maxLoRARank_ * beginSlice], true);
        matmulObj.template Iterate<false>();

        uint32_t baseM = min(tiling.baseM, tiling.singleCoreM);
        uint32_t maxElements = tiling.baseM * tiling.baseN;

        pipe_->InitBuffer(calcBuf, maxElements * sizeof(INNER_T));
        pipe_->InitBuffer(matmulQueue, 1, maxElements * sizeof(INNER_T));
        pipe_->InitBuffer(vectorYInQueue, 1, maxElements * sizeof(Y_T));
        pipe_->InitBuffer(vectorOutQueue, 1, maxElements * sizeof(Y_T));

        // Walk the staged [singleCoreM, slice] C result in baseM*baseN tiles,
        // emitted M-then-N. baseM is clamped to singleCoreM (== 1), so every
        // tile is a single valid row; tail tiles (slice % baseN != 0) hold
        // slice - n0 valid columns. tileM != 0 is defensive (only reachable
        // if the tiling ever pads singleCoreM past org M): drain those tiles
        // without emitting.
        uint32_t nTilesPerRow = AscendC::Ceil(slice, tiling.baseN);
        uint32_t iterateTimes = AscendC::Ceil(tiling.singleCoreM, baseM) * nTilesPerRow;
        uint32_t outputRowOffset = tokenIdx * tiling.N + beginSlice;
        for (uint32_t i = 0; i < iterateTimes; ++i) {
            uint32_t tileM = i / nTilesPerRow;
            uint32_t tileN = i % nTilesPerRow;
            uint32_t n0 = tileN * tiling.baseN;
            uint32_t curN = min(static_cast<uint32_t>(tiling.baseN), static_cast<uint32_t>(slice) - n0);
            uint32_t elements = curN;  // only the first (valid) row
            uint32_t offset = outputRowOffset + n0;
            AscendC::DataCopyParams copyParams = {
                (uint16_t)1, (uint16_t)(curN * sizeof(Y_T) / AscendC::DEFAULT_C0_SIZE), (uint16_t)0, (uint16_t)0};
            auto cInLocal = matmulQueue.AllocTensor<INNER_T>();
            matmulObj.template GetTensorC<false>(cInLocal);
            matmulObj.WaitGetTensorC();
            matmulQueue.EnQue(cInLocal);
            if (tileM != 0) {
                // Padding row of the padded singleCoreM: drain the tile (the
                // fetch must stay balanced with Iterate) but emit nothing.
                AscendC::LocalTensor<INNER_T> padTile = matmulQueue.DeQue<INNER_T>();
                matmulQueue.FreeTensor(padTile);
                continue;
            }

            AscendC::LocalTensor<Y_T> yInLocalCube = vectorYInQueue.AllocTensor<Y_T>();
            DataCopy(yInLocalCube, yInGm_[offset], elements);
            vectorYInQueue.EnQue(yInLocalCube);

            AscendC::LocalTensor<INNER_T> tmpTensor = calcBuf.Get<INNER_T>();
            AscendC::LocalTensor<Y_T> yInLocal = vectorYInQueue.DeQue<Y_T>();
            AscendC::Cast(tmpTensor, yInLocal, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::PipeBarrier<PIPE_V>();
            vectorYInQueue.FreeTensor(yInLocal);

            AscendC::LocalTensor<INNER_T> yLocal = matmulQueue.DeQue<INNER_T>();
            AscendC::Add(tmpTensor, tmpTensor, yLocal, elements);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<Y_T> yOutLocal = vectorOutQueue.AllocTensor<Y_T>();
            AscendC::Cast(yOutLocal, tmpTensor, AscendC::RoundMode::CAST_RINT, elements);
            AscendC::PipeBarrier<PIPE_V>();

            vectorOutQueue.EnQue<Y_T>(yOutLocal);
            calcBuf.FreeTensor(tmpTensor);
            matmulQueue.FreeTensor(yLocal);

            AscendC::LocalTensor<Y_T> outputCopy = vectorOutQueue.DeQue<Y_T>();
            DataCopy(yOutGm_[offset], outputCopy, copyParams);
            vectorOutQueue.FreeTensor(outputCopy);
        }
        matmulObj.End();
    }

private:
    AscendC::TPipe *pipe_;

    MAT_TYPE matmulObj;
    TCubeTiling tiling;

    AscendC::GlobalTensor<INNER_T> workspaceGlobal;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> matmulQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> vectorYInQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> vectorOutQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<X_T> xInGm_;
    AscendC::GlobalTensor<W_T> wInGm_;
    AscendC::GlobalTensor<Y_T> yInGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;

    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> loraIndicesGm_;
    AscendC::GlobalTensor<int32_t> loraRanksGm_;
    AscendC::GlobalTensor<int32_t> sliceOffsetsGm_;

    uint32_t batchSize_;
    uint32_t sliceCount_;
    uint32_t maxLoRARank_;
    uint32_t outputHiddenDim_;
    uint32_t sliceOffset_;
    uint32_t outputFullDim_;
    uint32_t singleLoRAWeightLen_;
    int64_t reqLoRAIndex_;
    int32_t reqLoRARank_;
    uint64_t reqLoRAWeightOffset_;
    int32_t reqSlice_;
    uint32_t numOutputElementsPerInputTile_;
    uint32_t numStreamInPerOutputTile_;
    uint64_t yOffset_;
};

extern "C" __global__ __aicore__ void sgemmc_expand(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices,
                                                    uint32_t loraIndicesSize, GM_ADDR seqLen, uint32_t seqLenSize,
                                                    GM_ADDR loraRanks, uint32_t loraRanksSize, GM_ADDR sliceOffsets,
                                                    uint32_t sliceOffsetsSize, GM_ADDR yIn, GM_ADDR yOut,
                                                    uint32_t batchSize, uint32_t maxLoRARank, uint32_t outputFullDim,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    AscendC::TPipe pipe;
    sglang::npu_kernel::SGEMMCTilingData tilingData;
    kernel_utils::CopyTiling(&tilingData, tiling);

    if (tilingData.tilingKey == 1) {
        SGEMMCExpand<bfloat16_t, float> op(&pipe);
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, sliceOffsets,
                sliceOffsetsSize, yIn, yOut, batchSize, maxLoRARank, outputFullDim, workspace, tilingData.cubeTiling);
        op.Process();
    } else {
        SGEMMCExpand<half, float> op(&pipe);
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, sliceOffsets,
                sliceOffsetsSize, yIn, yOut, batchSize, maxLoRARank, outputFullDim, workspace, tilingData.cubeTiling);
        op.Process();
    }
}

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMC_EXPAND_H
