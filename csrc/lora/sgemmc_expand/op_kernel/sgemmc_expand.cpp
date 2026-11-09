// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
//
// Migrated from csrc/lora/op_kernel/sgemmc_expand_kernel.cpp. The computation
// body is kept unchanged; only the entry point and parameter plumbing follow
// the independent CANN operator framework.

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sgemmc_expand_tiling.h"
#include "../../common/kernel/lora_common_kernel.h"

template <typename scalar_t, typename inner_t>
class KernelSgemmcExpand
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
    __aicore__ explicit KernelSgemmcExpand(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR sliceOffsets, uint32_t sliceOffsetsSize, GM_ADDR yIn, GM_ADDR yOut,
                                uint32_t batchSize, uint32_t maxLoRARank, uint32_t outputFullDim, GM_ADDR workspace,
                                const TCubeTiling &tiling)
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

        workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ INNER_T *>(workspace));
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
        uint32_t baseM = min(tiling.baseM, tiling.singleCoreM);
        uint32_t baseN = min(tiling.baseN, min(tiling.singleCoreN, slice));
        uint32_t elements = baseM * baseN;
        uint32_t maxElements = tiling.baseM * tiling.baseN;
        // The cube stages every N chunk of a block under that block's scratch
        // base, so the per-block stride has to cover the whole N handled by the
        // block. Reserving a single chunk makes the last chunk of a block land
        // on the next block's base and overwrite it.
        uint32_t scratchStride = tiling.baseM * tiling.N;

        workspaceGlobal = workspaceGlobal[blockIdx * scratchStride];

        REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), matmulObj, &tiling);

        matmulObj.DisableBias();
        matmulObj.SetWorkspace(workspaceGlobal);
        matmulObj.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
        matmulObj.SetSingleShape(tiling.singleCoreM, slice, reqLoRARank_);
        matmulObj.SetTensorA(xInGm_[tokenIdx * sliceCount_ * maxLoRARank_ + sliceIdx * reqLoRARank_], false);
        matmulObj.SetTensorB(wInGm_[reqLoRAWeightOffset_ + maxLoRARank_ * beginSlice], true);
        matmulObj.template Iterate<false>();

        pipe_->InitBuffer(calcBuf, maxElements * sizeof(INNER_T));
        pipe_->InitBuffer(matmulQueue, 1, maxElements * sizeof(INNER_T));
        pipe_->InitBuffer(vectorYInQueue, 1, maxElements * sizeof(Y_T));
        pipe_->InitBuffer(vectorOutQueue, 1, maxElements * sizeof(Y_T));

        AscendC::DataCopyParams copyParams = {(uint16_t)baseM,
                                              (uint16_t)(baseN * sizeof(Y_T) / AscendC::DEFAULT_C0_SIZE), (uint16_t)0,
                                              (uint16_t)((tiling.N - baseN) * sizeof(Y_T) / AscendC::DEFAULT_C0_SIZE)};
        uint32_t iterateTimes = AscendC::Ceil(tiling.singleCoreM, baseM) * AscendC::Ceil(slice, baseN);
        uint32_t outputOffset = tokenIdx * tiling.N + beginSlice;
        for (uint32_t i = 0; i < iterateTimes; ++i) {
            uint32_t offset = outputOffset + i * baseN;
            auto cInLocal = matmulQueue.AllocTensor<INNER_T>();
            matmulObj.template GetTensorC<false>(cInLocal);
            matmulObj.WaitGetTensorC();
            matmulQueue.EnQue(cInLocal);

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

extern "C" __global__ __aicore__ void sgl_sgemmc_expand(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, GM_ADDR seqLen,
                                                        GM_ADDR loraRanks, GM_ADDR sliceOffsets, GM_ADDR yIn,
                                                        GM_ADDR yOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(optiling::SglSgemmcExpandTilingData);
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    KernelSgemmcExpand<DTYPE_X, float> op(&pipe);
    op.Init(x, weight, loraIndices, data.loraIndicesSize, seqLen, data.seqLenSize, loraRanks, data.loraRanksSize,
            sliceOffsets, data.sliceOffsetsSize, yIn, yOut, data.batchSize, data.maxLoRARank, data.outputFullDim,
            workspace, data.cubeTiling);
    op.Process();
}
