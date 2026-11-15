// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
//
// Migrated from csrc/lora/op_kernel/sgemmc_shrink_kernel.cpp. The computation
// body is kept unchanged; only the entry point and parameter plumbing follow
// the independent CANN operator framework.

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sgemmc_shrink_tiling.h"
#include "../../common/kernel/lora_common_kernel.h"

constexpr uint32_t BLOCK_SIZE = 16U;

template <typename scalar_t, typename inner_t>
class KernelSgemmcShrink
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
    __aicore__ explicit KernelSgemmcShrink(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR loraScales, uint32_t loraScalesSize, GM_ADDR y, uint32_t batchSize,
                                uint32_t inputHiddenDim, uint32_t maxLoRARank, uint32_t slices, GM_ADDR workspace,
                                const TCubeTiling &tiling)
    {
        this->tiling = tiling;

        slices_ = slices;
        batchSize_ = batchSize;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        singleLoRAWeightLen_ = inputHiddenDim_ * maxLoRARank_ * slices_;

        xInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        yOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(y));
        wInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        loraIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraIndices), loraIndicesSize);
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen), seqLenSize);
        loraRanksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraRanks), loraRanksSize);
        loraScalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(loraScales), loraScalesSize);

        workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ INNER_T *>(workspace));
    }

    __aicore__ inline void Process()
    {
        if (GetSysWorkSpacePtr() == nullptr) {
            return;
        }

        int64_t blocks = AscendC::GetBlockNum();
        int64_t blockIdx = AscendC::GetBlockIdx();

        if ASCEND_IS_AIV {
            if (AscendC::GetSubBlockIdx() == 1) {
                return;
            }
            blockIdx /= AscendC::GetSubBlockNum();
        }

        int64_t tokenIdx = blockIdx / slices_;
        int64_t sliceIdx = blockIdx % slices_;

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
        reqLoRAScale_ = loraScalesGm_.GetValue(reqLoRAIndex_);

        if (reqLoRARank_ == 0) {
            return;
        }

        uint32_t baseM = min(tiling.baseM, tiling.singleCoreM);
        uint32_t baseN = min(tiling.baseN, min(tiling.singleCoreN, reqLoRARank_));
        uint32_t elements = baseM * baseN;
        uint32_t maxElements = tiling.baseM * tiling.baseN;
        // Same per-block scratch rule as sgemmc_expand: the cube stages every N
        // chunk under the block's scratch base, so reserve the whole N.
        uint32_t scratchStride = tiling.baseM * tiling.N;

        workspaceGlobal = workspaceGlobal[blockIdx * scratchStride];

        REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), matmulObj, &tiling);

        matmulObj.DisableBias();
        matmulObj.SetWorkspace(workspaceGlobal);
        matmulObj.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
        matmulObj.SetSingleShape(tiling.singleCoreM, reqLoRARank_, tiling.singleCoreK);
        matmulObj.SetTensorA(xInGm_[tokenIdx * inputHiddenDim_], false);
        matmulObj.SetTensorB(wInGm_[reqLoRAWeightOffset_ + sliceIdx * inputHiddenDim_ * reqLoRARank_], true);
        matmulObj.template Iterate<false>();

        pipe_->InitBuffer(calcBuf, maxElements * sizeof(INNER_T));
        pipe_->InitBuffer(matmulQueue, 1, maxElements * sizeof(INNER_T));
        pipe_->InitBuffer(outQueue, 1, maxElements * sizeof(Y_T));

        AscendC::DataCopyParams copyParams = {
            (uint16_t)baseM, (uint16_t)(baseN * sizeof(Y_T) / AscendC::DEFAULT_C0_SIZE), (uint16_t)0,
            (uint16_t)((slices_ * tiling.N - baseN) * sizeof(Y_T) / AscendC::DEFAULT_C0_SIZE)};
        uint32_t iteratations = AscendC::Ceil(tiling.singleCoreM, baseM) * AscendC::Ceil(reqLoRARank_, baseN);
        uint32_t outputOffset = tokenIdx * slices_ * maxLoRARank_ + sliceIdx * reqLoRARank_;
        for (uint32_t i = 0; i < iteratations; ++i) {
            AscendC::LocalTensor<INNER_T> cInLocal = matmulQueue.AllocTensor<INNER_T>();
            matmulObj.template GetTensorC<false>(cInLocal);
            matmulObj.WaitGetTensorC();
            matmulQueue.EnQue(cInLocal);

            AscendC::LocalTensor<INNER_T> tmpTensor = calcBuf.Get<INNER_T>();
            AscendC::LocalTensor<INNER_T> mmResTensor = matmulQueue.DeQue<INNER_T>();
            AscendC::LocalTensor<Y_T> output = outQueue.AllocTensor<Y_T>();

            AscendC::Muls(tmpTensor, mmResTensor, reqLoRAScale_, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(output, tmpTensor, AscendC::RoundMode::CAST_RINT, elements);
            AscendC::PipeBarrier<PIPE_V>();

            outQueue.EnQue(output);
            matmulQueue.FreeTensor(mmResTensor);
            calcBuf.FreeTensor(tmpTensor);

            AscendC::LocalTensor<Y_T> outputCopy = outQueue.DeQue<Y_T>();
            DataCopy(yOutGm_[outputOffset + i * baseN], outputCopy, copyParams);
            outQueue.FreeTensor(outputCopy);
        }
        matmulObj.End();
    }

private:
    AscendC::TPipe *pipe_;

    TCubeTiling tiling;
    MAT_TYPE matmulObj;

    AscendC::GlobalTensor<X_T> xInGm_;
    AscendC::GlobalTensor<W_T> wInGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    AscendC::GlobalTensor<int32_t> loraIndicesGm_;
    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> loraRanksGm_;
    AscendC::GlobalTensor<half> loraScalesGm_;

    AscendC::GlobalTensor<INNER_T> workspaceGlobal;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> matmulQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;

    uint32_t slices_;
    uint32_t batchSize_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    uint32_t singleLoRAWeightLen_;

    uint64_t reqLoRAWeightOffset_;
    int32_t reqLoRAIndex_;
    int32_t reqLoRARank_;
    INNER_T reqLoRAScale_;
};

extern "C" __global__ __aicore__ void sgl_sgemmc_shrink(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, GM_ADDR seqLen,
                                                        GM_ADDR loraRanks, GM_ADDR loraScales, GM_ADDR y, GM_ADDR yRef,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(optiling::SglSgemmcShrinkTilingData);
    GET_TILING_DATA(data, tiling);
    (void)yRef;
    AscendC::TPipe pipe;
    KernelSgemmcShrink<DTYPE_X, float> op(&pipe);
    // y and yRef alias the same buffer; write through the input pointer to keep
    // the legacy single-pointer behaviour.
    op.Init(x, weight, loraIndices, data.loraIndicesSize, seqLen, data.seqLenSize, loraRanks, data.loraRanksSize,
            loraScales, data.loraScalesSize, y, data.batchSize, data.inputHiddenDim, data.maxLoRARank, data.slicesCount,
            workspace, data.cubeTiling);
    op.Process();
}
