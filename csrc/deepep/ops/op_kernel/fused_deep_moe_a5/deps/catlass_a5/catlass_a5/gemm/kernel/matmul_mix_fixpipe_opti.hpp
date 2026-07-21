/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/status.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulMixFixpipeOpti
{
public:
    CATLASS_DEVICE
    KernelMatmulMixFixpipeOpti()
    {
        // Init UbTensor
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubTensorList_[i] = resource.ubBuf.template GetBufferByByte<ElementC>(UB_TILE_SIZE * i);
        }
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < UB_STAGES; ++i) {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + i);
            }
        }
    }
    CATLASS_DEVICE
    ~KernelMatmulMixFixpipeOpti()
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < UB_STAGES; ++i) {
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + i);
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX + i);
            }
        }
    }

    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;

    AscendC::GlobalTensor<ElementA> aGlobal_;
    AscendC::GlobalTensor<ElementB> bGlobal_;
    AscendC::GlobalTensor<ElementC> cGlobal_;

    constexpr static uint32_t ML1_ = BlockMmad::ML1_;
    constexpr static uint32_t NL1_ = BlockMmad::NL1_;
    constexpr static uint32_t KL1_ = BlockMmad::KL1_;

    constexpr static uint32_t ML0_ = BlockMmad::ML0_;
    constexpr static uint32_t NL0_ = BlockMmad::NL0_;
    constexpr static uint32_t KL0_ = BlockMmad::KL0_;

    constexpr static uint32_t UB_STAGES = 2;
    constexpr static uint32_t UB_TILE_SIZE = ArchTag::UB_SIZE / UB_STAGES;

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        Arguments() = default;
    };

    struct Params {
        GemmCoord problemShape;
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        Params() = default;
    };

    CATLASS_DEVICE
    void operator()(Params const &params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        m_ = params.problemShape.m();
        n_ = params.problemShape.n();
        k_ = params.problemShape.k();

        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.aGmAddr), m_ * k_);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.bGmAddr), k_ * n_);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.cGmAddr), m_ * n_);

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m_, k_);
        auto aTlaTensor = tla::MakeTensor(aGlobal_, layoutA, Arch::PositionGM{});

        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k_, n_);
        auto bTlaTensor = tla::MakeTensor(bGlobal_, layoutB, Arch::PositionGM{});

        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m_, n_);
        auto cTlaTensor = tla::MakeTensor(cGlobal_, layoutC, Arch::PositionGM{});

        // dualDstCtrl is not supported in quant/dequant scenarios
        constexpr bool enableDualDst = (AscendC::IsSameType<ElementC, ElementAccumulator>::value);
        if ASCEND_IS_AIV {
            if (!enableDualDst && AscendC::GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        // Instantiate
        BlockMmad blockMmadOp(resource);
        BlockEpilogue epilogueOp;
        BlockScheduler bs(curBlockIdx, blockNum, params.problemShape);

        if (bs.endBlockIdx_ + 1 <= blockNum / 2) {
            bs.UpdateTailTile();
        }

        uint32_t ubPingPongFlag = UB_STAGES > 1 ? 1 : 0;
        uint32_t coreLoops = bs.round_;
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            bool isLastLoop = (loopIdx == coreLoops - 1 && curBlockIdx <= bs.endBlockIdx_);
            bs.UpdateMNTileIdx(loopIdx, isLastLoop);
            bs.UpdateBlockShape(loopIdx, isLastLoop);

            auto blockShape = bs.GetBlockShape();
            auto blkElemCoord = bs.GetBlockCoordByElement();

            uint32_t blockM = blockShape.m();
            uint32_t blockN = blockShape.n();

            uint32_t mCoord = blkElemCoord.m();
            uint32_t nCoord = blkElemCoord.n();

            auto aTileTensor = GetTile(aTlaTensor, tla::MakeCoord(mCoord, 0), tla::MakeShape(blockM, k_));

            auto bTileTensor = GetTile(bTlaTensor, tla::MakeCoord(0, nCoord), tla::MakeShape(k_, blockN));

            auto cTileTensor = GetTile(cTlaTensor, tla::MakeCoord(mCoord, nCoord), tla::MakeShape(blockM, blockN));

            uint32_t ubListId = (loopIdx / blockNum) & ubPingPongFlag;

            int64_t alignN = RoundUp(blockN, static_cast<int64_t>(Catlass::BYTE_PER_BLK / sizeof(ElementC)));
            auto ubLayout = tla::MakeLayout<ElementC, LayoutTagC>(blockM, alignN);
            auto cLocalTensor = tla::MakeTensor(ubTensorList_[ubListId], ubLayout, Arch::PositionUB{});

            if ASCEND_IS_AIC {
                // Synchronize with aiv
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + (ubListId));
                if constexpr (enableDualDst) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + (ubListId) +
                                                                              FLAG_ID_MAX);
                }
                // Calculate blockMmad
                blockMmadOp(aTileTensor, bTileTensor, cLocalTensor, blockShape);
                // Notify aiv
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + (ubListId));
                if constexpr (enableDualDst) {
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + (ubListId) +
                                                                             FLAG_ID_MAX);
                }
            }
            if ASCEND_IS_AIV {
                // Synchronize with aic
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG + (ubListId));
                // Calculate epilogue
                epilogueOp(cTileTensor, cLocalTensor);
                // Notify aic
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + (ubListId));
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_HOST_DEVICE
    static bool CanImplement(Arguments const &args)
    {
        return true;
    }

    CATLASS_HOST_DEVICE
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(Arguments const &args, GM_ADDR workspace)
    {
        return {args.problemShape, args.aGmAddr, args.bGmAddr, args.cGmAddr};
    }

private:
    Arch::Resource<ArchTag> resource;
    AscendC::LocalTensor<ElementC> ubTensorList_[UB_STAGES];
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static int16_t FLAG_ID_MAX = 16;
    int64_t m_{1};
    int64_t n_{1};
    int64_t k_{1};
};

}  // namespace Catlass::Gemm::Kernel
#endif  // CATLASS_GEMM_KERNEL_MATMUL_MIX_FIXPIPE_OPTI_HPP
