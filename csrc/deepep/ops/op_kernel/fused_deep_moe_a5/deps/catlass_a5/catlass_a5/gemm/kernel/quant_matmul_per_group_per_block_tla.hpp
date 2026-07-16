/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_QUANT_MATMUL_PER_GROUP_PER_BLOCK_TLA_HPP
#define CATLASS_GEMM_KERNEL_QUANT_MATMUL_PER_GROUP_PER_BLOCK_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"
#include "catlass_a5/epilogue/block/block_epilogue.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_per_group_per_block.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class QuantMatmulPerGroupPerBlockTla {
public:
    CATLASS_DEVICE QuantMatmulPerGroupPerBlockTla()
    {
        if ASCEND_IS_AIV {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIV_SYNC_AIC_FLAG);      // ping
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIV_SYNC_AIC_FLAG + 1);  // pong
        }
    }
    CATLASS_DEVICE ~QuantMatmulPerGroupPerBlockTla()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);                    // ping
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1);                // pong
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);  
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX + 1);
        }
    }

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using LayoutTagA = typename BlockMmad::LayoutTagA;
    using LayoutTagB = typename BlockMmad::LayoutTagB;
    using LayoutTagC = typename BlockMmad::LayoutTagC;
    using YType = typename BlockEpilogue_::YType;

    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using ProblemShape = ProblemShape_;
    using BlockEpilogueParams = typename BlockEpilogue::Params;

    static constexpr bool transA = tla::detail::isColumnMajor<LayoutA>::value;
 	static constexpr bool transB = tla::detail::isColumnMajor<LayoutB>::value;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    using BlockOffset = tla::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;
        BlockEpilogueParams epilogueParams;
        
        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrBias_,
               BlockEpilogueParams epilogueParams_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrBias(ptrBias_), epilogueParams(epilogueParams_) {}

    };

    struct Arguments {
        GemmCoord problemShape;
        uint8_t *ptrA; LayoutA layoutA;
        uint8_t *ptrB; LayoutB layoutB;
        uint8_t *ptrC; LayoutC layoutC;
        uint8_t *ptrBias{nullptr};
        BlockEpilogueParams epilogueParams;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{args.problemShape,
            args.ptrA, args.layoutA,
            args.ptrB, args.layoutB,
            args.ptrC, args.layoutC,
            args.ptrBias, args.epilogueParams};
        return params;
    }


    CATLASS_DEVICE void UpdateMMGlobalAddr()
    {
        if ASCEND_IS_AIV {
            tla::Shape<int64_t, int64_t, int64_t, int64_t> baseOffset{
                static_cast<int64_t>(tla::get<IDX_C_OFFSET>(baseOffset_)),
                static_cast<int64_t>(tla::get<IDX_X2SCALE_OFFSET>(baseOffset_)),
                static_cast<int64_t>(tla::get<IDX_X1SCALE_OFFSET>(baseOffset_)),
                static_cast<int64_t>(tla::get<IDX_BIAS_OFFSET>(baseOffset_))};
            epilogueOp_.UpdateGlobalAddr(baseOffset);
        }
    }

    CATLASS_DEVICE void Init(const Params& params)
    {
        isPergroup_ = params.epilogueParams.groupSizeM == 1;
        constexpr uint32_t elems = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        mmResUb_ = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, 0, elems * UB_SUB_BANK_NUM);
        epilogueOp_.Init(&params.epilogueParams);

        problemShape_ = params.problemShape;
        if ASCEND_IS_AIV {
            epilogueOp_.UpdateParamsForNextProblem(problemShape_);
        }
    }
    
    CATLASS_DEVICE void operator()(const Params& params)
    {
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();

        Init(params);

        Arch::Resource<ArchTag> resource;

        if ASCEND_IS_AIV {
            curBlockIdx /= AscendC::GetTaskRation();
        }
        
        GemmCoord problemShape{params.problemShape.m(), params.problemShape.n(), params.problemShape.k()};
        BlockMmad blockMmad(problemShape);
        BlockScheduler bs(curBlockIdx, blockNum, problemShape);

        if (bs.endBlockIdx_ + 1 <= blockNum / 2) {
            bs.UpdateTailTile();
        }

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
        // data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        uint8_t crossPingPongID_{0};

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});

        UpdateMMGlobalAddr();

        uint32_t coreLoops = bs.round_;
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            bool isLastLoop = (loopIdx == coreLoops - 1 && curBlockIdx <= bs.endBlockIdx_);
            bs.UpdateMNTileIdx(loopIdx, isLastLoop);
            bs.UpdateBlockShape(loopIdx, isLastLoop);

            auto blockShape = bs.GetBlockShape();
            auto blkElemCoord = bs.GetBlockCoordByElement();
            
            uint32_t mCoord = blkElemCoord.m();
            uint32_t nCoord = blkElemCoord.n();
            uint32_t kCoord = blkElemCoord.k();

            uint32_t blockM = blockShape.m();
            uint32_t blockN = blockShape.n();
            uint32_t blockK = blockShape.k();

            int64_t alignN = RoundUp(blockN, UB_TWO_BANK_ELEMS_B32);
            auto ubLayout = tla::MakeLayout<ElementC, LayoutTagC>(blockM, alignN);
            auto tensorC = tla::MakeTensor(mmResUb_, ubLayout, Arch::PositionUB{});

            if ASCEND_IS_AIC {
                // Make tiled views
                auto tensorBlockA = GetTile(tensorA, 
                                            tla::MakeCoord(mCoord, kCoord),
                                            tla::MakeShape(blockM, blockK));
                auto tensorBlockB = GetTile(tensorB,
                                            tla::MakeCoord(kCoord, nCoord),
                                            tla::MakeShape(blockK, blockN));
                auto tensorBlockC = GetTile(tensorC,
                                            tla::MakeCoord(0, 0),
                                            tla::MakeShape(blockM, alignN));

                blockMmad(tensorBlockC, tensorBlockA, tensorBlockB, blockShape);
            }
            if ASCEND_IS_AIV {
                auto tensorBlockEpiolgue = GetTile(tensorC,
                                               tla::MakeCoord(mCoord, nCoord),
                                               tla::MakeShape(blockM, blockN));
                epilogueOp_(tensorBlockEpiolgue);
            }
        }
    }

private:
    BlockEpilogue epilogueOp_;
    GemmCoord problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0};

    static constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
    static constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
    static constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
    static constexpr uint64_t IDX_C_OFFSET = 5UL;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 8;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;
    static constexpr uint16_t FLAG_ID_MAX = 16;
    static constexpr uint32_t UB_TWO_BANK_ELEMS_B32 = 128U;
    static constexpr int64_t PER_BLOCK_SIZE = 128LL;
    static constexpr uint32_t UB_SUB_BANK_NUM = 2U;

    AscendC::GlobalTensor<ElementA> aGlobal_;
    AscendC::GlobalTensor<ElementB> bGlobal_;
    AscendC::LocalTensor<ElementC> mmResUb_;

    bool isBias_{false};
    bool isPergroup_;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_QUANT_MATMUL_PER_GROUP_PER_BLOCK_TLA_HPP