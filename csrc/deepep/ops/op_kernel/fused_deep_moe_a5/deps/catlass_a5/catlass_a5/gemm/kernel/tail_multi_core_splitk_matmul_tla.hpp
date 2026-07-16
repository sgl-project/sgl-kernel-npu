/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_TAIL_SPLITK_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_TAIL_SPLITK_MATMUL_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class TailMultiCoreSplitkMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementBias = typename BlockMmad::ElementBias;

    using BlockScheduler = BlockScheduler_;

    static_assert(BlockMmad::TileCopy::ReluEnable == false, "Splitk template can not use the Relu with FixPipe !!!");

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrBias;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrWorkspace_, GM_ADDR ptrBias_ = nullptr)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWorkspace(ptrWorkspace_), ptrBias(ptrBias_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        uint32_t aicCoreNum;
        GM_ADDR ptrBias{nullptr};
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t minSpaceSize = 10 * 1024 * 1024; // 2M
        size_t workspaceSize =
            static_cast<size_t>(L1_TILE_M) * L1_TILE_N * sizeof(ElementAccumulator) * args.aicCoreNum;
        return minSpaceSize > workspaceSize ? minSpaceSize : workspaceSize;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{
            args.problemShape,
            args.ptrA,
            args.layoutA,
            args.ptrB,
            args.layoutB,
            args.ptrC,
            args.layoutC,
            workspace,
            args.ptrBias
        };
        return params;
    }

    // Methods
    CATLASS_DEVICE
    TailMultiCoreSplitkMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        BlockScheduler matmulBlockScheduler(params.problemShape, GemmCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K), blockDim);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementAccumulator> gmW;
        gmW.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrWorkspace);

        using GlobalTensorBiasType = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;
        AscendC::GlobalTensor<GlobalTensorBiasType> gmBias;
        if constexpr (!std::is_void_v<ElementBias>) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
        // data reads will bypass L2 Cache.
        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        auto layoutW = tla::MakeLayout<ElementAccumulator, layout::RowMajor>(L1_TILE_M, L1_TILE_N);
        auto layoutBias = tla::MakeLayout(params.problemShape.n());

        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorW = tla::MakeTensor(gmW[blockIdx * L1_TILE_M * L1_TILE_N], layoutW, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        // The number of normal blocks
        uint32_t normalBlockNum = matmulBlockScheduler.GetNormalBlockNum();
        uint32_t tailCores = coreLoops - normalBlockNum;

        if (blockIdx >= tailCores) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        for (uint32_t loopIdx = blockIdx; loopIdx < coreLoops; loopIdx += blockDim) {
            uint32_t actualLoopIdx = loopIdx;
            if (loopIdx == normalBlockNum - blockDim + blockIdx && blockIdx < tailCores && normalBlockNum > 0) {
                actualLoopIdx = normalBlockNum + blockIdx;
            } else if (loopIdx >= normalBlockNum && normalBlockNum > 0) {
                actualLoopIdx = normalBlockNum - blockDim + blockIdx;
            }

            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(actualLoopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord, actualLoopIdx);

            // Make tiled views
            auto tensorBlockA = GetTile(
                tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );
            auto tensorBlockW =
                GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            // Compute block-scoped matrix multiply-add
            if (!matmulBlockScheduler.IsTailBlock(actualLoopIdx)) {
                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
                } else {
                    auto tensorBlockBias = GetTile(
                        tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n())
                    );
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockBias);
                }
            } else {
                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockW, actualBlockShape);
                } else {
                    if (blockCoord.k() == 0) {
                        auto tensorBlockBias = GetTile(
                            tensorBias, tla::MakeCoord(blockCoord.n() * L1_TILE_N), tla::MakeShape(actualBlockShape.n())
                        );
                        blockMmad(tensorBlockA, tensorBlockB, tensorBlockW, actualBlockShape, tensorBlockBias);
                    } else {
                        blockMmad(tensorBlockA, tensorBlockB, tensorBlockW, actualBlockShape);
                    }
                }
            }

            if (loopIdx == normalBlockNum - blockDim + blockIdx && blockIdx < tailCores && normalBlockNum > 0) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore);
            }
            if (normalBlockNum == 0) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagAicFinishStore);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmWorkspace;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        gmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrWorkspace));

        uint32_t SPACE_CAPATICY = ArchTag::UB_SIZE / sizeof(ElementAccumulator);
        AscendC::LocalTensor<ElementAccumulator> accumulatorBuffer =
            resource.ubBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::LocalTensor<ElementC> outputBuffer = resource.ubBuf.template GetBufferByByte<ElementC>(0);

        constexpr uint32_t ELE_NUM_ALIGN = BYTE_PER_BLK / sizeof(ElementC);

        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t aivNum = blockDim * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aicId = aivId / AscendC::GetSubBlockNum();

        BlockScheduler matmulBlockScheduler(params.problemShape, GemmCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K), blockDim);
        // The number of tail block using the streamk algorithm
        uint32_t tailBlockNum = matmulBlockScheduler.GetTailBlockNum();
        // The number of normal blocks
        uint32_t normalBlockNum = matmulBlockScheduler.GetNormalBlockNum();
        uint32_t splitkFactor = matmulBlockScheduler.GetSplitkFactor();

        if (aicId >= tailBlockNum * splitkFactor) {
            return;
        }

        uint32_t startCoreIdx = aicId / splitkFactor * splitkFactor;
        uint32_t laborCoreNum = splitkFactor * AscendC::GetSubBlockNum();

        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(normalBlockNum + aicId);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord, normalBlockNum + aicId);
        uint32_t tileNum = actualBlockShape.m();
        uint32_t tileLen = actualBlockShape.n();
        uint32_t tilePerCoreMax = (SPACE_CAPATICY / laborCoreNum) / RoundUp(tileLen, ELE_NUM_ALIGN);
        uint32_t tilePerCore = CeilDiv(tileNum, laborCoreNum);
        if (tilePerCore > tilePerCoreMax) {
            tilePerCore = tilePerCoreMax;
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        uint32_t loopsNum = CeilDiv(tileNum, tilePerCore);
        uint32_t loopStart = aivId - startCoreIdx * AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = loopStart; loopIdx < loopsNum; loopIdx += laborCoreNum) {
            uint32_t tilesActual = tilePerCore;
            uint32_t computeNum = tilesActual * RoundUp(tileLen, ELE_NUM_ALIGN);
            if (loopIdx == loopsNum - 1) {
                tilesActual = tileNum - loopIdx * tilePerCore;
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopyExtParams dataCopyParamsIn(
                tilesActual, tileLen * sizeof(ElementAccumulator), (L1_TILE_N - tileLen) * sizeof(ElementAccumulator),
                (RoundUp(tileLen, ELE_NUM_ALIGN) - tileLen) * sizeof(ElementAccumulator) / BYTE_PER_BLK, 0
            );
            AscendC::DataCopyPadExtParams<ElementAccumulator> padParams(false, 0, 0, 0);

            uint64_t srcOffset = startCoreIdx * L1_TILE_M * L1_TILE_N + loopIdx * tilePerCore * L1_TILE_N;
            for (uint32_t sliceIdx = 0; sliceIdx < splitkFactor; ++sliceIdx) {
                AscendC::DataCopyPad(
                    accumulatorBuffer[computeNum * sliceIdx], gmWorkspace[srcOffset + sliceIdx * L1_TILE_M * L1_TILE_N],
                    dataCopyParamsIn, padParams
                );
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            for (uint32_t sliceIdx = 1; sliceIdx < splitkFactor; ++sliceIdx) {
                AscendC::Add(accumulatorBuffer, accumulatorBuffer, accumulatorBuffer[computeNum * sliceIdx], computeNum);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if constexpr (!std::is_same_v<ElementAccumulator, ElementC>) {
                if constexpr (std::is_same_v<ElementC, half>) {
                    AscendC::Cast(outputBuffer, accumulatorBuffer, AscendC::RoundMode::CAST_NONE, computeNum);
                } else {
                    AscendC::Cast(outputBuffer, accumulatorBuffer, AscendC::RoundMode::CAST_RINT, computeNum);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // layoutC can only be RowMajor
            uint64_t dstStride = tla::get<0>(params.layoutC.stride());
            uint64_t dstOffset = (static_cast<uint64_t>(blockCoord.m()) * L1_TILE_M + loopIdx * tilePerCore) * dstStride
                                 + static_cast<uint64_t>(blockCoord.n()) * L1_TILE_N;
            AscendC::DataCopyExtParams dataCopyParamsOut(
                tilesActual, tileLen * sizeof(ElementC), (RoundUp(tileLen, ELE_NUM_ALIGN) - tileLen) / ELE_NUM_ALIGN,
                (dstStride - tileLen) * sizeof(ElementC), 0
            );
            AscendC::DataCopyPad(gmC[dstOffset], outputBuffer, dataCopyParamsOut);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 0;
    Arch::CrossCoreFlag flagAicFinishStore{FLAG_AIC_FINISH};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_TAIL_SPLITK_MATMUL_TLA_HPP