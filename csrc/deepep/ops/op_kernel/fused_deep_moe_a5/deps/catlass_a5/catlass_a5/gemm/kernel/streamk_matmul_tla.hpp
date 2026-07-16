/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_STREAMK_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_STREAMK_MATMUL_TLA_HPP

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
class StreamkMatmulTla {
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
        size_t minSpaceSize = 2 * 1024 * 1024; // 2M
        size_t workspaceSize =
            static_cast<size_t>(L1_TILE_M) * L1_TILE_N * sizeof(ElementAccumulator) * args.aicCoreNum * 2;
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
    StreamkMatmulTla() {}

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

        auto layoutBias = tla::MakeLayout(params.problemShape.n());
        // Represent the full tensors
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, params.layoutC, Arch::PositionGM{});
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});

        typename BlockScheduler::StreamkBlockDec streamkBlockDec;

        // The number of normal blocks
        uint32_t normalBlockNum = matmulBlockScheduler.GetNormalBlockNum();
        uint32_t streamkCores = coreLoops - normalBlockNum;

        if (blockIdx >= streamkCores) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        for (uint32_t loopIdx = blockIdx; loopIdx < coreLoops; loopIdx += blockDim) {

            uint32_t actualLoopIdx = loopIdx;
            if (loopIdx == normalBlockNum - blockDim + blockIdx && blockIdx < streamkCores && normalBlockNum > 0) {
                actualLoopIdx = normalBlockNum + blockIdx;
            } else if (loopIdx >= normalBlockNum && normalBlockNum > 0) {
                actualLoopIdx = normalBlockNum - blockDim + blockIdx;
            }

            // Compute block location
            matmulBlockScheduler.GetStreamkBlockDec(actualLoopIdx, streamkBlockDec);

            // Make tiled views
            auto tensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(streamkBlockDec.blockCoord.m() * L1_TILE_M, streamkBlockDec.blockCoord.k() * L1_TILE_K),
                tla::MakeShape(streamkBlockDec.actualBlockShape.m(), streamkBlockDec.actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(streamkBlockDec.blockCoord.k() * L1_TILE_K, streamkBlockDec.blockCoord.n() * L1_TILE_N),
                tla::MakeShape(streamkBlockDec.actualBlockShape.k(), streamkBlockDec.actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC,
                tla::MakeCoord(streamkBlockDec.blockCoord.m() * L1_TILE_M, streamkBlockDec.blockCoord.n() * L1_TILE_N),
                tla::MakeShape(streamkBlockDec.actualBlockShape.m(), streamkBlockDec.actualBlockShape.n())
            );

            auto layoutW = MakeLayout(
                tla::MakeShape(streamkBlockDec.actualBlockShape.m(), streamkBlockDec.actualBlockShape.n()),
                tla::MakeStride(L1_TILE_N, tla::Int<1>{})
            );
            auto tensorW = tla::MakeTensor(gmW[L1_TILE_M * L1_TILE_N * 2 * blockIdx], layoutW, Arch::PositionGM{});

            // Compute block-scoped matrix multiply-add
            if (!streamkBlockDec.isStreamkBlock) {
                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, streamkBlockDec.actualBlockShape);
                } else {
                    auto tensorBlockBias = GetTile(
                        tensorBias, tla::MakeCoord(streamkBlockDec.blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(streamkBlockDec.actualBlockShape.n())
                    );
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorBlockC, streamkBlockDec.actualBlockShape, tensorBlockBias
                    );
                }
            } else {
                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockB, tensorW, streamkBlockDec.actualBlockShape);
                } else {
                    if (streamkBlockDec.blockCoord.k() == 0) {
                        auto tensorBlockBias = GetTile(
                            tensorBias, tla::MakeCoord(streamkBlockDec.blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(streamkBlockDec.actualBlockShape.n())
                        );
                        blockMmad(
                            tensorBlockA, tensorBlockB, tensorW, streamkBlockDec.actualBlockShape, tensorBlockBias
                        );

                    } else {
                        blockMmad(tensorBlockA, tensorBlockB, tensorW, streamkBlockDec.actualBlockShape);
                    }
                }
            }

            if (streamkBlockDec.isCrossBlock) {
                auto tensorBlockA = GetTile(
                    tensorA,
                    tla::MakeCoord(
                        streamkBlockDec.streamkBlockCoord.m() * L1_TILE_M,
                        streamkBlockDec.streamkBlockCoord.k() * L1_TILE_K
                    ),
                    tla::MakeShape(
                        streamkBlockDec.streamkActualBlockShape.m(), streamkBlockDec.streamkActualBlockShape.k()
                    )
                );
                auto tensorBlockB = GetTile(
                    tensorB,
                    tla::MakeCoord(
                        streamkBlockDec.streamkBlockCoord.k() * L1_TILE_K,
                        streamkBlockDec.streamkBlockCoord.n() * L1_TILE_N
                    ),
                    tla::MakeShape(
                        streamkBlockDec.streamkActualBlockShape.k(), streamkBlockDec.streamkActualBlockShape.n()
                    )
                );
                auto layoutW = MakeLayout(
                    tla::MakeShape(
                        streamkBlockDec.streamkActualBlockShape.m(), streamkBlockDec.streamkActualBlockShape.n()
                    ),
                    tla::MakeStride(L1_TILE_N, tla::Int<1>{})
                );
                auto tensorW =
                    tla::MakeTensor(gmW[L1_TILE_M * L1_TILE_N * (2 * blockIdx + 1)], layoutW, Arch::PositionGM{});

                if constexpr (std::is_void_v<ElementBias>) {
                    blockMmad(tensorBlockA, tensorBlockB, tensorW, streamkBlockDec.streamkActualBlockShape);
                } else {
                    auto tensorBlockBias = GetTile(
                        tensorBias, tla::MakeCoord(streamkBlockDec.streamkBlockCoord.n() * L1_TILE_N),
                        tla::MakeShape(streamkBlockDec.streamkActualBlockShape.n())
                    );
                    blockMmad(
                        tensorBlockA, tensorBlockB, tensorW, streamkBlockDec.streamkActualBlockShape, tensorBlockBias
                    );
                }
            }

            if (loopIdx == normalBlockNum - blockDim + blockIdx && blockIdx < streamkCores && normalBlockNum > 0) {
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

        constexpr uint32_t ELE_NUM_ALIGN = BYTE_PER_BLK / sizeof(ElementC);

        AscendC::LocalTensor<ElementAccumulator> accumulatorBuffer =
            resource.ubBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::LocalTensor<ElementC> outputBuffer = resource.ubBuf.template GetBufferByByte<ElementC>(0);

        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t aivNum = blockDim * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aicId = aivId / AscendC::GetSubBlockNum();

        BlockScheduler matmulBlockScheduler(params.problemShape, GemmCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K), blockDim);
        // The number of tail block using the streamk algorithm
        uint32_t streamkBlockNum = matmulBlockScheduler.GetStreamkBlockNum();
        // The number of normal blocks
        uint32_t normalBlockNum = matmulBlockScheduler.GetNormalBlockNum();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        for (uint32_t skBlockId = 0; skBlockId < streamkBlockNum; ++skBlockId) {
            // If the head part of the current block and tail part of previous block are computed by the same core,
            // this flag is set to true.
            bool isHeadCross = matmulBlockScheduler.IsCross(skBlockId);
            bool isTailCross = matmulBlockScheduler.IsCross(skBlockId + 1);
            // Get the ID of first core for the current skBlock computation.
            uint32_t startCoreIdx = matmulBlockScheduler.GetCoreIdx(skBlockId);
            // Get the ID of first core for the next skBlock computation.
            uint32_t endCoreIdx = matmulBlockScheduler.GetCoreIdx(skBlockId + 1);

            uint32_t laborCoreNum = (endCoreIdx - startCoreIdx) * AscendC::GetSubBlockNum();

            if ((aicId < startCoreIdx) || (aicId >= endCoreIdx)) {
                continue;
            }

            if (isTailCross) {
                endCoreIdx = endCoreIdx + 1;
            }

            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(normalBlockNum + skBlockId);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            uint32_t tileNum = actualBlockShape.m();
            uint32_t tileLen = actualBlockShape.n();

            uint32_t SPACE_CAPATICY = ArchTag::UB_SIZE / sizeof(ElementAccumulator);
            uint32_t tilePerCoreMax = (SPACE_CAPATICY / laborCoreNum) / RoundUp(tileLen, ELE_NUM_ALIGN);
            uint32_t tilePerCore = CeilDiv(tileNum, laborCoreNum);
            if (tilePerCore > tilePerCoreMax) {
                tilePerCore = tilePerCoreMax;
            }
            uint32_t splitkSliceNum = endCoreIdx - startCoreIdx;
            uint32_t loopsNum = CeilDiv(tileNum, tilePerCore);
            uint32_t loopStart = aivId - startCoreIdx * AscendC::GetSubBlockNum();

            for (uint32_t loopIdx = loopStart; loopIdx < loopsNum; loopIdx += laborCoreNum) {
                uint32_t tilesActual = tilePerCore;
                if (loopIdx == loopsNum - 1) {
                    tilesActual = tileNum - loopIdx * tilePerCore;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::DataCopyExtParams dataCopyParamsIn(
                    tilesActual, tileLen * sizeof(ElementAccumulator),
                    (L1_TILE_N - tileLen) * sizeof(ElementAccumulator),
                    (RoundUp(tileLen, ELE_NUM_ALIGN) - tileLen) * sizeof(ElementAccumulator) / BYTE_PER_BLK, 0
                );
                AscendC::DataCopyPadExtParams<ElementAccumulator> padParams(false, 0, 0, 0);
                if (isHeadCross) {
                    uint64_t skBlockOffset = (startCoreIdx * 2 + 1) * L1_TILE_M * L1_TILE_N;
                    uint64_t srcOffset = skBlockOffset + loopIdx * tilePerCore * L1_TILE_N;
                    AscendC::DataCopyPad(accumulatorBuffer, gmWorkspace[srcOffset], dataCopyParamsIn, padParams);
                }

                uint32_t sliceStart = isHeadCross ? 1 : 0;
                uint32_t computeNum = tilesActual * RoundUp(tileLen, ELE_NUM_ALIGN);
                for (uint32_t sliceIdx = sliceStart; sliceIdx < splitkSliceNum; ++sliceIdx) {
                    uint64_t skBlockOffset = (sliceIdx + startCoreIdx) * 2 * L1_TILE_M * L1_TILE_N;
                    uint64_t srcOffset = skBlockOffset + loopIdx * tilePerCore * L1_TILE_N;
                    uint64_t dstOffset = computeNum * sliceIdx;
                    AscendC::DataCopyPad(
                        accumulatorBuffer[dstOffset], gmWorkspace[srcOffset], dataCopyParamsIn, padParams
                    );
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                for (uint32_t sliceIdx = 1; sliceIdx < splitkSliceNum; ++sliceIdx) {
                    AscendC::Add(
                        accumulatorBuffer, accumulatorBuffer, accumulatorBuffer[computeNum * sliceIdx], computeNum
                    );
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
                uint64_t dstOffset = (static_cast<uint64_t>(blockCoord.m()) * L1_TILE_M + loopIdx * tilePerCore)
                                         * dstStride
                                     + static_cast<uint64_t>(blockCoord.n()) * L1_TILE_N;
                AscendC::DataCopyExtParams dataCopyParamsOut(
                    tilesActual, tileLen * sizeof(ElementC),
                    (RoundUp(tileLen, ELE_NUM_ALIGN) - tileLen) / ELE_NUM_ALIGN,
                    (dstStride - tileLen) * sizeof(ElementC), 0
                );
                AscendC::DataCopyPad(gmC[dstOffset], outputBuffer, dataCopyParamsOut);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
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

#endif // CATLASS_GEMM_KERNEL_STREAMK_MATMUL_TLA_HPP