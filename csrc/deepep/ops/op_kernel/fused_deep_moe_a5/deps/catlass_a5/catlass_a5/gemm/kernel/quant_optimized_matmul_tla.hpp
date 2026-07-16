/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_QUANT_OPTIMIZED_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_QUANT_OPTIMIZED_MATMUL_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "catlass_a5/gemm/tile/tile_copy_tla.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

template<
    class ArchTag_,
    class TensorIn_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrixBlockND {
public:
    using ArchTag = ArchTag_;
    using TensorIn = TensorIn_;
    using Element = typename TensorIn::Element;
    using LayoutIn = typename TensorIn::Layout;

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<AscendC::LocalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>,
        AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm =
        tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    using LayoutInnerDstGm = tla::Layout<
        tla::Shape<tla::Shape<uint32_t, uint32_t>, tla::Shape<uint32_t, uint32_t>>,
        tla::Stride<tla::Stride<int64_t, int64_t>, tla::Stride<tla::Int<1>, int64_t>>>;
    using TensorInnerDstGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInnerDstGm, tla::Coord<tla::_0, tla::_0>,
        AscendC::TPosition::GM>;

    using CopyGm2Ub = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerSrcGm, TensorInnerUb>;
    using CopyUb2Gm = Catlass::Gemm::Tile::TileCopyTlaExt<ArchTag, TensorInnerUb,
        TensorInnerDstGm, layout::RowMajor, layout::PaddingRowMajor>;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    PaddingMatrixBlockND(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            //在ub上分配空间
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            //每一片UB上的开均分到BUFFER_NUM的空间
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    template<class Tensor>
    CATLASS_DEVICE
    auto GetPaddingTensorSrc(Tensor const &tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInner>) {
            return tensor;
        } else {
            auto shape = tla::MakeShape(tla::get<1>(tensor.shape()), tla::get<0>(tensor.shape()));
            auto stride = tla::MakeStride(tla::get<1>(tensor.stride()), tla::get<0>(tensor.stride()));
            return tla::MakeTensor(tensor.data(), MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    template<class Tensor>
    CATLASS_DEVICE
    auto GetPaddingTensorDst(Tensor const &tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInnerDstGm>) {
            return tensor;
        } else {
            auto shape = tla::MakeShape(tla::get<1>(tensor.shape()), tla::get<0>(tensor.shape()));
            auto stride = tla::MakeStride(tla::get<1>(tensor.stride()), tla::get<0>(tensor.stride()));
            return tla::MakeTensor(tensor.data(), MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst &tensorDst, TensorSrc const& tensorSrc)
    {
        auto paddingTensorSrc = GetPaddingTensorSrc(tensorSrc);
        auto paddingTensorDst = GetPaddingTensorDst(tensorDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // 按照行切块，每行为一个tile块
        uint32_t tilesNum = tla::get<0>(paddingTensorSrc.shape());
        uint32_t tileLen = tla::get<1>(paddingTensorSrc.shape());
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(tla::get<1>(paddingTensorSrc.shape()));
        // 计算每一个aiv要计算的大小，对于剩余的工作从前向后增加
        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        //因为前面进行了工作重分配，所以相应后面的aiv处理的偏移量要后移
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);
        // 配置UB到GM的信号量
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        uint32_t coreLoops{ 0 };
        if (roundTileLen > COMPUTE_LENGTH) {
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = (tileLen + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                auto offset = tla::MakeCoord(mIdx + tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc = GetTile(
                    paddingTensorSrc,
                    offset,
                    tla::MakeShape(static_cast<uint32_t>(1), actualDataNum)
                );
                auto tensorTileDst = GetTile(
                    paddingTensorDst,
                    offset,
                    tla::MakeShape(static_cast<uint32_t>(1), actualDataNum)
                );

                auto layoutDstUb = MakeLayout(
                    tla::MakeShape(static_cast<uint32_t>(1), actualDataNum),
                    tla::MakeStride(static_cast<int64_t>(COMPUTE_LENGTH), tla::Int<1>{})
                );
                auto tensorDstUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto layoutSrcUb = MakeLayout(
                    tla::MakeShape(
                        CeilDiv(actualDataNum, tla::get<1, 0>(paddingTensorDst.shape())),
                        tla::get<1, 0>(paddingTensorDst.shape())
                    ),
                    tla::MakeStride(static_cast<int64_t>(tla::get<1, 0>(paddingTensorDst.shape())), tla::Int<1>{})
                );
                auto tensorSrcUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutSrcUb, Arch::PositionUB{});
                copyUb2Gm(tensorTileDst, tensorSrcUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc = GetTile(
                    paddingTensorSrc,
                    offset,
                    tla::MakeShape(actualTilesNum, tileLen)
                );

                auto layoutDstUb = MakeLayout(
                    tla::MakeShape(actualTilesNum, tileLen),
                    tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{})
                );
                auto tensorDstUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto layoutSrcUb = MakeLayout(
                    tla::MakeShape(
                        CeilDiv(tileLen, tla::get<1, 0>(paddingTensorDst.shape())),
                        tla::get<1, 0>(paddingTensorDst.shape())
                    ),
                    tla::MakeStride(static_cast<int64_t>(tla::get<1, 0>(paddingTensorDst.shape())), tla::Int<1>{})
                );
                for (uint32_t i = 0; i < actualTilesNum; ++i) {
                    auto tensorTileDst = GetTile(
                        paddingTensorDst,
                        tla::MakeCoord(mIdx + tileIdx + i, static_cast<uint32_t>(0)),
                        tla::MakeShape(static_cast<uint32_t>(1), tileLen)
                    );
                    auto tensorSrcUb = tla::MakeTensor(
                        inputBuffer[bufferIndex][i * roundTileLen],
                        layoutSrcUb,
                        Arch::PositionUB{}
                    );
                    copyUb2Gm(tensorTileDst, tensorSrcUb);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    CATLASS_DEVICE
    ~PaddingMatrixBlockND() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class PaddingA,
    class PaddingB,
    uint32_t WORKSPACE_STAGES_
>
class QuantOptimizedMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = std::conditional_t<std::is_void_v<PaddingA>, LayoutWA, typename LayoutHelper<PaddingA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /**
     * @struct BlockMmadArguments
     * @brief Kernel arguments for the host side
     */
    struct BlockMmadArguments {
        GM_ADDR ptrA{nullptr};          ///< The global memory address of matrix A
        LayoutA layoutA;
        GM_ADDR ptrB{nullptr};          ///< The global memory address of matrix B
        LayoutB layoutB;
        GM_ADDR ptrWA{nullptr};         ///< The global memory address of padding matrix A
        LayoutWA layoutWA;
        GM_ADDR ptrWB{nullptr};         ///< The global memory address of padding matrix B
        LayoutWB layoutWB;
    };

    using BlockMmadParams = BlockMmadArguments;
    using EpilogueParams = typename BlockEpilogue::Params;
    using EpilogueArguments = EpilogueParams;


    /// Parameters structure
    struct Params {
        GemmCoord problemShape;                 ///< Problem shape
        uint32_t aicCoreNum;
        BlockMmadParams mmadParams;             ///< MMAD parameters
        EpilogueParams epiParams;               ///< Epilogue parameters
        GM_ADDR ptrWorkspace;
        Params() = default;                     ///< Default constructor
    };

    struct Arguments {
        GemmCoord problemShape;                 ///< Problem shape
        uint32_t aicCoreNum;                    ///< Platform related info
        BlockMmadArguments mmadArgs;            ///< MMAD parameters
        EpilogueArguments epiArgs;              ///< Epilogue parameters
        Arguments() = default;                  ///< Default constructor
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t lenWorkspace = static_cast<size_t>(L1_TILE_M) * L1_TILE_N *
            args.aicCoreNum * WORKSPACE_STAGES;
        size_t sizeWorkspace = lenWorkspace * sizeof(ElementC);
        return sizeWorkspace;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        BlockMmadParams mmadParams = {args.mmadArgs.ptrA, args.mmadArgs.layoutA,
                                    args.mmadArgs.ptrB, args.mmadArgs.layoutB, 
                                    args.mmadArgs.ptrWA, args.mmadArgs.layoutWA,
                                    args.mmadArgs.ptrWB, args.mmadArgs.layoutWB};
        EpilogueParams epiParams = {args.epiArgs.ptrScale, args.epiArgs.layoutScale,
                                    args.epiArgs.ptrPerTokenScale, args.epiArgs.layoutPerTokenScale,
                                    args.epiArgs.ptrD, args.epiArgs.layoutD};
        // mmad params with epilogue takes workspaceGm as output
        Params params = {args.problemShape, args.aicCoreNum, mmadParams, epiParams, workspace};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    QuantOptimizedMatmulTla()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.mmadParams.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.mmadParams.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        auto tensorA = tla::MakeTensor(gmA, params.mmadParams.layoutWA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.mmadParams.layoutWB, Arch::PositionGM{});
        auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N);
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            Callback callbackBeforeFixpipe{};
            if (stageUsed == WORKSPACE_STAGES) {
                callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
            } else {
                ++stageUsed;
            }
            Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

            // Compute initial location in logical coordinates
            auto tensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC,
                tla::MakeCoord((stageId * coreNum + coreIdx) * L1_TILE_M, 0),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            bool isFirstBlock = (loopIdx == coreIdx);
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + coreNum < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + coreNum);
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
            }

            auto nextTensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(nextBlockCoord.m() * L1_TILE_M, nextBlockCoord.k() * L1_TILE_K),
                tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k())
            );
            auto nextTensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(nextBlockCoord.k() * L1_TILE_K, nextBlockCoord.n() * L1_TILE_N),
                tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n())
            );

            // Compute block-scoped matrix multiply-add
            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB,
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock,
                callbackBeforeFixpipe, callbackAfterFixpipe
            );

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId = (stageId >= stageUsed) ?
                (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if constexpr (!std::is_void_v<PaddingA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.mmadParams.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.mmadParams.ptrWA));
            auto tensorA = tla::MakeTensor(gmA, params.mmadParams.layoutA, Arch::PositionGM{});
            auto tensorWA = tla::MakeTensor(gmWA, params.mmadParams.layoutWA, Arch::PositionGM{});
            PaddingA paddingA(resource);
            paddingA(tensorWA, tensorA);
        }

        if constexpr (!std::is_void_v<PaddingB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.mmadParams.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.mmadParams.ptrWB));
            auto tensorB = tla::MakeTensor(gmB, params.mmadParams.layoutB, Arch::PositionGM{});
            auto tensorWB = tla::MakeTensor(gmWB, params.mmadParams.layoutWB, Arch::PositionGM{});
            PaddingB paddingB(resource);
            paddingB(tensorWB, tensorB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        BlockScheduler blockScheduler;
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N);
        auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementScale> gmScale;
        gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(params.epiParams.ptrScale));
        auto tensorScale = tla::MakeTensor(gmScale, params.epiParams.layoutScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(params.epiParams.ptrPerTokenScale));
        auto tensorPerTokenScale = tla::MakeTensor(gmPerTokenScale, params.epiParams.layoutPerTokenScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.epiParams.ptrD));
        auto tensorD = tla::MakeTensor(gmD, params.epiParams.layoutD, Arch::PositionGM{});

        uint32_t stageId = 0;

        blockScheduler.Update(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        blockEpilogue.UpdateParams(params.epiParams);
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoord);

            auto tensorBlockC = GetTile(
                tensorC, tla::MakeCoord((stageId * coreNum + coreIdx) * L1_TILE_M, 0),
                tla::MakeShape(actualBlockShapeMNK.m(), actualBlockShapeMNK.n())
            );
            
            auto tensorBlockScale = GetTile(
                tensorScale, tla::MakeCoord(0, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(tla::Int<1>{}, actualBlockShapeMNK.n())
            );
            
            auto tensorBlockPerTokenScaleScale = GetTile(
                tensorPerTokenScale, tla::MakeCoord(0, blockCoord.m() * L1_TILE_M),
                tla::MakeShape(tla::Int<1>{}, actualBlockShapeMNK.m())
            );
            
            auto tensorBlockD = GetTile(
                tensorD, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShapeMNK.m(), actualBlockShapeMNK.n())
            );

            Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
            blockEpilogue(tensorBlockC, tensorBlockScale, tensorBlockPerTokenScaleScale,
                          tensorBlockD, actualBlockShapeMNK);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler,
            PaddingA, PaddingB, WORKSPACE_STAGES>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = QuantOptimizedMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler,
            PaddingA, PaddingB, WORKSPACE_STAGES>;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];

    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 4;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_QUANT_OPTIMIZED_MATMUL_TLA_HPP
