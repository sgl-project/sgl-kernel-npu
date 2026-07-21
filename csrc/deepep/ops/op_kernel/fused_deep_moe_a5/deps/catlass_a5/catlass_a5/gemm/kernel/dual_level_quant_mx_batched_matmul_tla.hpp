/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DUAL_LEVEL_QUANT_MX_BATCHED_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_DUAL_LEVEL_QUANT_MX_BATCHED_MATMUL_TLA_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Kernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

/**
 * Kernel: Dual-Level Quantization + MX FP4 Batch Matmul single-kernel path
 *
 * Single-kernel path:
 *   - 同一份输入 A/B (fp16/bf16),同一份输出 C (bf16)。
 *   - AIV 把 A/B 全量量化一次写入 workspace,再统一通知 AIC。
 *   - 取消 WORKSPACE_STAGES 模板参数 (新路径无 stage 复用)
 *   - 同步: AIV 和 AIC 两侧都调用 AscendC::SyncAll<false>(),硬件层面一次完成
 *     V-V / C-C / V↔C 三组同步,**不再需要单独的 FFTS V→C flag**。
 *     <isAIVOnly=false> 这一开关就是为 mix kernel 设计的。
 *
 * 收益: A/B 每行只量化一次,减少方阵场景下的重复量化工作。
 *
 * Template params:
 *   BlockMmad_      - MX matmul block
 *   BlockQuant_     - BlockQuantDualLevelMx<...>
 *   BlockScheduler_ - AIC MN tile scheduler
 *   ElementInput_   - float16_t | bfloat16_t
 */
template <class BlockMmad_, class BlockQuant_, class BlockScheduler_, typename ElementInput_>
class DualLevelQuantMxBatchedMatmulTla
{
public:
    using BlockMmad = BlockMmad_;
    using BlockQuant = BlockQuant_;
    using MmadArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;

    // FP4 matmul types (from BlockMmad,AIC 侧)
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;
    using LayoutMxScaleA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;
    using LayoutMxScaleB = typename BlockMmad::TileCopy::LayoutMxScaleB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    // Quantization types (from BlockQuant,AIV 侧)
    using ElementInput = ElementInput_;
    using LayoutInput = typename BlockQuant::LayoutInput;  // 实际为 RowMajor (A/B 物理都是 RowMajor)
    using LayoutOutput = typename BlockQuant::LayoutOutput;
    using ElementScale1 = typename BlockQuant::ElementScale1;
    using LayoutScale1 = typename BlockQuant::LayoutScale1;
    using ElementScale2 = typename BlockQuant::ElementScale2;
    using LayoutScale2 = typename BlockQuant::LayoutScale2;

    // Host 侧 layout 类型别名
    using LayoutInputA = LayoutInput;
    using LayoutInputB = LayoutInput;
    using LayoutOutputA = LayoutOutput;
    using LayoutOutputB = LayoutOutput;
    using LayoutScaleA1 = LayoutScale1;
    using LayoutScaleA2 = LayoutScale2;
    using LayoutScaleB1 = LayoutScale1;
    using LayoutScaleB2 = LayoutScale2;
    using ElementScaleA1 = ElementScale1;
    using ElementScaleB1 = ElementScale1;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    // *** P2-A: tile 尺寸必须从 BlockQuant 派生,不可写死 ***
    static constexpr uint32_t QUANT_TILE_ROWS = BlockQuant::SUB_TILE_M;
    static constexpr uint32_t QUANT_TILE_K = BlockQuant::SUB_TILE_K;
    static_assert(QUANT_TILE_ROWS == BlockQuant::SUB_TILE_M,
                  "QuantAllScheduler row tile must match BlockQuant::SUB_TILE_M");
    static_assert(QUANT_TILE_K == BlockQuant::SUB_TILE_K, "QuantAllScheduler K tile must match BlockQuant::SUB_TILE_K");

    static constexpr uint32_t LEVEL0_BLOCK_SIZE = BlockQuant::LEVEL0_BLOCK_SIZE;
    static constexpr uint32_t LEVEL1_BLOCK_SIZE = BlockQuant::LEVEL1_BLOCK_SIZE;

    // -----------------------------------------------------------------------
    // Arguments: host-side interface (与旧 kernel 故意保持一致,方便 diff)
    // -----------------------------------------------------------------------
    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;

        uint8_t *ptrInputA;
        LayoutInputA layoutInputA;
        uint8_t *ptrInputB;
        LayoutInputB layoutInputB;

        uint8_t *ptrC;
        LayoutC layoutC;

        uint8_t *ptrScaleA1;
        uint8_t *ptrScaleA2;
        LayoutMxScaleA layoutMxScaleA;
        uint8_t *ptrScaleB1;
        uint8_t *ptrScaleB2;
        LayoutMxScaleB layoutMxScaleB;

        LayoutOutputA layoutOutputA;
        LayoutOutputB layoutOutputB;
        LayoutScaleA1 layoutScaleA1;
        LayoutScaleA2 layoutScaleA2;
        LayoutScaleB1 layoutScaleB1;
        LayoutScaleB2 layoutScaleB2;

        LayoutA layoutQuantA;
        LayoutB layoutQuantB;
    };

    // -----------------------------------------------------------------------
    // Params: device-side
    // -----------------------------------------------------------------------
    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;

        GM_ADDR ptrInputA;
        LayoutInputA layoutInputA;
        int64_t strideInputA;
        GM_ADDR ptrInputB;
        LayoutInputB layoutInputB;
        int64_t strideInputB;

        GM_ADDR ptrQuantA;
        LayoutA layoutQuantA;
        LayoutOutputA layoutOutputA;
        int64_t strideQuantA;
        int64_t strideQuantABytes;
        GM_ADDR ptrQuantB;
        LayoutB layoutQuantB;
        LayoutOutputB layoutOutputB;
        int64_t strideQuantB;
        int64_t strideQuantBBytes;

        GM_ADDR ptrMxScaleA;
        LayoutMxScaleA layoutMxScaleA;
        LayoutScaleA2 layoutScaleA2;
        int64_t strideMxScaleA;
        GM_ADDR ptrMxScaleB;
        LayoutMxScaleB layoutMxScaleB;
        LayoutScaleB2 layoutScaleB2;
        int64_t strideMxScaleB;

        GM_ADDR ptrC;
        LayoutC layoutC;
        int64_t strideC;

        GM_ADDR ptrScaleA1;
        LayoutScaleA1 layoutScaleA1;
        int64_t strideScaleA1;
        GM_ADDR ptrScaleB1;
        LayoutScaleB1 layoutScaleB1;
        int64_t strideScaleB1;

        CATLASS_HOST_DEVICE Params() {}
    };

    // -----------------------------------------------------------------------
    // Static interface (与旧 kernel 一致)
    // -----------------------------------------------------------------------

    static bool CanImplement(const Arguments &args)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();

        if (k == 0 || k % 2 != 0) {
            return false;
        }
        if (m == 0 || n == 0 || args.batchCount == 0) {
            return false;
        }
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t sizeQuantA = static_cast<size_t>(args.layoutOutputA.Capacity()) / 2;
        size_t sizeQuantB = static_cast<size_t>(args.layoutOutputB.Capacity()) / 2;
        return args.batchCount * (sizeQuantA + sizeQuantB);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        uint32_t m = args.problemShape.m();
        uint32_t n = args.problemShape.n();
        uint32_t k = args.problemShape.k();

        size_t sizeQuantAPerBatch = static_cast<size_t>(args.layoutOutputA.Capacity()) / 2;
        size_t sizeQuantBPerBatch = static_cast<size_t>(args.layoutOutputB.Capacity()) / 2;

        Params params;
        params.batchCount = args.batchCount;
        params.problemShape = args.problemShape;

        params.ptrInputA = args.ptrInputA;
        params.layoutInputA = args.layoutInputA;
        params.strideInputA = static_cast<int64_t>(m) * k;
        params.ptrInputB = args.ptrInputB;
        params.layoutInputB = args.layoutInputB;
        params.strideInputB = static_cast<int64_t>(k) * n;

        params.ptrQuantA = workspace;
        params.layoutQuantA = args.layoutQuantA;
        params.layoutOutputA = args.layoutOutputA;
        params.strideQuantA = args.layoutOutputA.Capacity();
        params.strideQuantABytes = static_cast<int64_t>(sizeQuantAPerBatch);
        params.ptrQuantB = workspace + args.batchCount * sizeQuantAPerBatch;
        params.layoutQuantB = args.layoutQuantB;
        params.layoutOutputB = args.layoutOutputB;
        params.strideQuantB = args.layoutOutputB.Capacity();
        params.strideQuantBBytes = static_cast<int64_t>(sizeQuantBPerBatch);

        params.ptrMxScaleA = args.ptrScaleA2;
        params.layoutMxScaleA = args.layoutMxScaleA;
        params.layoutScaleA2 = args.layoutScaleA2;
        params.strideMxScaleA =
            static_cast<int64_t>(m) * static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(k) * MX_SCALE_COPY_GROUP_NUM);
        params.ptrMxScaleB = args.ptrScaleB2;
        params.layoutMxScaleB = args.layoutMxScaleB;
        params.layoutScaleB2 = args.layoutScaleB2;
        params.strideMxScaleB =
            static_cast<int64_t>(CeilDiv<MX_BASEK_FACTOR>(k) * MX_SCALE_COPY_GROUP_NUM) * static_cast<int64_t>(n);

        params.ptrC = args.ptrC;
        params.layoutC = args.layoutC;
        params.strideC = static_cast<int64_t>(m) * n;

        uint32_t scaleA1K = CeilDiv<LEVEL0_BLOCK_SIZE>(k);
        params.ptrScaleA1 = args.ptrScaleA1;
        params.layoutScaleA1 = args.layoutScaleA1;
        params.strideScaleA1 = static_cast<int64_t>(m) * scaleA1K;
        params.ptrScaleB1 = args.ptrScaleB1;
        params.layoutScaleB1 = args.layoutScaleB1;
        params.strideScaleB1 = static_cast<int64_t>(scaleA1K) * n;

        return params;
    }

    // -----------------------------------------------------------------------
    // Device-side execution
    //
    // 无成员需要初始化 (V↔C 同步走 SyncAll<false>(),不需要 FFTS flag),依赖默认构造。
    // -----------------------------------------------------------------------

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    // -----------------------------------------------------------------------
    // AIV path: 全量量化 A 与 B,然后参与 V↔C 全核同步通知 AIC
    //
    // 调度: QuantAllScheduler 把任务编码为 (operand, batch, rowTile, kTile),
    //       所有 AIV subblock round-robin 拿任务。详见 design.md §4.2。
    //
    // 同步: PipeBarrier<PIPE_ALL> → AscendC::SyncAll<false>()。
    //       SyncAll<false>() (isAIVOnly=false) 在 mix kernel 下:
    //         (1) 先做 V-V 全核同步 → 所有 AIV 的 MTE3 写出可见
    //         (2) 再做 C-C 全核同步 → 所有 AIC 互相对齐
    //         (3) 最后做 V↔C 同步 → AIC 知道 AIV 已完成
    //       AIV 与 AIC 两侧必须配对调用 SyncAll<false>(),次数和参数完全一致。
    // -----------------------------------------------------------------------
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::ICachePreLoad(1);

        uint32_t m = params.problemShape.m();
        uint32_t n = params.problemShape.n();
        uint32_t k = params.problemShape.k();

        // P2-A: tile 尺寸全部从 BlockQuant 派生
        QuantAllScheduler scheduler(params.batchCount, m, n, k);
        uint32_t totalQuantTasks = scheduler.GetTaskCount();

        // AIV worker 编号: 详见 design.md §4.2 末尾的说明
        //   GetBlockIdx()  → 全局 AIV subblock 序号  (与旧 BlockEpilogue 用法一致)
        //   GetBlockNum() * GetSubBlockNum() → 全部 AIV subblock 数
        uint32_t aivWorkerId = AscendC::GetBlockIdx();
        uint32_t aivWorkerNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

        Arch::Resource<MmadArchTag> resource;
        BlockQuant blockQuant(resource);

        AscendC::GlobalTensor<ElementInput> gmInputA;
        gmInputA.SetGlobalBuffer((__gm__ ElementInput *)params.ptrInputA);
        AscendC::GlobalTensor<ElementInput> gmInputB;
        gmInputB.SetGlobalBuffer((__gm__ ElementInput *)params.ptrInputB);

        AscendC::GlobalTensor<uint8_t> gmQuantA;
        gmQuantA.SetGlobalBuffer((__gm__ uint8_t *)params.ptrQuantA);
        AscendC::GlobalTensor<uint8_t> gmQuantB;
        gmQuantB.SetGlobalBuffer((__gm__ uint8_t *)params.ptrQuantB);

        AscendC::GlobalTensor<ElementScale1> gmScaleA1;
        gmScaleA1.SetGlobalBuffer((__gm__ ElementScale1 *)params.ptrScaleA1);
        AscendC::GlobalTensor<ElementScale2> gmScaleA2;
        gmScaleA2.SetGlobalBuffer((__gm__ ElementScale2 *)params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementScale1> gmScaleB1;
        gmScaleB1.SetGlobalBuffer((__gm__ ElementScale1 *)params.ptrScaleB1);
        AscendC::GlobalTensor<ElementScale2> gmScaleB2;
        gmScaleB2.SetGlobalBuffer((__gm__ ElementScale2 *)params.ptrMxScaleB);

        // ---- Round-robin 量化任务 ----
        for (uint32_t task = aivWorkerId; task < totalQuantTasks; task += aivWorkerNum) {
            QuantTask t = scheduler.GetTask(task);

            uint32_t rowStart = t.rowTile * QUANT_TILE_ROWS;
            uint32_t kStart = t.kTile * QUANT_TILE_K;

            if (t.isA) {
                uint32_t actualRows = (rowStart + QUANT_TILE_ROWS <= m) ? QUANT_TILE_ROWS : (m - rowStart);
                uint32_t actualK = (kStart + QUANT_TILE_K <= k) ? QUANT_TILE_K : (k - kStart);
                MatrixCoord absOffset{rowStart, kStart};
                MatrixCoord actualShape{actualRows, actualK};

                // Input GM tile (P2-B: 走 layout API,不手写 stride)
                int64_t inputOff = t.batchIdx * params.strideInputA + params.layoutInputA.GetOffset(absOffset);
                auto gmInputTile = gmInputA[inputOff];
                auto layoutInputTile = params.layoutInputA.GetTileLayout(actualShape);

                // FP4 output byte tile
                int64_t outputByteOff = t.batchIdx * params.strideQuantABytes +
                                        BlockQuant::GetPackedFp4ByteOffset(params.layoutOutputA, absOffset);
                auto gmOutputTile = gmQuantA[outputByteOff];
                auto layoutOutputTile = BlockQuant::MakePackedFp4ByteGmLayout(params.layoutOutputA, actualShape);

                // Scale1 tile [rows, ceil(actualK/512)]
                MatrixCoord s1AbsOffset{rowStart, kStart / LEVEL0_BLOCK_SIZE};
                MatrixCoord s1ActualShape{actualRows, CeilDiv<LEVEL0_BLOCK_SIZE>(actualK)};
                int64_t s1Off = t.batchIdx * params.strideScaleA1 + params.layoutScaleA1.GetOffset(s1AbsOffset);
                auto gmScale1Tile = gmScaleA1[s1Off];
                auto layoutScale1Tile = params.layoutScaleA1.GetTileLayout(s1ActualShape);

                // Scale2 tile [rows, round_up(ceil(actualK/32), 2)].
                MatrixCoord s2AbsOffset{rowStart, kStart / LEVEL1_BLOCK_SIZE};
                uint32_t s2Cols = RoundUp<2>(CeilDiv<LEVEL1_BLOCK_SIZE>(actualK));
                MatrixCoord s2ActualShape{actualRows, s2Cols};
                int64_t s2Off = t.batchIdx * params.strideMxScaleA + params.layoutScaleA2.GetOffset(s2AbsOffset);
                auto gmScale2Tile = gmScaleA2[s2Off];
                auto layoutScale2Tile = params.layoutScaleA2.GetTileLayout(s2ActualShape);

                blockQuant.QuantizeTilePerRow(gmInputTile, layoutInputTile, gmOutputTile, layoutOutputTile,
                                              gmScale1Tile, layoutScale1Tile, gmScale2Tile, layoutScale2Tile,
                                              actualShape);
            } else {
                // B: 物理布局也是 RowMajor [N, K]。AIC 的 ColumnMajor [K, N] 视图与之 byte 等价。
                uint32_t actualRows = (rowStart + QUANT_TILE_ROWS <= n) ? QUANT_TILE_ROWS : (n - rowStart);
                uint32_t actualK = (kStart + QUANT_TILE_K <= k) ? QUANT_TILE_K : (k - kStart);
                MatrixCoord absOffset{rowStart, kStart};
                MatrixCoord actualShape{actualRows, actualK};

                int64_t inputOff = t.batchIdx * params.strideInputB + params.layoutInputB.GetOffset(absOffset);
                auto gmInputTile = gmInputB[inputOff];
                auto layoutInputTile = params.layoutInputB.GetTileLayout(actualShape);

                int64_t outputByteOff = t.batchIdx * params.strideQuantBBytes +
                                        BlockQuant::GetPackedFp4ByteOffset(params.layoutOutputB, absOffset);
                auto gmOutputTile = gmQuantB[outputByteOff];
                auto layoutOutputTile = BlockQuant::MakePackedFp4ByteGmLayout(params.layoutOutputB, actualShape);

                MatrixCoord s1AbsOffset{rowStart, kStart / LEVEL0_BLOCK_SIZE};
                MatrixCoord s1ActualShape{actualRows, CeilDiv<LEVEL0_BLOCK_SIZE>(actualK)};
                int64_t s1Off = t.batchIdx * params.strideScaleB1 + params.layoutScaleB1.GetOffset(s1AbsOffset);
                auto gmScale1Tile = gmScaleB1[s1Off];
                auto layoutScale1Tile = params.layoutScaleB1.GetTileLayout(s1ActualShape);

                MatrixCoord s2AbsOffset{rowStart, kStart / LEVEL1_BLOCK_SIZE};
                uint32_t s2Cols = RoundUp<2>(CeilDiv<LEVEL1_BLOCK_SIZE>(actualK));
                MatrixCoord s2ActualShape{actualRows, s2Cols};
                int64_t s2Off = t.batchIdx * params.strideMxScaleB + params.layoutScaleB2.GetOffset(s2AbsOffset);
                auto gmScale2Tile = gmScaleB2[s2Off];
                auto layoutScale2Tile = params.layoutScaleB2.GetTileLayout(s2ActualShape);

                blockQuant.QuantizeTilePerRow(gmInputTile, layoutInputTile, gmOutputTile, layoutOutputTile,
                                              gmScale1Tile, layoutScale1Tile, gmScale2Tile, layoutScale2Tile,
                                              actualShape);
            }
        }

        // ---- V↔C 全核同步 (必须与 AIC 侧 SyncAll<false>() 配对) ----
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll<false>();
    }

    // -----------------------------------------------------------------------
    // AIC path: 与 AIV 配对的 SyncAll<false>(),然后做 MX FP4 matmul
    //
    // 与旧 kernel 区别:
    //   1. 开头一次 SyncAll<false>() 等 AIV 全量 quant 完成,不再 per-tile callback
    //   2. blockMmad 调用无 callbackBeforeFixpipe / callbackAfterFixpipe
    //   3. 空闲 AIC (例如 launch 24 个但 coreLoops < 24) 自然跳过 for 循环,无害退出
    //      但仍必须参与 SyncAll<false>(),否则 AIV 侧的 V↔C 同步会卡死
    // -----------------------------------------------------------------------
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        AscendC::SyncAll<false>();

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = params.batchCount * matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<MmadArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrQuantA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrQuantB);
        AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
        gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA *)params.ptrMxScaleA);
        AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
        gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB *)params.ptrMxScaleB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        if (CeilDiv(params.problemShape.m(), L1_TILE_M) == 1) {
            gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (CeilDiv(params.problemShape.n(), L1_TILE_N) == 1) {
            gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = matmulBlockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            int64_t batchOffsetA = batchIdx * params.strideQuantA;
            int64_t batchOffsetB = batchIdx * params.strideQuantB;
            int64_t batchOffsetC = batchIdx * params.strideC;
            int64_t batchOffsetMxScaleA = batchIdx * params.strideMxScaleA;
            int64_t batchOffsetMxScaleB = batchIdx * params.strideMxScaleB;

            auto tensorA = tla::MakeTensor(gmA[batchOffsetA], params.layoutQuantA, Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(gmB[batchOffsetB], params.layoutQuantB, Arch::PositionGM{});
            auto tensorMxScaleA =
                tla::MakeTensor(gmMxScaleA[batchOffsetMxScaleA], params.layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB =
                tla::MakeTensor(gmMxScaleB[batchOffsetMxScaleB], params.layoutMxScaleB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC[batchOffsetC], params.layoutC, Arch::PositionGM{});

            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockMxScaleA =
                GetTile(tensorMxScaleA,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                        tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
            auto tensorBlockMxScaleB =
                GetTile(tensorMxScaleB,
                        tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
            auto tensorBlockC = GetTile(tensorC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape, tensorBlockMxScaleA,
                      tensorBlockMxScaleB);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // -----------------------------------------------------------------------
    // QuantAllScheduler
    //
    // 任务编码: (operand, batch, rowTile, kTile)
    //   - operand:  A | B
    //   - batch:    [0, batchCount)
    //   - rowTile:  [0, ceil(M/SUB_TILE_M)) for A;  [0, ceil(N/SUB_TILE_M)) for B
    //   - kTile:    [0, ceil(K/SUB_TILE_K))
    //
    // 任务排布(单 batch 内): 先排完该 batch 的全部 A,再排完该 batch 的全部 B,再到下个 batch。
    // -----------------------------------------------------------------------
    struct QuantTask {
        bool isA;
        uint32_t batchIdx;
        uint32_t rowTile;
        uint32_t kTile;
    };

    struct QuantAllScheduler {
        uint32_t batchCount;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t aRowTiles;
        uint32_t bRowTiles;
        uint32_t kTiles;

        CATLASS_DEVICE
        QuantAllScheduler(uint32_t batchCount_, uint32_t m_, uint32_t n_, uint32_t k_)
            : batchCount(batchCount_), m(m_), n(n_), k(k_)
        {
            aRowTiles = CeilDiv<QUANT_TILE_ROWS>(m);
            bRowTiles = CeilDiv<QUANT_TILE_ROWS>(n);
            kTiles = CeilDiv<QUANT_TILE_K>(k);
        }

        CATLASS_DEVICE
        uint32_t GetTaskCount() const
        {
            return batchCount * (aRowTiles + bRowTiles) * kTiles;
        }

        CATLASS_DEVICE
        QuantTask GetTask(uint32_t taskIdx) const
        {
            uint32_t tasksPerBatchA = aRowTiles * kTiles;
            uint32_t tasksPerBatchB = bRowTiles * kTiles;
            uint32_t tasksPerBatch = tasksPerBatchA + tasksPerBatchB;

            uint32_t batchIdx = taskIdx / tasksPerBatch;
            uint32_t inner = taskIdx - batchIdx * tasksPerBatch;

            if (inner < tasksPerBatchA) {
                return QuantTask{true, batchIdx, inner / kTiles, inner % kTiles};
            }
            uint32_t bInner = inner - tasksPerBatchA;
            return QuantTask{false, batchIdx, bInner / kTiles, bInner % kTiles};
        }
    };
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_DUAL_LEVEL_QUANT_MX_BATCHED_MATMUL_TLA_HPP
