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

#ifndef CATLASS_GEMM_KERNEL_BROADCAST_MATMUL_PERBLOCK_QUANT_TLA_HPP
#define CATLASS_GEMM_KERNEL_BROADCAST_MATMUL_PERBLOCK_QUANT_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/detail/tag_to_layout.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/epilogue/block/block_epilogue.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Gemm::Kernel {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using namespace AscendC::MicroAPI;
#endif

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BroadcastMatmulPerblockQuantTla {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementDst = typename BlockEpilogue::ElementDst;
    using LayoutDst = typename BlockEpilogue::LayoutDst;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = detail::TagToLayout_t<float, layout::VectorLayout>;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    static_assert(std::is_same_v<LayoutA, detail::TagToLayout_t<ElementA, layout::RowMajor>>,
        "LayoutA must be RowMajor");
    static_assert(std::is_same_v<LayoutB, detail::TagToLayout_t<ElementB, layout::RowMajor>>,
        "LayoutB must be RowMajor");
    static_assert(std::is_same_v<LayoutC, detail::TagToLayout_t<ElementC, layout::RowMajor>>,
        "LayoutC must be RowMajor");
    static_assert(std::is_same_v<LayoutDst, detail::TagToLayout_t<ElementDst, layout::RowMajor>>,
        "LayoutDst must be RowMajor");
    static_assert(std::is_same_v<ElementA, bfloat16_t> && std::is_same_v<ElementB, bfloat16_t> &&
        std::is_same_v<ElementC, bfloat16_t>, "ElementA, ElementB, ElementC must be bfloat16_t");

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t FLOAT_ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(float);

    /// Parameters structure
    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        int64_t strideA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        int64_t strideB;
        LayoutC layoutC;
        int64_t strideC;
        GM_ADDR ptrDst;
        LayoutDst layoutDst;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;

        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(uint32_t batchCount_, GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, int64_t strideA_,
               GM_ADDR ptrB_, LayoutB layoutB_, int64_t strideB_,
               LayoutC layoutC_, int64_t strideC_, GM_ADDR ptrDst_, LayoutDst layoutDst_,
               GM_ADDR ptrScale_, LayoutScale layoutScale_)
            : batchCount(batchCount_), problemShape(problemShape_),
              ptrA(ptrA_), layoutA(layoutA_), strideA(strideA_),
              ptrB(ptrB_), layoutB(layoutB_), strideB(strideB_),
              layoutC(layoutC_), strideC(strideC_), ptrDst(ptrDst_), layoutDst(layoutDst_),
              ptrScale(ptrScale_), layoutScale(layoutScale_) {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrA; LayoutA layoutA;
        GM_ADDR ptrB; LayoutB layoutB;
        LayoutC layoutC;
        GM_ADDR ptrDst;
        GM_ADDR ptrScale;
    };

    static bool CanImplement(const Arguments &args)
    {
        if (args.batchCount <= 0 || args.batchCount > 65536) {
            return false;
        }
        if (args.problemShape.m() != 128 && args.problemShape.m() != 256) {
            return false;
        }
        if (args.problemShape.n() != 128 || args.problemShape.k() != 128) {
            return false;
        }
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        int64_t strideA = m * k;
        int64_t strideB = k * n;
        int64_t strideC = m * n;
        LayoutDst layoutDst = tla::MakeLayout<uint8_t, layout::RowMajor>(m, n);
        LayoutScale layoutScale = tla::MakeLayout(args.batchCount * FLOAT_ELE_NUM_PER_BLK);
        Params params{args.batchCount,
            problemShape,
            args.ptrA,
            args.layoutA,
            strideA,
            args.ptrB,
            args.layoutB,
            strideB,
            args.layoutC,
            strideC,
            args.ptrDst,
            layoutDst,
            args.ptrScale,
            layoutScale};
        return params;
    }

    CATLASS_DEVICE
    BroadcastMatmulPerblockQuantTla() {
#ifdef __DAV_VEC__
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
#endif
    }

    CATLASS_DEVICE
    ~BroadcastMatmulPerblockQuantTla() {
#ifdef __DAV_CUBE__
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
#endif
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one BatchedMatmulPerblockQuant
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::LocalTensor<ElementC> ubC;
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(0);

        for (uint32_t batchIdx = AscendC::GetBlockIdx(); batchIdx < params.batchCount; batchIdx += AscendC::GetBlockNum()) {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx += 1) {
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                int64_t batchOffsetA = batchIdx * params.strideA;

                auto tensorA = tla::MakeTensor(gmA[batchOffsetA], params.layoutA, Arch::PositionGM{});
                auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});
                auto tensorUbC = tla::MakeTensor(ubC, params.layoutC, Arch::PositionUB{});

                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
                );
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
                );
                auto tensorBlockC = GetTile(
                    tensorUbC, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
                );

                blockMmad(tensorBlockA, tensorBlockB, tensorBlockC, actualBlockShape);
            }
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {
        Arch::Resource<ArchTag> resource;

        typename BlockEpilogue::Params epilogueParams(
            params.layoutC,
            params.layoutDst,
            params.layoutScale,
            params.strideC
        );
        BlockEpilogue blockEpilogue(resource, epilogueParams);

        AscendC::GlobalTensor<ElementDst> gmDst;
        gmDst.SetGlobalBuffer((__gm__ ElementDst *)params.ptrDst);
        AscendC::LocalTensor<ElementC> ubC;
        AscendC::LocalTensor<ElementDst> ubDst;
        AscendC::LocalTensor<ElementScale> ubScale;
        size_t ubOffset = 0;
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += params.strideC * sizeof(ElementC);
        ubDst = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
        ubOffset += params.strideC * sizeof(ElementDst);
        ubScale = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);

        const uint32_t aiCoreNum = AscendC::GetBlockNum();
        const uint32_t aiCoreIdx = AscendC::GetBlockIdx() >> 1;

        for (uint32_t batchIdx = aiCoreIdx; batchIdx < params.batchCount; batchIdx += aiCoreNum) {
            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG);
            if (AscendC::GetSubBlockIdx() == 0) {
                const int64_t batchOffsetDst = batchIdx * params.strideC;
                const int64_t batchOffsetScale = batchIdx / aiCoreNum * BlockEpilogue::FLOAT_ELE_NUM_PER_BLK;
                blockEpilogue(ubC, ubDst, ubScale[batchOffsetScale], gmDst[batchOffsetDst]);
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            } else {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIV_SYNC_AIC_FLAG);
            }
        }
        if (AscendC::GetSubBlockIdx() == 0) {
            uint16_t blockCount = params.batchCount / aiCoreNum;
            if (aiCoreIdx < params.batchCount % aiCoreNum) {
                blockCount += 1;
            }
            if (blockCount > 0) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::GlobalTensor<float> gmScale;
                gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.ptrScale));
                AscendC::DataCopyExtParams dataCopyExtParams(blockCount, sizeof(float), 0, (aiCoreNum - 1) * sizeof(float), 0);
                AscendC::DataCopyPad(gmScale[aiCoreIdx], ubScale, dataCopyExtParams);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

private:
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint16_t FLAG_ID_MAX = 16;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BROADCAST_MATMUL_PERBLOCK_QUANT_TLA_HPP
