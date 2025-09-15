/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_FP8_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_FP8_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm/block/block_dequant.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t mScalar, uint32_t nScalar,
    uint32_t splitkLength>
class FP8Matmul {
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

    using BlockScheduler = BlockScheduler_;

    static const uint32_t COMPUTE_LENGTH_A = 16 * 1024 / sizeof(int8_t);
    using PrologueA = Block::DequantFP8toFP16<ArchTag, int8_t, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 16 * 1024 / sizeof(int8_t);
    using PrologueB = Block::DequantFP8toFP16<ArchTag, int8_t, LayoutB, COMPUTE_LENGTH_B>;

    using Cast = Block::DequantFP8toFP16<ArchTag, int8_t, LayoutB, COMPUTE_LENGTH_B>;

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
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrWC;
        half scalar;
        half zeroPoint;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrWA_, GM_ADDR ptrWB_, GM_ADDR ptrWC_, half scalar_,
            half zeroPoint_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_), ptrC(ptrC_),
              layoutC(layoutC_), ptrWA(ptrWA_), ptrWB(ptrWB_), ptrWC(ptrWC_), scalar(scalar_), zeroPoint(zeroPoint_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrWC;
        half scalar;
        half zeroPoint;
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
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
        Params params{args.problemShape,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            args.ptrWA,
            args.ptrWB,
            args.ptrWC,
            args.scalar,
            args.zeroPoint};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    FP8Matmul()
    {
        flag0[0].id = flagID0;
        flag0[1].id = flagID1;
        flag1[0].id = flagID2;
        flag1[1].id = flagID3;
    }

    /// Executes one GEMM
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    __attribute__((always_inline)) void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler blockScheduler(
            params.problemShape, MakeCoord((L1TileShape::M * mScalar), (L1TileShape::N * nScalar)));
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        AscendC::GlobalTensor<int8_t> gmA;
        gmA.SetGlobalBuffer((__gm__ int8_t *)params.ptrA);
        AscendC::GlobalTensor<half> gmWA;
        gmWA.SetGlobalBuffer((__gm__ half *)params.ptrWA);

        AscendC::GlobalTensor<int8_t> gmB;
        gmB.SetGlobalBuffer((__gm__ int8_t *)params.ptrB);
        AscendC::GlobalTensor<half> gmWB;
        gmWB.SetGlobalBuffer((__gm__ half *)params.ptrWB);

        AscendC::GlobalTensor<half> gmC;
        gmC.SetGlobalBuffer((__gm__ half *)params.ptrC);
        AscendC::GlobalTensor<float> gmWC;
        gmWC.SetGlobalBuffer((__gm__ float *)params.ptrWC);

        uint32_t srcAStride = params.problemShape.k();
        uint32_t srcBStride = params.problemShape.n();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        for (uint32_t loopIdx = AscendC::GetBlockIdx() / AIVPERCORE; loopIdx < coreLoops;
             loopIdx += AscendC::GetBlockNum()) {  // 一次for循环完成两个行块或者两个列块的反量化
            // 当前任务块信息
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
            MatrixCoord offsetA{blockCoord.m() * (L1TileShape::M * mScalar), 0};
            MatrixCoord offsetB{0, blockCoord.n() * (L1TileShape::N * nScalar)};
            MatrixCoord offsetC{
                blockCoord.m() * (L1TileShape::M * mScalar), blockCoord.n() * (L1TileShape::N * nScalar)};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            // 下一个任务块的信息
            bool isFirstBlock = (loopIdx == (AscendC::GetBlockIdx() / AIVPERCORE));
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = blockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = blockScheduler.GetActualBlockShape(nextBlockCoord);
            }
            MatrixCoord offsetNextA{nextBlockCoord.m() * (L1TileShape::M * mScalar), 0};
            MatrixCoord offsetNextB{0, nextBlockCoord.n() * (L1TileShape::N * nScalar)};
            MatrixCoord offsetNextC{
                nextBlockCoord.m() * (L1TileShape::M * mScalar), nextBlockCoord.n() * (L1TileShape::N * nScalar)};
            int64_t gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
            int64_t gmOffsetNextC = params.layoutC.GetOffset(offsetNextC);

            Arch::Resource<ArchTag> resource;
            uint32_t kLoop = (params.problemShape.k() + splitkLength - 1) / splitkLength;
            for (uint32_t ldk = 0; ldk < kLoop; ldk++) {  // 一次for循环完成切K后的一个行块/列块

                // 反量化后的workspace索引
                int64_t gmOffsetWA =
                    (AscendC::GetBlockIdx() / AIVPERCORE) * (mScalar * L1TileShape::M) * splitkLength * STAGES +
                    (mScalar * L1TileShape::M) * splitkLength * crossCoreBufferIndexAIV;
                int64_t gmOffsetWB =
                    (AscendC::GetBlockIdx() / AIVPERCORE) * splitkLength * (nScalar * L1TileShape::N) * STAGES +
                    splitkLength * (nScalar * L1TileShape::N) * crossCoreBufferIndexAIV;
                int64_t gmOffsetNextWA =
                    (AscendC::GetBlockIdx() / AIVPERCORE) * (mScalar * L1TileShape::M) * splitkLength * STAGES +
                    (mScalar * L1TileShape::M) * splitkLength * (1 - crossCoreBufferIndexAIV);
                int64_t gmOffsetNextWB =
                    (AscendC::GetBlockIdx() / AIVPERCORE) * splitkLength * (nScalar * L1TileShape::N) * STAGES +
                    splitkLength * (nScalar * L1TileShape::N) * (1 - crossCoreBufferIndexAIV);

                uint32_t kActual = (params.problemShape.k() < (ldk + 1) * splitkLength)
                                       ? params.problemShape.k() % splitkLength
                                       : splitkLength;
                uint32_t kActualAligned = (kActual + 256 - 1) / 256 * 256;

                LayoutA layoutWA(actualBlockShape.m(), kActual, kActualAligned);
                LayoutB layoutWB(kActual, actualBlockShape.n(), actualBlockShape.n());

                if (ldk == 0 && isFirstBlock) {  // 第一个任务块的第一个K切块
                    Catlass::Arch::CrossCoreWaitFlag(flag0[crossCoreBufferIndexAIV]);
                    if (std::is_same_v<LayoutA, Catlass::layout::RowMajor>) {  // A行优先
                        PrologueA prologueA(resource);
                        prologueA(gmA[gmOffsetA],
                            gmWA[gmOffsetWA],
                            layoutWA,
                            srcAStride,
                            kActualAligned,
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    } else {  // A列优先
                        srcAStride = params.problemShape.m();
                        PrologueA prologueA(resource);
                        prologueA(gmA[gmOffsetA],
                            gmWA[gmOffsetWA],
                            layoutWA,
                            srcAStride,
                            actualBlockShape.m(),
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    }
                    if (std::is_same_v<LayoutB, Catlass::layout::RowMajor>) {  // B行优先
                        PrologueB prologueB(resource);
                        prologueB(gmB[gmOffsetB],
                            gmWB[gmOffsetWB],
                            layoutWB,
                            srcBStride,
                            actualBlockShape.n(),
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    } else {  // B列优先
                        srcBStride = params.problemShape.k();
                        PrologueB prologueB(resource);
                        prologueB(gmB[gmOffsetB],
                            gmWB[gmOffsetWB],
                            layoutWB,
                            srcBStride,
                            kActualAligned,
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    }
                }
                if (ldk < kLoop - 1) {  // 后续块
                    uint32_t kActualNext = (params.problemShape.k() < (ldk + 2) * splitkLength)
                                               ? params.problemShape.k() % splitkLength
                                               : splitkLength;
                    uint32_t kActualNextAligned = (kActualNext + 256 - 1) / 256 * 256;

                    LayoutA layoutNextWA(actualBlockShape.m(), kActualNext, kActualNextAligned);
                    LayoutB layoutNextWB(kActualNext, actualBlockShape.n(), actualBlockShape.n());

                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag1[crossCoreBufferIndexAIV]);
                    Catlass::Arch::CrossCoreWaitFlag(flag0[1 - crossCoreBufferIndexAIV]);
                    if (std::is_same_v<LayoutA, Catlass::layout::RowMajor>) {  // A行优先
                        PrologueA prologueA(resource);
                        gmOffsetA += kActual;
                        prologueA(gmA[gmOffsetA],
                            gmWA[gmOffsetNextWA],
                            layoutNextWA,
                            srcAStride,
                            kActualNextAligned,
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    } else {  // A列优先
                        srcAStride = params.problemShape.m();
                        PrologueA prologueA(resource);
                        gmOffsetA += kActual * params.problemShape.m();
                        prologueA(gmA[gmOffsetA],
                            gmWA[gmOffsetNextWA],
                            layoutNextWA,
                            srcAStride,
                            actualBlockShape.m(),
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    }
                    if (std::is_same_v<LayoutB, Catlass::layout::RowMajor>) {  // B行优先
                        PrologueB prologueB(resource);
                        gmOffsetB += kActual * params.problemShape.n();
                        prologueB(gmB[gmOffsetB],
                            gmWB[gmOffsetNextWB],
                            layoutNextWB,
                            srcBStride,
                            actualBlockShape.n(),
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    } else {  // B列优先
                        srcBStride = params.problemShape.k();
                        PrologueB prologueB(resource);
                        gmOffsetB += kActual;
                        prologueB(gmB[gmOffsetB],
                            gmWB[gmOffsetNextWB],
                            layoutNextWB,
                            srcBStride,
                            kActualNextAligned,
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    }
                }
                if ((ldk == kLoop - 1) && hasNextBlock) {  // 当前切块为K方向最后一个切块且有下一个任务块
                    uint32_t kActualNext = (params.problemShape.k() < splitkLength)
                                               ? params.problemShape.k() % splitkLength
                                               : splitkLength;
                    uint32_t kActualNextAligned = (kActualNext + 256 - 1) / 256 * 256;

                    LayoutA layoutNextWA(nextActualBlockShape.m(), kActualNext, kActualNext);
                    LayoutB layoutNextWB(kActualNext, nextActualBlockShape.n(), nextActualBlockShape.n());

                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag1[crossCoreBufferIndexAIV]);
                    Catlass::Arch::CrossCoreWaitFlag(flag0[1 - crossCoreBufferIndexAIV]);
                    if (std::is_same_v<LayoutA, Catlass::layout::RowMajor>) {  // A行优先
                        PrologueA prologueA(resource);
                        prologueA(gmA[gmOffsetNextA],
                            gmWA[gmOffsetNextWA],
                            layoutNextWA,
                            srcAStride,
                            kActualNextAligned,
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    } else {  // A列优先
                        srcAStride = params.problemShape.m();
                        PrologueA prologueA(resource);
                        prologueA(gmA[gmOffsetNextA],
                            gmWA[gmOffsetNextWA],
                            layoutNextWA,
                            srcAStride,
                            nextActualBlockShape.m(),
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    }
                    if (std::is_same_v<LayoutB, Catlass::layout::RowMajor>) {  // B行优先
                        PrologueB prologueB(resource);
                        prologueB(gmB[gmOffsetNextB],
                            gmWB[gmOffsetNextWB],
                            layoutNextWB,
                            srcBStride,
                            nextActualBlockShape.n(),
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    } else {  // B列优先
                        srcBStride = params.problemShape.k();
                        PrologueB prologueB(resource);
                        prologueB(gmB[gmOffsetNextB],
                            gmWB[gmOffsetNextWB],
                            layoutNextWB,
                            srcBStride,
                            kActualNextAligned,
                            params.scalar,
                            params.zeroPoint,
                            bufferIndex);
                    }

                    Catlass::Arch::CrossCoreWaitFlag(flag4);
                    Catlass::layout::RowMajor layoutBlockC(
                        actualBlockShape.m(), actualBlockShape.n(), params.problemShape.n());
                    int64_t gmOffsetWC =
                        (AscendC::GetBlockIdx() / AIVPERCORE) * (mScalar * L1TileShape::M) * (nScalar * L1TileShape::N);
                    Cast cast;
                    cast.castFP32toFP16(gmWC[gmOffsetWC],
                        gmC[gmOffsetC],
                        layoutBlockC,
                        nScalar * L1TileShape::N,
                        params.problemShape.n());
                }
                if ((ldk == kLoop - 1) && (!hasNextBlock)) {  // 切块为K方向最后一个切块且没有下一个任务块
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag1[crossCoreBufferIndexAIV]);
                    Catlass::Arch::CrossCoreWaitFlag(flag4);
                    Catlass::layout::RowMajor layoutBlockC(
                        actualBlockShape.m(), actualBlockShape.n(), params.problemShape.n());
                    int64_t gmOffsetWC =
                        (AscendC::GetBlockIdx() / AIVPERCORE) * (mScalar * L1TileShape::M) * (nScalar * L1TileShape::N);
                    Cast cast;
                    cast.castFP32toFP16(gmWC[gmOffsetWC],
                        gmC[gmOffsetC],
                        layoutBlockC,
                        nScalar * L1TileShape::N,
                        params.problemShape.n());
                }
                crossCoreBufferIndexAIV = 1 - crossCoreBufferIndexAIV;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);

        Catlass::Arch::CrossCoreWaitFlag(flag0[0]);
        Catlass::Arch::CrossCoreWaitFlag(flag0[1]);
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler(
            params.problemShape, MakeCoord((L1TileShape::M * mScalar), (L1TileShape::N * nScalar)));
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<half> gmWA;
        gmWA.SetGlobalBuffer((__gm__ half *)params.ptrWA);
        AscendC::GlobalTensor<half> gmWB;
        gmWB.SetGlobalBuffer((__gm__ half *)params.ptrWB);
        AscendC::GlobalTensor<float> gmWC;
        gmWC.SetGlobalBuffer((__gm__ float *)params.ptrWC);

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag0[0]);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag0[1]);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
             loopIdx += AscendC::GetBlockNum()) {  // 一次for循环完成一个大基本结果块(256,512)
            // Compute block location
            // 获取当前大基本结果块的左上角坐标以及实际大小
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBigBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

            uint32_t kLoop = (params.problemShape.k() + splitkLength - 1) / splitkLength;
            for (uint32_t ldk = 0; ldk < kLoop; ldk++) {  // 一次for循环完成切K后的一个行块/列块
                bool isFirstKSlice = (ldk == 0) ? true : false;
                uint32_t kActual = (params.problemShape.k() < (ldk + 1) * splitkLength)
                                       ? params.problemShape.k() % splitkLength
                                       : splitkLength;
                uint32_t kActualAligned = (kActual + 256 - 1) / 256 * 256;

                uint32_t mLoop = (actualBigBlockShape.m() + L1TileShape::M - 1) / L1TileShape::M;
                uint32_t nLoop = (actualBigBlockShape.n() + L1TileShape::N - 1) / L1TileShape::N;
                Catlass::Arch::CrossCoreWaitFlag(flag1[crossCoreBufferIndexAIC]);
                for (uint32_t processIdm = 0; processIdm < mLoop; processIdm++) {
                    for (uint32_t processIdn = 0; processIdn < nLoop; processIdn++) {
                        bool hasNextBlock = ((processIdm * nLoop + processIdn) < mLoop * nLoop - 1) ? true : false;
                        bool isFirstBlock = (processIdm == 0 && processIdn == 0) ? true : false;
                        uint32_t processIdxNext = processIdm * nLoop + processIdn + 1;
                        uint32_t processIdmNext = processIdxNext / nLoop;
                        uint32_t processIdnNext = processIdxNext % nLoop;
                        // Compute initial location in logical coordinates
                        MatrixCoord offsetBlockC{
                            blockCoord.m() * (L1TileShape::M * mScalar) + L1TileShape::M * processIdm,
                            blockCoord.n() * (L1TileShape::N * nScalar) + L1TileShape::N * processIdn};

                        uint32_t mActual = L1TileShape::M;
                        uint32_t nActual = L1TileShape::N;
                        if (actualBigBlockShape.m() % L1TileShape::M != 0 && processIdm == mLoop - 1) {
                            mActual = actualBigBlockShape.m() % L1TileShape::M;
                        }
                        if (actualBigBlockShape.n() % L1TileShape::N != 0 && processIdn == nLoop - 1) {
                            nActual = actualBigBlockShape.n() % L1TileShape::N;
                        }
                        GemmCoord actualSmallBlockShape(mActual, nActual, kActual);

                        uint32_t mActualNext = L1TileShape::M;
                        uint32_t nActualNext = L1TileShape::N;
                        if (actualBigBlockShape.m() % L1TileShape::M != 0 && processIdmNext == mLoop - 1) {
                            mActualNext = actualBigBlockShape.m() % L1TileShape::M;
                        }
                        if (actualBigBlockShape.n() % L1TileShape::N != 0 && processIdnNext == nLoop - 1) {
                            nActualNext = actualBigBlockShape.n() % L1TileShape::N;
                        }
                        GemmCoord nextSmallBlockShape(mActualNext, nActualNext, kActual);

                        // 当前块的地址偏移
                        int64_t gmOffsetWA =
                            AscendC::GetBlockIdx() * (L1TileShape::M * mScalar) * splitkLength * STAGES +
                            (L1TileShape::M * mScalar) * splitkLength * crossCoreBufferIndexAIC +
                            processIdm * L1TileShape::M * kActualAligned;
                        int64_t gmOffsetWB =
                            AscendC::GetBlockIdx() * splitkLength * (L1TileShape::N * nScalar) * STAGES +
                            splitkLength * (L1TileShape::N * nScalar) * crossCoreBufferIndexAIC +
                            processIdn * L1TileShape::N;
                        int64_t gmOffsetWC =
                            AscendC::GetBlockIdx() * (L1TileShape::M * mScalar) * (L1TileShape::N * nScalar) +
                            processIdm * L1TileShape::M * (L1TileShape::N * nScalar) + processIdn * L1TileShape::N;

                        uint32_t AStride = kActualAligned;
                        uint32_t BStride = actualBigBlockShape.n();
                        if (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>) {  // A列优先
                            gmOffsetWA = AscendC::GetBlockIdx() * (L1TileShape::M * mScalar) * splitkLength * STAGES +
                                         (L1TileShape::M * mScalar) * splitkLength * crossCoreBufferIndexAIC +
                                         processIdm * L1TileShape::M;
                            AStride = actualBigBlockShape.m();
                        }
                        if (std::is_same_v<LayoutB, Catlass::layout::ColumnMajor>) {  // B列优先
                            gmOffsetWB = AscendC::GetBlockIdx() * splitkLength * (L1TileShape::N * nScalar) * STAGES +
                                         splitkLength * (L1TileShape::N * nScalar) * crossCoreBufferIndexAIC +
                                         processIdn * L1TileShape::N * kActualAligned;
                            BStride = kActualAligned;
                        }

                        // 下一个块的地址偏移
                        int64_t gmOffsetWANext =
                            AscendC::GetBlockIdx() * (L1TileShape::M * mScalar) * splitkLength * STAGES +
                            (L1TileShape::M * mScalar) * splitkLength * crossCoreBufferIndexAIC +
                            processIdmNext * L1TileShape::M * kActualAligned;
                        int64_t gmOffsetWBNext =
                            AscendC::GetBlockIdx() * splitkLength * (L1TileShape::N * nScalar) * STAGES +
                            splitkLength * (L1TileShape::N * nScalar) * crossCoreBufferIndexAIC +
                            processIdnNext * L1TileShape::N;
                        int64_t gmOffsetWCNext =
                            AscendC::GetBlockIdx() * (L1TileShape::M * mScalar) * (L1TileShape::N * nScalar) +
                            processIdmNext * L1TileShape::M * (L1TileShape::N * nScalar) +
                            processIdnNext * L1TileShape::N;

                        if (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>) {  // A列优先
                            gmOffsetWANext =
                                AscendC::GetBlockIdx() * (L1TileShape::M * mScalar) * splitkLength * STAGES +
                                (L1TileShape::M * mScalar) * splitkLength * crossCoreBufferIndexAIC +
                                processIdmNext * L1TileShape::M;
                            AStride = actualBigBlockShape.m();
                        }
                        if (std::is_same_v<LayoutB, Catlass::layout::ColumnMajor>) {  // B列优先
                            gmOffsetWBNext =
                                AscendC::GetBlockIdx() * splitkLength * (L1TileShape::N * nScalar) * STAGES +
                                splitkLength * (L1TileShape::N * nScalar) * crossCoreBufferIndexAIC +
                                processIdnNext * L1TileShape::N * kActualAligned;
                            BStride = kActualAligned;
                        }

                        LayoutA layoutWA(mActual, kActual, AStride);
                        LayoutB layoutWB(kActual, nActual, BStride);
                        LayoutC layoutWC(mActual, nActual, nScalar * L1TileShape::N);

                        LayoutA layoutWANext(mActualNext, kActual, AStride);
                        LayoutB layoutWBNext(kActual, nActualNext, BStride);
                        LayoutC layoutWCNext(mActualNext, nActualNext, nScalar * L1TileShape::N);

                        // 完成一个128 * 256的小结果矩阵基本块的运算
                        blockMmad(gmWA[gmOffsetWA],
                            layoutWA,
                            gmWB[gmOffsetWB],
                            layoutWB,
                            gmWC[gmOffsetWC],
                            layoutWC,
                            gmWA[gmOffsetWANext],
                            layoutWANext,
                            gmWB[gmOffsetWBNext],
                            layoutWBNext,
                            actualSmallBlockShape,
                            nextSmallBlockShape,
                            isFirstKSlice,
                            isFirstBlock,
                            hasNextBlock);
                    }
                }
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag0[crossCoreBufferIndexAIC]);
                crossCoreBufferIndexAIC = 1 - crossCoreBufferIndexAIC;
                if (ldk == kLoop - 1) {
                    // cast 256 * 512的fp32大结果基本块为fp16
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag4);
                }
            }
        }
    }

protected:
    static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t AIVPERCORE = 2;
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    static constexpr Arch::FlagID flagID0 = 0;
    static constexpr Arch::FlagID flagID1 = 1;
    static constexpr Arch::FlagID flagID2 = 2;
    static constexpr Arch::FlagID flagID3 = 3;
    static constexpr Arch::FlagID flagID4 = 4;

    Arch::CrossCoreFlag flag0[STAGES];
    Arch::CrossCoreFlag flag1[STAGES];
    Arch::CrossCoreFlag flag4{flagID4};

    uint32_t crossCoreBufferIndexAIC{0};
    uint32_t crossCoreBufferIndexAIV{0};
    uint32_t bufferIndex{0};
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_FP8_MATMUL_HPP