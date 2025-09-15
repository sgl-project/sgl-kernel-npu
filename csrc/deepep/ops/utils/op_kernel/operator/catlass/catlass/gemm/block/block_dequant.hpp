/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_DEQUANT_HPP
#define CATLASS_GEMM_BLOCK_DEQUANT_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

namespace Catlass::Gemm::Block {

template <class ArchTag_, class Element_, class LayoutIn_, uint32_t COMPUTE_LENGTH>
struct DequantFP8toFP16 {
public:
    using ArchTag = ArchTag_;
    using ElementIn = Element_;
    using LayoutIn = LayoutIn_;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<int8_t, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<half, Catlass::layout::RowMajor>>;

    using CopyGm2UbFP32 = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, Catlass::layout::RowMajor>>;
    using CopyUb2GmFP32 = Catlass::Epilogue::Tile::CopyUb2Gm<ArchTag, Gemm::GemmType<float, Catlass::layout::RowMajor>>;
    using LayoutC = Catlass::layout::RowMajor;

    static const uint32_t Alignment = 256;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;
    CopyGm2UbFP32 copyGm2UbFP32;
    CopyUb2GmFP32 copyUb2GmFP32;

    struct AiCoreInfo {
        uint32_t AivNum;
        uint32_t AivId;
    } aiCoreInfo;

    struct BlockLoopInfo {
        uint32_t m;
        uint32_t n;
        uint32_t aivId;
        uint32_t aivNum;
        uint64_t srcBlockOffset;
        uint64_t dstBlockOffset;
        uint32_t totalLoop;
        uint32_t nLoop;
        uint32_t taskPerAiv;
    };

    struct LoadStoreInfo {
        uint32_t loopIdx;
        uint32_t mIdx;
        uint32_t nIdx;
        uint64_t srcProcessOffset;
        uint64_t dstProcessOffset;
        // loader params
        uint32_t loadRepeat = 1;
        uint32_t loadLen;
        uint32_t srcLoadStride = 0;
        uint32_t dstLoadStride = 0;
        // storer params
        uint32_t storeRepeat = 1;
        uint32_t storeLen;
        uint32_t srcStoreStride = 0;
        uint32_t dstStoreStride;
    };

    CATLASS_DEVICE
    DequantFP8toFP16()
    {}

    CATLASS_DEVICE
    DequantFP8toFP16(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn));
            bufferOffset += COMPUTE_LENGTH;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            outputBuffer[i] = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                                  .template ReinterpretCast<half>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            workspace[i] = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                               .template ReinterpretCast<half>();
            bufferOffset += COMPUTE_LENGTH * 2;
        }
        int16_t value_uint = 0x4000;
        value_vector1 = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector1, value_uint, 128);
        pipe_barrier(PIPE_V);
        value_uint = 0x3FFF;
        value_vector2 = (resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset * sizeof(ElementIn)))
                            .template ReinterpretCast<int16_t>();
        bufferOffset += 256;
        AscendC::Duplicate<int16_t>(value_vector2, value_uint, 128);
        pipe_barrier(PIPE_V);
    }

    CATLASS_DEVICE
    void GetBlockLoopInfo(BlockLoopInfo &blockLoopInfo, uint32_t srcStride, uint32_t dstStride)
    {
        blockLoopInfo.taskPerAiv = blockLoopInfo.m / blockLoopInfo.aivNum;
        uint32_t taskRemain = blockLoopInfo.m % blockLoopInfo.aivNum;
        if (blockLoopInfo.aivId < taskRemain) {
            blockLoopInfo.taskPerAiv++;
        }

        uint32_t alignedN = RoundUp<Alignment, uint32_t>(blockLoopInfo.n);
        blockLoopInfo.srcBlockOffset = blockLoopInfo.aivId * blockLoopInfo.taskPerAiv * srcStride;
        blockLoopInfo.dstBlockOffset = blockLoopInfo.aivId * blockLoopInfo.taskPerAiv * dstStride;
        if (blockLoopInfo.aivId >= taskRemain) {
            blockLoopInfo.srcBlockOffset += taskRemain * srcStride;
            blockLoopInfo.dstBlockOffset += taskRemain * dstStride;
        }
        if (alignedN > COMPUTE_LENGTH / 2) {
            blockLoopInfo.nLoop = (blockLoopInfo.n + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            blockLoopInfo.totalLoop = blockLoopInfo.taskPerAiv * blockLoopInfo.nLoop;
        } else if (alignedN != 0) {
            blockLoopInfo.nLoop = COMPUTE_LENGTH / alignedN;
            blockLoopInfo.totalLoop = (blockLoopInfo.taskPerAiv + blockLoopInfo.nLoop - 1) / blockLoopInfo.nLoop;
        } else {
            blockLoopInfo.nLoop = 0;
            blockLoopInfo.totalLoop = 0;
        }
    }

    CATLASS_DEVICE
    void GetLoaderStorerInfo(
        BlockLoopInfo &blockLoopInfo, LoadStoreInfo &loadStoreInfo, uint32_t srcStride, uint32_t dstStride)
    {
        loadStoreInfo.loadLen = COMPUTE_LENGTH;
        uint32_t alignedN = RoundUp<Alignment, uint32_t>(blockLoopInfo.n);
        if (alignedN > COMPUTE_LENGTH / 2) {
            loadStoreInfo.mIdx = loadStoreInfo.loopIdx / blockLoopInfo.nLoop;
            loadStoreInfo.nIdx = loadStoreInfo.loopIdx % blockLoopInfo.nLoop;

            loadStoreInfo.srcProcessOffset =
                blockLoopInfo.srcBlockOffset + loadStoreInfo.mIdx * srcStride + loadStoreInfo.nIdx * COMPUTE_LENGTH;
            loadStoreInfo.dstProcessOffset =
                blockLoopInfo.dstBlockOffset + loadStoreInfo.mIdx * dstStride + loadStoreInfo.nIdx * COMPUTE_LENGTH;
            if ((loadStoreInfo.nIdx == blockLoopInfo.nLoop - 1) && (blockLoopInfo.n % COMPUTE_LENGTH != 0)) {
                loadStoreInfo.loadLen = blockLoopInfo.n % COMPUTE_LENGTH;
            }
            loadStoreInfo.storeLen = loadStoreInfo.loadLen;
        } else {
            loadStoreInfo.mIdx = loadStoreInfo.loopIdx * blockLoopInfo.nLoop;
            loadStoreInfo.srcProcessOffset = blockLoopInfo.srcBlockOffset + loadStoreInfo.mIdx * srcStride;
            loadStoreInfo.dstProcessOffset = blockLoopInfo.dstBlockOffset + loadStoreInfo.mIdx * dstStride;
            loadStoreInfo.loadLen = blockLoopInfo.n;
            loadStoreInfo.loadRepeat = blockLoopInfo.nLoop;
            loadStoreInfo.storeLen = blockLoopInfo.n;
            loadStoreInfo.storeRepeat = blockLoopInfo.nLoop;
            loadStoreInfo.dstStoreStride = dstStride;
            if ((loadStoreInfo.loopIdx == blockLoopInfo.totalLoop - 1) &&
                (blockLoopInfo.taskPerAiv % blockLoopInfo.nLoop != 0)) {
                loadStoreInfo.storeRepeat = blockLoopInfo.taskPerAiv % blockLoopInfo.nLoop;
                loadStoreInfo.loadRepeat = loadStoreInfo.storeRepeat;
            }
        }
        loadStoreInfo.srcLoadStride = srcStride;
        loadStoreInfo.dstLoadStride = alignedN;
        loadStoreInfo.srcStoreStride = alignedN;
    }

    CATLASS_DEVICE
    void Dequant(AscendC::LocalTensor<int8_t> &src, AscendC::LocalTensor<half> &dst,
        AscendC::LocalTensor<int16_t> &value_vector1, AscendC::LocalTensor<int16_t> &value_vector2,
        AscendC::LocalTensor<half> &workspace, half scalar, half zeroPoint)
    {
        pipe_barrier(PIPE_V);
        uint32_t num = COMPUTE_LENGTH;
        num = (num + 128 - 1) / 128 * 128;
        AscendC::Cast<half, uint8_t>(dst.template ReinterpretCast<half>(),
            src.template ReinterpretCast<uint8_t>(),
            AscendC::RoundMode::CAST_NONE,
            num);
        pipe_barrier(PIPE_V);

        AscendC::Adds<half>(dst, dst, 1024, num);
        pipe_barrier(PIPE_V);

        AscendC::ShiftLeft<uint16_t>(
            dst.template ReinterpretCast<uint16_t>(), dst.template ReinterpretCast<uint16_t>(), 7, num);
        pipe_barrier(PIPE_V);

        uint64_t mask = 128;
        AscendC::And<int16_t>(workspace.template ReinterpretCast<int16_t>(),
            dst.template ReinterpretCast<int16_t>(),
            value_vector1,
            mask,
            num / 128,
            {1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_V);

        AscendC::ShiftLeft<uint16_t>(
            workspace.template ReinterpretCast<uint16_t>(), workspace.template ReinterpretCast<uint16_t>(), 1, num);
        pipe_barrier(PIPE_V);

        AscendC::And<int16_t>(dst.template ReinterpretCast<int16_t>(),
            dst.template ReinterpretCast<int16_t>(),
            value_vector2,
            mask,
            num / 128,
            {1, 1, 1, 8, 8, 0});
        pipe_barrier(PIPE_V);

        AscendC::Or<int16_t>(dst.template ReinterpretCast<int16_t>(),
            dst.template ReinterpretCast<int16_t>(),
            workspace.template ReinterpretCast<int16_t>(),
            num);
        pipe_barrier(PIPE_V);

        AscendC::Muls<half>(dst.template ReinterpretCast<half>(), dst.template ReinterpretCast<half>(), 1 << 8, num);
        pipe_barrier(PIPE_V);

        AscendC::Adds(dst, dst, zeroPoint, num);
        pipe_barrier(PIPE_V);

        AscendC::Muls(dst, dst, scalar, num);
        pipe_barrier(PIPE_V);
    }

    CATLASS_DEVICE
    void castFP32toFP16(AscendC::GlobalTensor<float> src, AscendC::GlobalTensor<half> dst, LayoutC layout,
        uint32_t srcStride, uint32_t dstStride)
    {
        AscendC::LocalTensor<float> input[BUFFER_NUM];
        AscendC::LocalTensor<half> output[BUFFER_NUM];

        Arch::Resource<ArchTag> resource;
        int64_t bufferOffset = 0;
        const int64_t CAST_LENGTH = 32 * 1024 / sizeof(half);  // 一次处理16K个数据
        for (int i = 0; i < BUFFER_NUM; i++) {
            input[i] = resource.ubBuf.template GetBufferByByte<float>(bufferOffset);
            bufferOffset += CAST_LENGTH * 4;  // float 4字节
        }
        for (int i = 0; i < BUFFER_NUM; i++) {
            output[i] = resource.ubBuf.template GetBufferByByte<half>(bufferOffset);
            bufferOffset += CAST_LENGTH * 2;  // half 2字节
        }

        BlockLoopInfo blockLoopInfo;
        blockLoopInfo.m = layout.shape(0);
        blockLoopInfo.n = layout.shape(1);

        blockLoopInfo.aivNum = 2;
        blockLoopInfo.aivId = AscendC::GetSubBlockIdx();

        GetBlockLoopInfo(blockLoopInfo, srcStride, dstStride);
        for (int ldx = 0; ldx < blockLoopInfo.totalLoop; ldx++) {
            LoadStoreInfo loadStoreInfo;
            loadStoreInfo.loopIdx = ldx;
            GetLoaderStorerInfo(blockLoopInfo, loadStoreInfo, srcStride, dstStride);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventIdBufferForCast[bufferIndexForCast]);

            auto layoutSrcIn =
                layout::RowMajor(loadStoreInfo.loadRepeat, loadStoreInfo.loadLen, loadStoreInfo.srcLoadStride);
            auto layoutDstIn =
                layout::RowMajor(loadStoreInfo.loadRepeat, loadStoreInfo.loadLen, loadStoreInfo.dstLoadStride);
            copyGm2UbFP32(input[bufferIndexForCast], src[loadStoreInfo.srcProcessOffset], layoutDstIn, layoutSrcIn);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::Cast(
                output[bufferIndexForCast], input[bufferIndexForCast], AscendC::RoundMode::CAST_RINT, CAST_LENGTH);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventIdBufferForCast[bufferIndexForCast]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventIdBufferForCast[bufferIndexForCast]);

            auto layoutSrcOut =
                layout::RowMajor(loadStoreInfo.storeRepeat, loadStoreInfo.storeLen, loadStoreInfo.srcStoreStride);
            auto layoutDstOut =
                layout::RowMajor(loadStoreInfo.storeRepeat, loadStoreInfo.storeLen, loadStoreInfo.dstStoreStride);
            copyUb2Gm(dst[loadStoreInfo.dstProcessOffset], output[bufferIndexForCast], layoutDstOut, layoutSrcOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventIdBufferForCast[bufferIndexForCast]);
            bufferIndexForCast = (bufferIndexForCast + 1) % BUFFER_NUM;
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<int8_t> src, AscendC::GlobalTensor<half> dst, LayoutIn layout,
        uint32_t srcStride, uint32_t dstStride, half scalar, half zeroPoint, uint32_t &bufferIndex)
    {
        BlockLoopInfo blockLoopInfo;
        blockLoopInfo.m = layout.shape(0);
        blockLoopInfo.n = layout.shape(1);
        if (std::is_same<LayoutIn, Catlass::layout::ColumnMajor>::value) {
            blockLoopInfo.m = layout.shape(1);
            blockLoopInfo.n = layout.shape(0);
        }

        blockLoopInfo.aivNum = 2;
        blockLoopInfo.aivId = AscendC::GetSubBlockIdx();

        GetBlockLoopInfo(blockLoopInfo, srcStride, dstStride);
        for (int ldx = 0; ldx < blockLoopInfo.totalLoop; ldx++) {
            LoadStoreInfo loadStoreInfo;
            loadStoreInfo.loopIdx = ldx;
            GetLoaderStorerInfo(blockLoopInfo, loadStoreInfo, srcStride, dstStride);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[bufferIndex]);

            auto layoutSrcIn =
                layout::RowMajor(loadStoreInfo.loadRepeat, loadStoreInfo.loadLen, loadStoreInfo.srcLoadStride);
            auto layoutDstIn =
                layout::RowMajor(loadStoreInfo.loadRepeat, loadStoreInfo.loadLen, loadStoreInfo.dstLoadStride);
            copyGm2Ub(inputBuffer[bufferIndex], src[loadStoreInfo.srcProcessOffset], layoutDstIn, layoutSrcIn);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[bufferIndex]);
            Dequant(inputBuffer[bufferIndex],
                outputBuffer[bufferIndex],
                value_vector1,
                value_vector2,
                workspace[bufferIndex],
                scalar,
                zeroPoint);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EventIdBuffer[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EventIdBuffer[bufferIndex]);

            auto layoutSrcOut =
                layout::RowMajor(loadStoreInfo.storeRepeat, loadStoreInfo.storeLen, loadStoreInfo.srcStoreStride);
            auto layoutDstOut =
                layout::RowMajor(loadStoreInfo.storeRepeat, loadStoreInfo.storeLen, loadStoreInfo.dstStoreStride);
            copyUb2Gm(dst[loadStoreInfo.dstProcessOffset], outputBuffer[bufferIndex], layoutDstOut, layoutSrcOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EventIdBuffer[bufferIndex]);
            bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
        }
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<int8_t> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<int16_t> value_vector1;
    AscendC::LocalTensor<int16_t> value_vector2;
    AscendC::LocalTensor<half> outputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<half> workspace[BUFFER_NUM];
    AscendC::TEventID EventIdBuffer[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    AscendC::TEventID EventIdBufferForCast[BUFFER_NUM] = {EVENT_ID2, EVENT_ID3};
    uint32_t bufferIndexForCast{0};
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_DEQUANT_HPP