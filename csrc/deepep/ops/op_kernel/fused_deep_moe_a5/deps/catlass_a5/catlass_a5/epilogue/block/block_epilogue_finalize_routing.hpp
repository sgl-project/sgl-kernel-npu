/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FINALIZE_ROUTING_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FINALIZE_ROUTING_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/layout/matrix.hpp"
#include "catlass_a5/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass_a5/epilogue/tile/copy_ub_to_gm.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

namespace Catlass::Epilogue::Block {
using namespace Catlass::Epilogue::Tile;
template <
    class DispatchPolicy_,
    class ArchTag_,
    class VecTileShape_,
    class ElementC_,
    class ElementRowIndex_,
    class ElementSharedInput_
>
class BlockEpilogueFinalizeRouting {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = ArchTag_;
    using VecTileShape = VecTileShape_;
    using ElementC = ElementC_;
    using ElementRowIndex = ElementRowIndex_;
    using ElementSharedInput = ElementSharedInput_;
    using SafeSharedInput = std::conditional_t<std::is_void_v<ElementSharedInput>, uint8_t, ElementSharedInput>;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr uint32_t EPILOGUE_TILE_M = VecTileShape::ROW;
    static constexpr uint32_t EPILOGUE_TILE_N = VecTileShape::COLUMN;

    static constexpr uint32_t ALIGNED_VALUE = 8;

    static constexpr uint32_t ONE_REPEAT_SIZE = 256;
    static constexpr uint32_t ONE_DATA_BLOCK_SIZE = 32;
    static constexpr uint32_t ONE_BLOCK_ELEMENT = ONE_DATA_BLOCK_SIZE / sizeof(ElementC); // 8
    static constexpr uint32_t ONE_REPEAT_ELEMENT = ONE_REPEAT_SIZE / sizeof(ElementC); // 64
    static constexpr uint32_t MAX_VECTOR_REPEAT_COUNT = 255;
    static constexpr uint32_t MAX_REPEAT_SPLIT_HALF = (MAX_VECTOR_REPEAT_COUNT + 1) / 2;

    static constexpr size_t UB_BUF_GMM_BYTES = EPILOGUE_TILE_M * EPILOGUE_TILE_N * sizeof(ElementC);
    static constexpr size_t UB_BUF_ROW_INDEX_BYTES = EPILOGUE_TILE_M * sizeof(ElementRowIndex);
    static constexpr size_t UB_BUF_LOGIT_BYTES = EPILOGUE_TILE_M * sizeof(ElementC);
    static constexpr size_t UB_BUF_LOGIT_BRCB_BYTES = UB_BUF_LOGIT_BYTES * ONE_BLOCK_ELEMENT;
    static constexpr size_t UB_PER_STAGE = UB_BUF_GMM_BYTES + UB_BUF_LOGIT_BYTES + UB_BUF_LOGIT_BRCB_BYTES + UB_BUF_ROW_INDEX_BYTES;

    static constexpr size_t MAX_CLEAR_GM_COUNT = 50 * 1024;

    static constexpr size_t MAX_SOLVE_SHARED_INPUT_COUNT = 20 * 1024;
    static constexpr size_t UB_BUF_SHARED_INPUT = MAX_SOLVE_SHARED_INPUT_COUNT * sizeof(SafeSharedInput);
    static constexpr size_t UB_BUF_SHARED_INPUT_CAST = MAX_SOLVE_SHARED_INPUT_COUNT * sizeof(ElementC);
    static constexpr size_t UB_BUF_SHARED_OUTPUT = MAX_SOLVE_SHARED_INPUT_COUNT * sizeof(ElementC);
    static constexpr size_t UB_PER_STAGE_SHARED_INPUT = UB_BUF_SHARED_INPUT + UB_BUF_SHARED_INPUT_CAST + UB_BUF_SHARED_OUTPUT;

    static_assert(UB_STAGES * UB_PER_STAGE <= ArchTag::UB_SIZE, "UB budget exceeded for BlockEpilogueFinalizeRouting");
    static_assert(UB_STAGES * UB_PER_STAGE_SHARED_INPUT <= ArchTag::UB_SIZE, "UB budget exceeded for BlockEpilogueFinalizeRouting");
    
    GemmCoord problemShape;
    using LayoutTagMatric = layout::RowMajor;
    using GmMatricType = Gemm::GemmType<ElementC, LayoutTagMatric>;
    using CopyGm2UbMatrix = CopyGm2Ub<ArchTag, GmMatricType>;
    using CopyUb2GmMatrix = CopyUb2Gm<ArchTag, GmMatricType>;
    CopyGm2UbMatrix copyGm2UbMatrix;
    CopyUb2GmMatrix copyUb2GmMatrix;

    using LayoutTagVec = layout::VectorLayout;
    using GmVecType = Gemm::GemmType<ElementC, LayoutTagVec>;
    using CopyGm2UbVec = CopyGm2Ub<ArchTag, GmVecType>;
    CopyGm2UbVec copyGm2UbVec;

    using GmVecTypeRowIndex = Gemm::GemmType<ElementRowIndex, LayoutTagVec>;
    using CopyRowIndexGm2UbVec = CopyGm2Ub<ArchTag, GmVecTypeRowIndex>;
    CopyRowIndexGm2UbVec copyRowIndexGm2UbVec;

    using GmVecTypeSharedInput = Gemm::GemmType<SafeSharedInput, LayoutTagVec>;
    using CopySharedInputGm2UbVec = CopyGm2Ub<ArchTag, GmVecTypeSharedInput>;
    CopySharedInputGm2UbVec copySharedGm2UbVec;

    CATLASS_DEVICE
    BlockEpilogueFinalizeRouting(Arch::Resource<ArchTag> &resource)
    {
        AllocateUbBuffers(resource);
    }

    CATLASS_DEVICE
    ~BlockEpilogueFinalizeRouting()
    {
    }
    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_)
    {
        problemShape = problemShape_;
    }

    CATLASS_DEVICE
    void ClearOutTile(
        AscendC::GlobalTensor<ElementC> const &gmOutTile,
        MatrixCoord const &outSplitCoord
    )
    {
        int64_t clearCount = outSplitCoord.column() * problemShape.n();
        int64_t singleCount = clearCount > MAX_CLEAR_GM_COUNT ? MAX_CLEAR_GM_COUNT : clearCount;

        AscendC::Duplicate<ElementC>(ubBufClearGm, static_cast<ElementC>(0.0f), singleCount);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        for (size_t idx = 0; idx < clearCount; idx+=singleCount) {
            uint32_t curNum = ((idx + singleCount) < clearCount) ? singleCount : (clearCount - idx);
            auto layoutVecSrcOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            auto layoutVecDstOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            copyUb2GmMatrix(gmOutTile[idx], ubBufClearGm, layoutVecDstOut, layoutVecSrcOut);
        }
    }

    CATLASS_DEVICE
    void AssignSharedInputTile(
        AscendC::GlobalTensor<SafeSharedInput> const &sharedInputTile,
        AscendC::GlobalTensor<ElementC> const &sharedOutputTile,
        MatrixCoord const &outSplitCoord,
        float sharedInputWeight)
    {
        if constexpr (std::is_void_v<ElementSharedInput>) {
            return;
        }
        int64_t count = outSplitCoord.column() * problemShape.n();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        for(size_t idx = 0; idx < count; idx += MAX_SOLVE_SHARED_INPUT_COUNT) {
            uint32_t curNum = ((idx + MAX_SOLVE_SHARED_INPUT_COUNT) < count) ? MAX_SOLVE_SHARED_INPUT_COUNT : (count - idx);
            auto layoutVecSrc = LayoutTagVec(curNum);
            auto layoutVecDst = LayoutTagVec(curNum);
            copySharedGm2UbVec(ubBufSharedInput, sharedInputTile[idx], layoutVecDst, layoutVecSrc);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::Cast(ubBufSharedInputCast, ubBufSharedInput, AscendC::RoundMode::CAST_NONE, curNum);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

            AscendC::Muls(ubBufSharedOut, ubBufSharedInputCast, sharedInputWeight, curNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutVecSrcOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            auto layoutVecDstOut = LayoutTagMatric::MakeLayout<ElementC>(1, curNum);
            copyUb2GmMatrix(sharedOutputTile[idx], ubBufSharedOut, layoutVecDstOut, layoutVecSrcOut);
        }
    }

    CATLASS_DEVICE
    void LogitScatterAddTile(
        AscendC::LocalTensor<ElementC> const &ubBufGmm1,
        AscendC::GlobalTensor<ElementC> const &gmmInputTile,
        AscendC::GlobalTensor<ElementC> const &gmLogitTile,
        AscendC::GlobalTensor<ElementRowIndex> const &gmRowIndexTile,
        AscendC::GlobalTensor<ElementC> const &gmOutTile,
        GemmCoord const &tileShape,
        GemmCoord const &gmmTileShape)
    {
        if (tileShape.n() == 0) {
            return;
        }
        auto layoutGmmSrc = LayoutTagMatric(tileShape.m(), tileShape.n(), LayoutTagMatric::LongIndex(gmmTileShape.n()));
        auto layoutGmmDst = LayoutTagMatric(tileShape.m(), tileShape.n(), LayoutTagMatric::LongIndex(RoundUp(tileShape.n(), 8)));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        copyGm2UbMatrix(ubBufGmm, gmmInputTile, layoutGmmDst, layoutGmmSrc);

        auto layoutVecSrc = LayoutTagVec(tileShape.m());
        auto layoutVecDst = LayoutTagVec(tileShape.m());
        copyGm2UbVec(ubBufLogit, gmLogitTile, layoutVecDst, layoutVecSrc);
        copyRowIndexGm2UbVec(ubBufRowIndex, gmRowIndexTile, layoutVecDst, layoutVecSrc);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        bool isOverRepeatMax = tileShape.m() > MAX_VECTOR_REPEAT_COUNT;
        uint32_t repeatCount = isOverRepeatMax ? MAX_VECTOR_REPEAT_COUNT : tileShape.m();

        if(isOverRepeatMax) {
            AscendC::Brcb(ubBufLogitBrcb, ubBufLogit, MAX_REPEAT_SPLIT_HALF, {1, ONE_BLOCK_ELEMENT});
            AscendC::Brcb(ubBufLogitBrcb[MAX_REPEAT_SPLIT_HALF * ONE_BLOCK_ELEMENT], ubBufLogit[MAX_REPEAT_SPLIT_HALF], MAX_REPEAT_SPLIT_HALF, {1, ONE_BLOCK_ELEMENT});
        } else {
            AscendC::Brcb(ubBufLogitBrcb, ubBufLogit, repeatCount, {1, ONE_BLOCK_ELEMENT});
        }
        AscendC::PipeBarrier<PIPE_V>();
        uint8_t dstRepStrideIn = RoundUp(tileShape.n(), ALIGNED_VALUE) / ONE_BLOCK_ELEMENT;
        AscendC::BinaryRepeatParams repeatParams{1, 1, 0, dstRepStrideIn, dstRepStrideIn, 1};
        for(size_t idx = 0; idx < tileShape.n() / ONE_REPEAT_ELEMENT; idx++) {
            AscendC::Mul(ubBufGmm[idx * ONE_REPEAT_ELEMENT], ubBufGmm[idx * ONE_REPEAT_ELEMENT], ubBufLogitBrcb, ONE_REPEAT_ELEMENT, repeatCount, repeatParams);
        }
        if (tileShape.n() % ONE_REPEAT_ELEMENT != 0) {
            AscendC::Mul(ubBufGmm[tileShape.n() / ONE_REPEAT_ELEMENT * ONE_REPEAT_ELEMENT], ubBufGmm[tileShape.n() / ONE_REPEAT_ELEMENT * ONE_REPEAT_ELEMENT], ubBufLogitBrcb, tileShape.n() % ONE_REPEAT_ELEMENT, repeatCount, repeatParams);
        }
        if (isOverRepeatMax) {
            AscendC::Mul(ubBufGmm[repeatCount * dstRepStrideIn * ONE_BLOCK_ELEMENT], ubBufGmm[repeatCount * dstRepStrideIn * ONE_BLOCK_ELEMENT], ubBufLogitBrcb[repeatCount * ONE_BLOCK_ELEMENT], ONE_REPEAT_ELEMENT, CeilDiv(tileShape.n(), ONE_REPEAT_ELEMENT), {1, 1, 0, 8, 8, 0});
        }
        AscendC::PipeBarrier<PIPE_V>();
        auto layoutMatricOutSrc = LayoutTagMatric(1, tileShape.n());
        auto layoutMatricOutDst = LayoutTagMatric(1, tileShape.n());
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::SetAtomicAdd<ElementC>();
        for(size_t idx = 0; idx < tileShape.m(); idx++) {
            int64_t resRowIdx = ubBufRowIndex.GetValue(idx);
            int64_t ubOffset = idx * RoundUp(tileShape.n(), ALIGNED_VALUE);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
            copyUb2GmMatrix(gmOutTile[resRowIdx * problemShape.n()], ubBufGmm[ubOffset], layoutMatricOutDst, layoutMatricOutSrc);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
        AscendC::DisableDmaAtomic();
    }

private:
    AscendC::LocalTensor<ElementC> ubBufClearGm;
    AscendC::LocalTensor<SafeSharedInput> ubBufSharedInput;
    AscendC::LocalTensor<ElementC> ubBufSharedInputCast;
    AscendC::LocalTensor<ElementC> ubBufSharedOut;

    AscendC::LocalTensor<ElementC> ubBufGmm;
    AscendC::LocalTensor<ElementC> ubBufLogit;
    AscendC::LocalTensor<ElementC> ubBufLogitBrcb;
    AscendC::LocalTensor<ElementRowIndex> ubBufRowIndex;

    CATLASS_DEVICE
    void AllocateUbBuffers(Arch::Resource<ArchTag> &resource)
    {
        size_t offset = 0;

        ubBufClearGm = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        ubBufSharedInput = resource.ubBuf.template GetBufferByByte<SafeSharedInput>(offset);
        offset += UB_BUF_SHARED_INPUT;
        ubBufSharedInputCast = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_SHARED_INPUT_CAST;
        ubBufSharedOut = resource.ubBuf.template GetBufferByByte<ElementC>(offset);

        offset = 0;
        ubBufGmm = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_GMM_BYTES;
        ubBufRowIndex = resource.ubBuf.template GetBufferByByte<ElementRowIndex>(offset);
        offset += UB_BUF_ROW_INDEX_BYTES;
        ubBufLogit = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_LOGIT_BYTES;
        ubBufLogitBrcb = resource.ubBuf.template GetBufferByByte<ElementC>(offset);
        offset += UB_BUF_LOGIT_BRCB_BYTES;
    }
};

} // namespace Catlass::Epilogue::Block

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FINALIZE_ROUTING_HPP
