/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_CAST_COMBINE_H
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_CAST_COMBINE_H

#include "../../raw_distributed/cam_moe_distribute_combine.h"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "../dispatch_policy.h"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/epilogue/tile/tile_cast.hpp"
#include "catlass_a5/epilogue/tile/tile_copy.hpp"
#include "catlass_a5/epilogue/tile/tile_swizzle.hpp"

namespace Catlass::Epilogue::Block {

template <uint32_t EXEC_FLAG_, class ElementC_, class ElementD_, class TileShape_>
class BlockEpilogue <
    EpilogueAtlasA5CastCombine<EXEC_FLAG_>,
    ElementC_,
    ElementD_,
    TileShape_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA5CastCombine<EXEC_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t EXEC_FLAG = EXEC_FLAG_;
    using ElementC = ElementC_;
    using LayoutC = typename layout::RowMajor;
    using ElementD = ElementD_;
    using LayoutD = typename layout::RowMajor;

    using ElementCompute = ElementC;
    using TileShape = TileShape_;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr uint32_t TILE_M = TileShape::ROW;
    static constexpr uint32_t TILE_N = TileShape::COLUMN;
    static constexpr uint32_t TILE_COUNT = TileShape::COUNT;

    using EpilogueTileSwizzle = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    // Check the element type of C and D
    static_assert(std::is_same_v<ElementC, float>,
        "Element type of C must be float");


    // Epilogue params definition
    struct Params {
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrD;
        LayoutD layoutD;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrD_, LayoutD layoutD_)
            : ptrC(ptrC_), layoutC(layoutC_), ptrD(ptrD_), layoutD(layoutD_) {}
    };

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE void AlignUbOffset()
    {
        size_t ubMask = ubOffset & (MoeDistributeCombineImpl::UB_ALIGN - 1);
        if (ubMask != 0) {
            ubOffset += MoeDistributeCombineImpl::UB_ALIGN - ubMask;
        }
    }

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, MoeDistributeCombineImpl::CombineCalcInfo &calcInfo,
                  Params const &params = Params{})
        : resource(resource),
          calcInfo(calcInfo),
          params(params)
    {
        uint32_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
        if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            AlignUbOffset();
            epSendCountLocal_ = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffset);
            ubOffset += calcInfo.moeSendNum_ * sizeof(int32_t);
            AlignUbOffset();
            AscendC::GlobalTensor<int32_t> epSendCountGM;
            epSendCountGM.SetGlobalBuffer((__gm__ int32_t *)calcInfo.epSendCount_);
            uint32_t epSendCountSize = calcInfo.moeSendNum_;
            AscendC::DataCopyExtParams epSendCntParams = {1U, static_cast<uint32_t>(epSendCountSize * sizeof(uint32_t)),
                                                          0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
            AscendC::DataCopyPad(epSendCountLocal_, epSendCountGM, epSendCntParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventMTE2S);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventMTE2S);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }


    CATLASS_DEVICE GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return Mc2Kernel::GetBaseWindAddrByRankId(calcInfo.epWinContext_, rankId, calcInfo.epRankId_) + calcInfo.winDataSizeOffset_ + expertLocalId * calcInfo.expertPerSizeOnWin_ + rankId * OPT_RANK_OFFSET;
    }

    CATLASS_DEVICE void SetCombineSendEpRank(uint32_t epRank, uint32_t &remoteEpRank, uint32_t &localEpRank)
    {
        remoteEpRank = epRank;
        localEpRank = calcInfo.epRankId_;
    }

    CATLASS_DEVICE void DoCombineSend(AscendC::LocalTensor<ElementD> &ubD, uint32_t expertIdx,
        uint32_t startToken, uint32_t tokenOffset, uint32_t tokenNum, uint32_t tokenLength, uint32_t tileColumn)
    {
        const uint32_t copyTokenLen = tileColumn * sizeof(ElementD);
        const uint32_t copyTokenSrcStride = 0;
        const uint32_t copyTokenDstStride = (tokenLength - tileColumn) * sizeof(ElementD);

        uint32_t itToken = startToken;
        uint32_t endToken = startToken + tokenNum;
        constexpr uint32_t epRankStart = 0;
        uint32_t sendCount =
            expertIdx == 0 && epRankStart == 0 ? 0 : epSendCountLocal_.GetValue(expertOffset + epRankStart - 1);
        for (uint32_t epRank = epRankStart; epRank < calcInfo.epWorldSize_ && itToken < endToken; ++epRank) {
            uint32_t prevSendCount = sendCount;
            sendCount = epSendCountLocal_.GetValue(expertOffset + epRank);
            if (prevSendCount <= itToken && itToken < sendCount) {
                uint32_t copyTokenCount = (sendCount < endToken ? sendCount : endToken) - itToken;
                AscendC::DataCopyExtParams dataCopyParams(copyTokenCount, copyTokenLen, copyTokenSrcStride,
                                                          copyTokenDstStride, 0);
                uint32_t remoteEpRank;
                uint32_t localEpRank;
                SetCombineSendEpRank(epRank, remoteEpRank, localEpRank);
                GM_ADDR rankGM = GetWinAddrByRankId(remoteEpRank, expertIdx) +
                                 localEpRank * calcInfo.moeExpertPerRankNum_ * calcInfo.expertPerSizeOnWin_;
                AscendC::GlobalTensor<ElementD> rankWindow;
                rankWindow.SetGlobalBuffer((__gm__ ElementD *)rankGM);
                AscendC::DataCopyPad(rankWindow[(itToken - prevSendCount) * calcInfo.axisH_ + tokenOffset],
                                     ubD[(itToken - startToken) * tileColumn], dataCopyParams);
                itToken += copyTokenCount;
            }
        }
    }

    template <class TensorC, class TensorD>
    CATLASS_DEVICE
    void operator() (
        TensorC &tensorBlockC,
        TensorD &tensorBlockD,
        GemmCoord const &actualBlockShapeMNK,
        uint32_t expertIdx,
        uint32_t tokenIdx,
        uint32_t tokenOffset
    )
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
            expertOffset = expertIdx * calcInfo.epWorldSize_;
        }
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();

        auto ubTileStride = static_cast<uint32_t>(TileShape::COLUMN);
        auto ubTileStrideRow = static_cast<uint32_t>(TileShape::ROW);
        auto tileShape = MakeCoord(TileShape::ROW, TileShape::COLUMN);
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = expertIdx == UINT32_MAX ? 0 : AscendC::GetSubBlockIdx();
        uint32_t subblockNum = expertIdx == UINT32_MAX ? 1 : AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            MatrixCoord tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffsetInBlockRow = tileOffsetInBlock.row();
            auto tileOffsetInBlockColumn = tileOffsetInBlock.column();
            uint32_t count = actualTileShape[0] * actualTileShape[1];

            // build tensor C block in GM
            auto tensorSubBlockC = GetTile(
                tensorBlockC, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                tla::MakeShape(actualTileShape.row(), actualTileShape.column())
            );
            // build tensor C block in UB
            auto &ubC = ubCList[ubListId];
            auto layoutUbC = tla::MakeLayout(
                tla::MakeShape(actualTileShape.row(), actualTileShape.column()), tla::MakeStride(ubTileStride, tla::Int<1>{})
            );
            auto tensorUbC = tla::MakeTensor(ubC, layoutUbC, Arch::PositionUB{});
            using CopyGmToUbC = typename Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, TensorC, decltype(tensorUbC)>;
            CopyGmToUbC copyGmToUbC;
            // copy tensor C from GM to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(tensorUbC, tensorSubBlockC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            auto &ubD = ubDList[ubListId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            Cast(ubD, ubC, AscendC::RoundMode::CAST_RINT, count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            // build tensor D block in GM
            auto tensorSubBlockD = GetTile(
                tensorBlockD, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                tla::MakeShape(actualTileShape.row(), actualTileShape.column())
            );
            // build tensor D block in UB
            auto tensorUbD = tla::MakeTensor(ubD, layoutUbC, Arch::PositionUB{});
            using CopyUbToGmD = typename Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(tensorUbD), TensorD>;
            CopyUbToGmD copyUbToGmD;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
                if (expertIdx == UINT32_MAX) {
                    copyUbToGmD(tensorSubBlockD, tensorUbD);
                } else {
                    DoCombineSend(ubD, expertIdx, tokenIdx + loopIdx * TileShape::ROW, tokenOffset, actualTileShape[0], tla::get<0>(tensorBlockD.stride()), TileShape::COLUMN);
                }
            } else {
                copyUbToGmD(tensorSubBlockD, tensorUbD);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;
    Arch::Resource<ArchTag> &resource;
    MoeDistributeCombineImpl::CombineCalcInfo calcInfo;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];
    AscendC::LocalTensor<int32_t> epSendCountLocal_;

    size_t ubOffset{0};
    int32_t eventMTE2S{0};
    uint32_t expertOffset;
    uint32_t ubListId{0};

};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_CAST_COMBINE_H