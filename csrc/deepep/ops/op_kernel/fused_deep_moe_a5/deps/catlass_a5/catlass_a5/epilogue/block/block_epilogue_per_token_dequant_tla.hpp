/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_TLA_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Epilogue::Block {

template <uint32_t UB_STAGES_, class ElementC_, class ElementScale_, class ElementPerTokenScale_, class ElementD_,
          class TileRowBroadcastMul_, class TileBroadcastOneBlk_, class TileOneBlkColumnBroadcastMul_, class TileCopy_,
          class EpilogueTileSwizzle_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantTla<UB_STAGES_>, ElementC_, ElementScale_, ElementPerTokenScale_,
                    ElementD_, TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_, TileCopy_,
                    EpilogueTileSwizzle_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantTla<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    using TileCopy = TileCopy_;

    // Data infos
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementScale = ElementScale_;
    using LayoutScale = typename TileCopy::LayoutX;
    using ElementPerTokenScale = ElementPerTokenScale_;
    using LayoutPerTokenScale = typename TileCopy::LayoutY;
    using ElementD = ElementD_;
    using LayoutD = typename TileCopy::LayoutD;

    // Check data infos
    static_assert(std::is_same_v<ElementC, int32_t> &&
                      (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>) &&
                      std::is_same_v<ElementScale, ElementD> && std::is_same_v<ElementPerTokenScale, ElementD>,
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(tla::detail::isRowMajor<LayoutC>::value && tla::detail::isRowMajor<LayoutD>::value,
                  "The layout template parameters of BlockEpilogue are wrong");

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(float);

    static_assert(TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
                      std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
                  "TileShape must be consistent for all tile compute ops");

    static_assert((UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COLUMN * sizeof(ElementScale) +
                                TileShape::ROW * sizeof(ElementPerTokenScale) + TileShape::COUNT * sizeof(ElementD)) +
                   (TileShape::COUNT + TileShape::COLUMN + TileShape::COUNT + TileShape::ROW) * sizeof(float) +
                   TileShape::ROW * BYTE_PER_BLK) <= ArchTag::UB_SIZE,
                  "TileShape is too large to fit in UB");

    struct Params {
        GM_ADDR ptrScale{nullptr};
        LayoutScale layoutScale{};
        GM_ADDR ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        GM_ADDR ptrD{nullptr};
        LayoutD layoutD{};
        Params() = default;
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}, uint32_t sharedUbSize = 0)
        : params(params)
    {
        uint32_t ubOffset = sharedUbSize;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbScaleVMTE2List[i] = eventVMTE2++;
            eventUbScaleMTE2VList[i] = eventMTE2V++;
            eventUbPerTokenScaleVMTE2List[i] = eventVMTE2++;
            eventUbPerTokenScaleMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubScaleFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COLUMN * sizeof(float);
        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * sizeof(float);
        ubPerTokenScaleFp32Brcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubMul;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    template <class TensorC, class TensorScale, class TensorPerTokenScaleScale, class TensorD>
    CATLASS_DEVICE void operator()(TensorC &tensorBlockC, TensorScale &tensorBlockScale,
                                   TensorPerTokenScaleScale &tensorBlockPerTokenScaleScale, TensorD &tensorBlockD,
                                   GemmCoord const &actualBlockShapeMNK, Callback &&callback = Callback{})
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        callback();

        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();

        using CopyGmToUbC = typename TileCopy::template CopyGmToUbC<TensorC>;
        using CopyGmToUbScale = typename TileCopy::template CopyGmToUbX<TensorScale>;
        using CopyGmToUbPerTokenScale = typename TileCopy::template CopyGmToUbY<TensorPerTokenScaleScale>;
        using CopyUbToGmD = typename TileCopy::template CopyUbToGmD<TensorD>;
        CopyGmToUbC copyGmToUbC;
        CopyGmToUbScale copyGmToUbScale;
        CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
        CopyUbToGmD copyUbToGmD;

        auto ubTileStride = static_cast<uint32_t>(TileShape::COLUMN);
        auto ubTileStrideRow = static_cast<uint32_t>(TileShape::ROW);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            MatrixCoord tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffsetInBlockRow = tileOffsetInBlock.row();
            auto tileOffsetInBlockColumn = tileOffsetInBlock.column();

            // build tensor C block in GM
            auto tensorSubBlockC = GetTile(tensorBlockC, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                                           tla::MakeShape(actualTileShape.row(), actualTileShape.column()));
            // build tensor C block in UB
            auto &ubC = ubCList[ubListId];
            auto layoutUbC = tla::MakeLayout(tla::MakeShape(actualTileShape.row(), actualTileShape.column()),
                                             tla::MakeStride(ubTileStride, tla::Int<1>{}));
            auto tensorUbC = tla::MakeTensor(ubC, layoutUbC, Arch::PositionUB{});
            // copy tensor C from GM to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(tensorUbC, tensorSubBlockC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            ///////////////////////////////////////////////////
            // build tensor Scale block in GM
            auto tensorSubBlockScale = GetTile(tensorBlockScale, tla::MakeCoord(0, tileOffsetInBlockColumn),
                                               tla::MakeShape(tla::Int<1>{}, actualTileShape.column()));
            // build tensor Scale block in UB
            auto &ubScale = ubScaleList[ubListId];
            auto layoutUbScale = tla::MakeLayout(tla::MakeShape(tla::Int<1>{}, actualTileShape.column()),
                                                 tla::MakeStride(ubTileStride, tla::Int<1>{}));
            auto tensorUbScale = tla::MakeTensor(ubScale, layoutUbScale, Arch::PositionUB{});
            // copy tensor Scale from GM to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[ubListId]);
            copyGmToUbScale(tensorUbScale, tensorSubBlockScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            ///////////////////////////////////////////////////
            // build tensor PerTokenScale block in GM
            auto tensorSubBlockPerTokenScale =
                GetTile(tensorBlockPerTokenScaleScale, tla::MakeCoord(0, tileOffsetInBlockRow),
                        tla::MakeShape(1, actualTileShape.row()));
            // build tensor PerTokenScale block in UB
            auto &ubPerTokenScale = ubPerTokenScaleList[ubListId];
            auto layoutUbPerTokenScale = tla::MakeLayout(tla::MakeShape(tla::Int<1>{}, actualTileShape.row()),
                                                         tla::MakeStride(ubTileStrideRow, tla::Int<1>{}));
            auto tensorUbPerTokenScale = tla::MakeTensor(ubPerTokenScale, layoutUbPerTokenScale, Arch::PositionUB{});
            // copy tensor PerTokenScale from GM to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[ubListId]);
            copyGmToUbPerTokenScale(tensorUbPerTokenScale, tensorSubBlockPerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            ///////////////////////////////////////////////////
            auto tensorUbScaleFp32 = tla::MakeTensor(ubScaleFp32, layoutUbScale, Arch::PositionUB{});
            auto tensorUbPerTokenScaleFp32 =
                tla::MakeTensor(ubPerTokenScaleFp32, layoutUbPerTokenScale, Arch::PositionUB{});
            auto tensorUbCFp32 = tla::MakeTensor(ubCFp32, layoutUbC, Arch::PositionUB{});
            auto tensorUbMul = tla::MakeTensor(ubMul, layoutUbC, Arch::PositionUB{});

            auto layoutUbPerTokenScaleBrcb = tla::MakeLayout(tla::MakeShape(actualTileShape.row(), ELE_NUM_PER_BLK),
                                                             tla::MakeStride(ELE_NUM_PER_BLK, tla::Int<1>{}));
            auto tensorUbPerTokenScaleFp32Brcb =
                tla::MakeTensor(ubPerTokenScaleFp32Brcb, layoutUbPerTokenScaleBrcb, Arch::PositionUB{});
            auto tensorUbPerTokenMul = tla::MakeTensor(ubPerTokenMul, layoutUbC, Arch::PositionUB{});

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);
            AscendC::Cast(ubScaleFp32, ubScale, AscendC::RoundMode::CAST_NONE, TileShape::COLUMN);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);
            AscendC::Cast(ubPerTokenScaleFp32, ubPerTokenScale, AscendC::RoundMode::CAST_NONE, TileShape::ROW);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[ubListId]);

            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(tensorUbMul, tensorUbCFp32, tensorUbScaleFp32);
            tileBroadcastOneBlk(tensorUbPerTokenScaleFp32Brcb, tensorUbPerTokenScaleFp32);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(tensorUbPerTokenMul, tensorUbMul, tensorUbPerTokenScaleFp32Brcb);
            AscendC::PipeBarrier<PIPE_V>();

            auto &ubD = ubDList[ubListId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubPerTokenMul, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            // build tensor D block in GM
            auto tensorSubBlockD = GetTile(tensorBlockD, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                                           tla::MakeShape(actualTileShape.row(), actualTileShape.column()));
            // build tensor D block in UB
            auto tensorUbD = tla::MakeTensor(ubD, layoutUbC, Arch::PositionUB{});

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(tensorSubBlockD, tensorUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbScaleVMTE2List[UB_STAGES];
    int32_t eventUbScaleMTE2VList[UB_STAGES];
    int32_t eventUbPerTokenScaleVMTE2List[UB_STAGES];
    int32_t eventUbPerTokenScaleMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32;
    AscendC::LocalTensor<float> ubScaleFp32;
    AscendC::LocalTensor<float> ubMul;
    AscendC::LocalTensor<float> ubPerTokenScaleFp32;
    AscendC::LocalTensor<float> ubPerTokenScaleFp32Brcb;
    AscendC::LocalTensor<float> ubPerTokenMul;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;
};

template <uint32_t UB_STAGES_, class ElementC_, class ElementD_, class TileRowBroadcastMul_, class TileBroadcastOneBlk_,
          class TileOneBlkColumnBroadcastMul_, class TileCopy_, class EpilogueTileSwizzle_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantTla<UB_STAGES_>, ElementC_, float, float, ElementD_,
                    TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_, TileCopy_,
                    EpilogueTileSwizzle_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantTla<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    using TileCopy = TileCopy_;

    // Data infos
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementScale = float;
    using LayoutScale = typename TileCopy::LayoutX;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = typename TileCopy::LayoutY;
    using ElementD = ElementD_;
    using LayoutD = typename TileCopy::LayoutD;

    // Check data infos
    static_assert(std::is_same_v<ElementC, int32_t> &&
                      (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t> ||
                       std::is_same_v<ElementD, float>),
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(tla::detail::isRowMajor<LayoutC>::value && tla::detail::isRowMajor<LayoutD>::value,
                  "The layout template parameters of BlockEpilogue are wrong");

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(float);

    static_assert(TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
                      std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
                  "TileShape must be consistent for all tile compute ops");

    static_assert((UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COLUMN * sizeof(ElementScale) +
                                TileShape::ROW * sizeof(ElementPerTokenScale) + TileShape::COUNT * sizeof(ElementD)) +
                   (TileShape::COUNT + TileShape::COUNT) * sizeof(float) + TileShape::ROW * BYTE_PER_BLK) <=
                      ArchTag::UB_SIZE,
                  "TileShape is too large to fit in UB");

    struct Params {
        GM_ADDR ptrScale{nullptr};
        LayoutScale layoutScale{};
        GM_ADDR ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        GM_ADDR ptrD{nullptr};
        LayoutD layoutD{};
        Params() = default;
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}, uint32_t sharedUbSize = 0)
        : params(params)
    {
        uint32_t ubOffset = sharedUbSize;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbScaleVMTE2List[i] = eventVMTE2++;
            eventUbScaleMTE2VList[i] = eventMTE2V++;
            eventUbPerTokenScaleVMTE2List[i] = eventVMTE2++;
            eventUbPerTokenScaleMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubCFp32;
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    template <class TensorC, class TensorScale, class TensorPerTokenScaleScale, class TensorD>
    CATLASS_DEVICE void operator()(TensorC &tensorBlockC, TensorScale &tensorBlockScale,
                                   TensorPerTokenScaleScale &tensorBlockPerTokenScaleScale, TensorD &tensorBlockD,
                                   GemmCoord const &actualBlockShapeMNK, Callback &&callback = Callback{})
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        callback();

        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();

        using CopyGmToUbC = typename TileCopy::template CopyGmToUbC<TensorC>;
        using CopyGmToUbScale = typename TileCopy::template CopyGmToUbX<TensorScale>;
        using CopyGmToUbPerTokenScale = typename TileCopy::template CopyGmToUbY<TensorPerTokenScaleScale>;
        using CopyUbToGmD = typename TileCopy::template CopyUbToGmD<TensorD>;
        CopyGmToUbC copyGmToUbC;
        CopyGmToUbScale copyGmToUbScale;
        CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
        CopyUbToGmD copyUbToGmD;

        auto ubTileStride = static_cast<uint32_t>(TileShape::COLUMN);
        auto ubTileStrideRow = static_cast<uint32_t>(TileShape::ROW);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            MatrixCoord tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffsetInBlockRow = tileOffsetInBlock.row();
            auto tileOffsetInBlockColumn = tileOffsetInBlock.column();

            // build tensor C block in GM
            auto tensorSubBlockC = GetTile(tensorBlockC, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                                           tla::MakeShape(actualTileShape.row(), actualTileShape.column()));
            // build tensor C block in UB
            auto &ubC = ubCList[ubListId];
            auto layoutUbC = tla::MakeLayout(tla::MakeShape(actualTileShape.row(), actualTileShape.column()),
                                             tla::MakeStride(ubTileStride, tla::Int<1>{}));
            auto tensorUbC = tla::MakeTensor(ubC, layoutUbC, Arch::PositionUB{});
            // copy tensor C from GM to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(tensorUbC, tensorSubBlockC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            ///////////////////////////////////////////////////
            // build tensor Scale block in GM
            auto tensorSubBlockScale = GetTile(tensorBlockScale, tla::MakeCoord(0, tileOffsetInBlockColumn),
                                               tla::MakeShape(tla::Int<1>{}, actualTileShape.column()));
            // build tensor Scale block in UB
            auto &ubScale = ubScaleList[ubListId];
            auto layoutUbScale = tla::MakeLayout(tla::MakeShape(tla::Int<1>{}, actualTileShape.column()),
                                                 tla::MakeStride(ubTileStride, tla::Int<1>{}));
            auto tensorUbScale = tla::MakeTensor(ubScale, layoutUbScale, Arch::PositionUB{});
            // copy tensor Scale from GM to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[ubListId]);
            copyGmToUbScale(tensorUbScale, tensorSubBlockScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            ///////////////////////////////////////////////////
            // build tensor PerTokenScale block in GM
            auto tensorSubBlockPerTokenScale =
                GetTile(tensorBlockPerTokenScaleScale, tla::MakeCoord(0, tileOffsetInBlockRow),
                        tla::MakeShape(tla::Int<1>{}, actualTileShape.row()));
            // build tensor PerTokenScale block in UB
            auto &ubPerTokenScale = ubPerTokenScaleList[ubListId];
            auto layoutUbPerTokenScale = tla::MakeLayout(tla::MakeShape(tla::Int<1>{}, actualTileShape.row()),
                                                         tla::MakeStride(ubTileStrideRow, tla::Int<1>{}));
            auto tensorUbPerTokenScale = tla::MakeTensor(ubPerTokenScale, layoutUbPerTokenScale, Arch::PositionUB{});
            // copy tensor PerTokenScale from GM to UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[ubListId]);
            copyGmToUbPerTokenScale(tensorUbPerTokenScale, tensorSubBlockPerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            ///////////////////////////////////////////////////
            auto tensorUbCFp32 = tla::MakeTensor(ubCFp32, layoutUbC, Arch::PositionUB{});
            auto tensorUbMul = tla::MakeTensor(ubMul, layoutUbC, Arch::PositionUB{});

            auto layoutUbPerTokenScaleBrcb = tla::MakeLayout(tla::MakeShape(actualTileShape.row(), ELE_NUM_PER_BLK),
                                                             tla::MakeStride(ELE_NUM_PER_BLK, tla::Int<1>{}));
            auto tensorUbPerTokenScaleBrcb =
                tla::MakeTensor(ubPerTokenScaleBrcb, layoutUbPerTokenScaleBrcb, Arch::PositionUB{});
            auto tensorUbPerTokenMul = tla::MakeTensor(ubPerTokenMul, layoutUbC, Arch::PositionUB{});

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);
            tileRowBroadcastMul(tensorUbMul, tensorUbCFp32, tensorUbScale);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[ubListId]);

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);
            tileBroadcastOneBlk(tensorUbPerTokenScaleBrcb, tensorUbPerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenScaleVMTE2List[ubListId]);

            // build tensor D block in UB
            auto &ubD = ubDList[ubListId];
            auto tensorUbD = tla::MakeTensor(ubD, layoutUbC, Arch::PositionUB{});

            if constexpr (std::is_same_v<ElementD, float>) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                tileOneBlkColumnBroadcastMul(tensorUbD, tensorUbMul, tensorUbPerTokenScaleBrcb);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            } else if constexpr (std::is_same_v<ElementD, half>) {
                AscendC::PipeBarrier<PIPE_V>();
                tileOneBlkColumnBroadcastMul(tensorUbPerTokenMul, tensorUbMul, tensorUbPerTokenScaleBrcb);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                AscendC::Cast(ubD, ubPerTokenMul, AscendC::RoundMode::CAST_NONE, TileShape::COUNT);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            } else {  // std::is_same_v<ElementD, bfloat16_t>
                AscendC::PipeBarrier<PIPE_V>();
                tileOneBlkColumnBroadcastMul(tensorUbPerTokenMul, tensorUbMul, tensorUbPerTokenScaleBrcb);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                AscendC::Cast(ubD, ubPerTokenMul, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            }

            // build tensor D block in GM
            auto tensorSubBlockD = GetTile(tensorBlockD, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                                           tla::MakeShape(actualTileShape.row(), actualTileShape.column()));

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(tensorSubBlockD, tensorUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbScaleVMTE2List[UB_STAGES];
    int32_t eventUbScaleMTE2VList[UB_STAGES];
    int32_t eventUbPerTokenScaleVMTE2List[UB_STAGES];
    int32_t eventUbPerTokenScaleMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubCFp32;
    AscendC::LocalTensor<float> ubMul;
    AscendC::LocalTensor<float> ubPerTokenScaleBrcb;
    AscendC::LocalTensor<float> ubPerTokenMul;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;
};

template <uint32_t UB_STAGES_, class TileShape_, class ElementSrc_, class ElementScale_, class ElementPerToken_,
          class ElementDst_, class TilePerTokenDequant_, class TileCopy_>
class BlockEpilogue<EpilogueAscend950PerTokenDequantTla<UB_STAGES_>, TileShape_, ElementSrc_, ElementScale_,
                    ElementPerToken_, ElementDst_, TilePerTokenDequant_, TileCopy_>
{
public:
    using DispatchPolicy = EpilogueAscend950PerTokenDequantTla<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;

    using ElementSrc = ElementSrc_;
    using ElementScale = ElementScale_;
    using LayoutScale = typename TileCopy::LayoutTagX;
    using ElementPerToken = ElementPerToken_;
    using LayoutPerToken = typename TileCopy::LayoutTagY;
    using ElementDst = ElementDst_;
    using LayoutDst = typename TileCopy::LayoutTagD;

    using TilePerTokenDequant = TilePerTokenDequant_;
    using TileShape = TileShape_;

    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr int16_t N_BASE_SIZE = static_cast<int16_t>(TileShape::COLUMN);
    static constexpr uint16_t CV_RATIO = 2;
    static constexpr uint32_t BLOCK_SIZE = TileShape::COUNT / CV_RATIO;

    static_assert(UB_STAGES * (BLOCK_SIZE * sizeof(ElementSrc) + TileShape::COLUMN / CV_RATIO * sizeof(ElementScale) +
                               TileShape::ROW / CV_RATIO * sizeof(ElementPerToken)) <=
                      ArchTag::UB_SIZE,
                  "TileShape is too large to fit in UB");

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, uint32_t &ubOffset)
    {
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenList[i] = resource.ubBuf.template GetBufferByByte<ElementPerToken>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerToken);

            eventUbScaleVMTE2List[i] = eventVMTE2++;
            eventUbScaleMTE2VList[i] = eventMTE2V++;
            eventUbPerTokenVMTE2List[i] = eventVMTE2++;
            eventUbPerTokenMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    template <class TensorDst, class TensorSrc, class TensorScale, class TensorPerToken>
    CATLASS_DEVICE void operator()(TensorDst gmDequantOut, TensorSrc ubGmmRes, TensorScale gmScale,
                                   TensorPerToken gmPerToken, uint32_t ubListId)
    {
        using CopyGmToUbScale = typename TileCopy::template CopyGmToUbX<TensorScale>;
        using CopyGmToUbPerToken = typename TileCopy::template CopyGmToUbY<TensorPerToken>;
        using CopyUbToGmDequant = typename TileCopy::template CopyUbToGmD<TensorDst>;

        CopyGmToUbScale copyGmToUbScale;
        CopyGmToUbPerToken copyGmToUbPerToken;
        CopyUbToGmDequant copyUbToGmDequant;
        TilePerTokenDequant tilePerTokenDequant;

        uint32_t m = tla::get<0>(ubGmmRes.shape());
        uint32_t n = tla::get<1>(ubGmmRes.shape());

        auto scaleLayout = tla::MakeLayout<ElementScale>(n);
        auto ubScale = tla::MakeTensor(ubScaleList[ubListId], scaleLayout, Arch::PositionUB{});

        auto perTokenLayout = tla::MakeLayout<ElementPerToken>(m);
        auto ubPerToken = tla::MakeTensor(ubPerTokenList[ubListId], perTokenLayout, Arch::PositionUB{});

        auto ubDequantOut = tla::MakeTensor(ubGmmRes.data().template ReinterpretCast<ElementDst>(), ubGmmRes.layout(),
                                            Arch::PositionUB{});

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[ubListId]);
        copyGmToUbScale(ubScale, gmScale);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenVMTE2List[ubListId]);
        copyGmToUbPerToken(ubPerToken, gmPerToken);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenMTE2VList[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenMTE2VList[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
        tilePerTokenDequant(ubDequantOut, ubGmmRes, ubScale, ubPerToken);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbScaleVMTE2List[ubListId]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbPerTokenVMTE2List[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
        copyUbToGmDequant(gmDequantOut, ubDequantOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
    }

private:
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];
    AscendC::LocalTensor<ElementPerToken> ubPerTokenList[UB_STAGES];

    int32_t eventUbScaleVMTE2List[UB_STAGES];
    int32_t eventUbScaleMTE2VList[UB_STAGES];
    int32_t eventUbPerTokenVMTE2List[UB_STAGES];
    int32_t eventUbPerTokenMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_TLA_HPP
