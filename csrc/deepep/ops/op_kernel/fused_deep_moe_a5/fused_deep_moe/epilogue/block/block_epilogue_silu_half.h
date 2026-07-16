/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H

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

template <uint32_t UB_STAGES_, class ElementC_, class ElementI_, class ElementD_, class TileShape_>
class BlockEpilogue <
    EpilogueAtlasA5SiluHalf<UB_STAGES_>,
    ElementC_,
    ElementI_,
    ElementD_,
    TileShape_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasA5SiluHalf<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementC = ElementC_;
    using LayoutC = typename layout::RowMajor;
    using ElementI = ElementI_;
    using ElementD = ElementD_;
    using LayoutD = typename layout::RowMajor;

    using ElementCompute = ElementC;
    using TileShape = TileShape_;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr uint32_t TILE_M = TileShape::ROW;
    static constexpr uint32_t TILE_N = TileShape::COLUMN;
    static constexpr uint32_t TILE_COUNT = TileShape::COUNT;
    static constexpr uint32_t ROW_ONCE = 64;

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


    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        uint32_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubIList[i] = resource.ubBuf.template GetBufferByByte<ElementI>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementI);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
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

    template <class TensorC, class TensorD>
    CATLASS_DEVICE
    void operator() (
        TensorC &tensorBlockC,
        TensorD &tensorBlockD,
        GemmCoord const &actualBlockShapeMNK,
        bool isLeft
    )
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }

        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();

        auto ubTileStride = static_cast<uint32_t>(TileShape::COLUMN);
        auto ubTileStrideRow = static_cast<uint32_t>(TileShape::ROW);
        auto tileShape = MakeCoord(TileShape::ROW, TileShape::COLUMN);
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

            auto &ubI = ubIList[ubListId];
            auto &ubD = ubDList[ubListId];
            Cast(ubI, ubC, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            if (isLeft) {
                Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                Muls(ubD, ubC, (ElementCompute)-1, count);
                AscendC::PipeBarrier<PIPE_V>();
                Exp(ubD, ubD, count);
                AscendC::PipeBarrier<PIPE_V>();
                Adds(ubD, ubD, (ElementCompute)1, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                Div(ubD, ubC, ubD, count);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                Cast(ubD, ubI, AscendC::RoundMode::CAST_NONE, count);
            }
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
            copyUbToGmD(tensorSubBlockD, tensorUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementI> ubIList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};

};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H