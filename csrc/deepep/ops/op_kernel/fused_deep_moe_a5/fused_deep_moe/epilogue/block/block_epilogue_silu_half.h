#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H

// Half-precision-aligned activation epilogues for the A5 (Ascend 950)
// FusedDeepMoe GMM1 stage: the fp32 GEMM accumulator tile is rounded through
// ElementI, the gate half (isLeft) is activated, and the up half only gets the
// precision alignment. The gate * up multiplication stays in the downstream
// quantize stage. The runtime activation path is selected directly from
// params.activationType.

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../dispatch_policy.h"
#include "../../../fused_deep_moe_a5_tiling.h"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"

namespace Catlass::Epilogue::Block {

// ---------------------------------------------------------------------------
// Shared MTE2 / V / MTE3 pipeline skeleton for the A5 half-aligned activation
// epilogues. Per epilogue tile: MTE2 loads the fp32 tile into ubC, V computes
// the activation selected by activationType into ubD, MTE3 writes the result back to
// GM. Per-stage HardEvents keep the engines from overtaking each other on the
// recycled UB buffers; ubListId rotates across the UB_STAGES buffers.
// ---------------------------------------------------------------------------
template <class DispatchPolicy_, class ElementC_, class ElementI_, class ElementD_, class TileShape_>
class BlockEpilogueActivationHalfBase
{
public:
    // Type aliases
    using DispatchPolicy = DispatchPolicy_;
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

    // Check the element type of C
    static_assert(std::is_same_v<ElementC, float>, "Element type of C must be float");

    struct ActivationParams {
        uint32_t activationType;
        float beta;
        float linearBeta;
        bool hasLinearBeta;

        CATLASS_HOST_DEVICE
        ActivationParams() {}

        CATLASS_HOST_DEVICE
        ActivationParams(uint32_t activationType_, float beta_, float linearBeta_, bool hasLinearBeta_)
            : activationType(activationType_), beta(beta_), linearBeta(linearBeta_), hasLinearBeta(hasLinearBeta_)
        {}
    };
    using Params = ActivationParams;

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    BlockEpilogueActivationHalfBase(Arch::Resource<ArchTag> &resource, Params const &params_) : params(params_)
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
    ~BlockEpilogueActivationHalfBase()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    template <class TensorC, class TensorD>
    CATLASS_DEVICE void operator()(TensorC &tensorBlockC, TensorD &tensorBlockD, GemmCoord const &actualBlockShapeMNK,
                                   bool isLeft)
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }

        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();

        auto ubTileStride = static_cast<uint32_t>(TileShape::COLUMN);
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
            auto tensorSubBlockC = GetTile(tensorBlockC, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                                           tla::MakeShape(actualTileShape.row(), actualTileShape.column()));
            // build tensor C block in UB
            auto &ubC = ubCList[ubListId];
            auto layoutUbC = tla::MakeLayout(tla::MakeShape(actualTileShape.row(), actualTileShape.column()),
                                             tla::MakeStride(ubTileStride, tla::Int<1>{}));
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
            // Half-precision alignment: round the fp32 accumulator through
            // ElementI so both halves carry the same rounding as the real
            // half-stored inference data path.
            Cast(ubI, ubC, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();

            ComputeActivation(ubC, ubI, ubD, count, isLeft, eventUbDMTE3VList[ubListId]);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            // build tensor D block in GM
            auto tensorSubBlockD = GetTile(tensorBlockD, tla::MakeCoord(tileOffsetInBlockRow, tileOffsetInBlockColumn),
                                           tla::MakeShape(actualTileShape.row(), actualTileShape.column()));
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
    CATLASS_DEVICE void ComputeActivation(AscendC::LocalTensor<ElementC> &ubC, AscendC::LocalTensor<ElementI> &ubI,
                                          AscendC::LocalTensor<ElementD> &ubD, uint32_t count, bool isLeft,
                                          int32_t eventUbDMTE3V)
    {
        if (params.activationType == Cam::ACTIVATION_SITU) {
            ComputeSitu(ubC, ubI, ubD, count, isLeft, eventUbDMTE3V);
        } else {
            ComputeSilu(ubC, ubI, ubD, count, isLeft, eventUbDMTE3V);
        }
    }

    CATLASS_DEVICE void ComputeSilu(AscendC::LocalTensor<ElementC> &ubC, AscendC::LocalTensor<ElementI> &ubI,
                                    AscendC::LocalTensor<ElementD> &ubD, uint32_t count, bool isLeft,
                                    int32_t eventUbDMTE3V)
    {
        if (isLeft) {
            Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);
            Muls(ubD, ubC, static_cast<ElementCompute>(-1.0F), count);
            AscendC::PipeBarrier<PIPE_V>();
            Exp(ubD, ubD, count);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(ubD, ubD, static_cast<ElementCompute>(1.0F), count);
            AscendC::PipeBarrier<PIPE_V>();
            Div(ubD, ubC, ubD, count);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);
            Cast(ubD, ubI, AscendC::RoundMode::CAST_NONE, count);
        }
    }

    CATLASS_DEVICE void ComputeSitu(AscendC::LocalTensor<ElementC> &ubC, AscendC::LocalTensor<ElementI> &ubI,
                                    AscendC::LocalTensor<ElementD> &ubD, uint32_t count, bool isLeft,
                                    int32_t eventUbDMTE3V)
    {
        if (isLeft) {
            Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);

            Muls(ubD, ubC, static_cast<ElementCompute>(1.0F / params.beta), count);
            AscendC::PipeBarrier<PIPE_V>();
            Tanh(ubC, ubD, count);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(ubD, ubC, static_cast<ElementCompute>(params.beta), count);
            AscendC::PipeBarrier<PIPE_V>();

            Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(ubC, ubC, static_cast<ElementCompute>(-1.0F), count);
            AscendC::PipeBarrier<PIPE_V>();
            Exp(ubC, ubC, count);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(ubC, ubC, static_cast<ElementCompute>(1.0F), count);
            AscendC::PipeBarrier<PIPE_V>();
            Div(ubD, ubD, ubC, count);
        } else {
            if (params.hasLinearBeta) {
                Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);

                Muls(ubD, ubC, static_cast<ElementCompute>(1.0F / params.linearBeta), count);
                AscendC::PipeBarrier<PIPE_V>();
                Tanh(ubC, ubD, count);
                AscendC::PipeBarrier<PIPE_V>();
                Muls(ubD, ubC, static_cast<ElementCompute>(params.linearBeta), count);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);
                Cast(ubD, ubI, AscendC::RoundMode::CAST_NONE, count);
            }
        }
    }

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

template <uint32_t UB_STAGES_, class ElementC_, class ElementI_, class ElementD_, class TileShape_>
class BlockEpilogue<EpilogueAtlasA5ActivationHalf<UB_STAGES_>, ElementC_, ElementI_, ElementD_, TileShape_>
    : public BlockEpilogueActivationHalfBase<EpilogueAtlasA5ActivationHalf<UB_STAGES_>, ElementC_, ElementI_, ElementD_,
                                             TileShape_>
{
public:
    using Base = BlockEpilogueActivationHalfBase<EpilogueAtlasA5ActivationHalf<UB_STAGES_>, ElementC_, ElementI_,
                                                 ElementD_, TileShape_>;
    using Params = typename Base::Params;

    static_assert(std::is_same_v<ElementD_, float>, "Element type of D must be float");

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<Arch::Ascend950> &resource, Params const &params_) : Base(resource, params_) {}
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H
