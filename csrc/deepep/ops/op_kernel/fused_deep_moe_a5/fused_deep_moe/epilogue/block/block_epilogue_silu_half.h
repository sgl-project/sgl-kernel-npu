#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H

// Half-precision-aligned activation epilogues for the A5 (Ascend 950)
// FusedDeepMoe GMM1 stage: the fp32 GEMM accumulator tile is rounded through
// ElementI, the gate half (isLeft) is activated, and the up half only gets the
// precision alignment. The gate * up multiplication stays in the downstream
// quantize stage. Hosts the SiLU (EpilogueAtlasA5SiluHalf) and SiTU
// (EpilogueAtlasA5SituHalf) specializations, which share the MTE2 / V / MTE3
// pipeline skeleton in detail::BlockEpilogueActivationHalfBase and differ
// only in the activation functor (detail::SiluHalfAct / detail::SituHalfAct).

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../dispatch_policy.h"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"

namespace Catlass::Epilogue::Block {
namespace detail {

// ---------------------------------------------------------------------------
// Activation functors: the V-engine compute between the shared
// Cast(ubI, ubC, CAST_RINT) half-precision alignment and the V_MTE3 / V_MTE2
// release flags of the pipeline skeleton.
// Contract on entry: ubC = fp32 accumulator tile, ubI = ElementI-rounded copy
// of ubC, ubD = recycled output buffer of the current UB stage.
// Contract on exit: ubD = final fp32 result of this tile.
// Each functor owns the MTE3_V wait guarding its first overwrite of ubD (the
// previous tile's write-out must have released the buffer).
// ---------------------------------------------------------------------------

// SiLU: gate half x * sigmoid(x) = x / (1 + exp(-x)); up half passes through.
template <class ElementC, class ElementI, class ElementD>
struct SiluHalfAct {
    using LayoutC = typename layout::RowMajor;
    using LayoutD = typename layout::RowMajor;

    struct Params {
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrD;
        LayoutD layoutD;

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrD_, LayoutD layoutD_)
            : ptrC(ptrC_), layoutC(layoutC_), ptrD(ptrD_), layoutD(layoutD_)
        {}
    };

    using ElementCompute = ElementC;

    CATLASS_DEVICE void Compute(AscendC::LocalTensor<ElementC> &ubC, AscendC::LocalTensor<ElementI> &ubI,
                                AscendC::LocalTensor<ElementD> &ubD, uint32_t count, bool isLeft, int32_t eventUbDMTE3V,
                                Params const &params)
    {
        (void)params;  // SiLU is parameter-free; Params kept for the uniform act interface.
        if (isLeft) {
            Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);
            Muls(ubD, ubC, (ElementCompute)-1, count);
            AscendC::PipeBarrier<PIPE_V>();
            Exp(ubD, ubD, count);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(ubD, ubD, (ElementCompute)1, count);
            AscendC::PipeBarrier<PIPE_V>();
            Div(ubD, ubC, ubD, count);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);
            Cast(ubD, ubI, AscendC::RoundMode::CAST_NONE, count);
        }
    }
};

// SiTU (Kimi K3 soft-saturation gating): gate half
// beta * tanh(gate / beta) * sigmoid(gate); up half
// linear_beta * tanh(up / linear_beta) when hasLinearBeta, else passes through.
template <class ElementC, class ElementI, class ElementD>
struct SituHalfAct {
    struct Params {
        // Kimi K3 reference bounds.
        float beta{4.0F};
        float linearBeta{25.0F};
        bool hasLinearBeta{true};

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(float beta_, float linearBeta_ = 25.0F, bool hasLinearBeta_ = true)
            : beta(beta_), linearBeta(linearBeta_), hasLinearBeta(hasLinearBeta_)
        {}
    };

    using ElementCompute = ElementC;

    CATLASS_DEVICE void Compute(AscendC::LocalTensor<ElementC> &ubC, AscendC::LocalTensor<ElementI> &ubI,
                                AscendC::LocalTensor<ElementD> &ubD, uint32_t count, bool isLeft, int32_t eventUbDMTE3V,
                                Params const &params)
    {
        if (isLeft) {
            Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);

            // ubD = beta * tanh(gate / beta).
            Muls(ubD, ubC, static_cast<ElementCompute>(1.0F / params.beta), count);
            AscendC::PipeBarrier<PIPE_V>();
            // Tanh requires distinct source and destination tensors.
            Tanh(ubC, ubD, count);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(ubD, ubC, static_cast<ElementCompute>(params.beta), count);
            AscendC::PipeBarrier<PIPE_V>();

            // Restore gate and divide by 1 + exp(-gate) to apply sigmoid(gate).
            Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(ubC, ubC, static_cast<ElementCompute>(-1.0F), count);
            AscendC::PipeBarrier<PIPE_V>();
            Exp(ubC, ubC, count);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(ubC, ubC, static_cast<ElementCompute>(1.0F), count);
            AscendC::PipeBarrier<PIPE_V>();
            Div(ubD, ubD, ubC, count);
        } else if (params.hasLinearBeta) {
            Cast(ubC, ubI, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3V);

            // ubD = linear_beta * tanh(up / linear_beta).
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
};

// Runtime-selectable SiLU/SiTU activation used by fused_deep_moe.
template <class ElementC, class ElementI, class ElementD>
struct SiluSituHalfAct {
    struct Params {
        uint32_t activationType{ACTIVATION_SILU};
        float beta{4.0F};
        float linearBeta{25.0F};
        bool hasLinearBeta{true};

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(uint32_t activationType_, float beta_, float linearBeta_, bool hasLinearBeta_)
            : activationType(activationType_), beta(beta_), linearBeta(linearBeta_), hasLinearBeta(hasLinearBeta_)
        {}
    };

    CATLASS_DEVICE void Compute(AscendC::LocalTensor<ElementC> &ubC, AscendC::LocalTensor<ElementI> &ubI,
                                AscendC::LocalTensor<ElementD> &ubD, uint32_t count, bool isLeft, int32_t eventUbDMTE3V,
                                Params const &params)
    {
        if (params.activationType == ACTIVATION_SITU) {
            typename SituHalfAct<ElementC, ElementI, ElementD>::Params situParams(params.beta, params.linearBeta,
                                                                                  params.hasLinearBeta);
            SituHalfAct<ElementC, ElementI, ElementD> situ;
            situ.Compute(ubC, ubI, ubD, count, isLeft, eventUbDMTE3V, situParams);
        } else {
            typename SiluHalfAct<ElementC, ElementI, ElementD>::Params siluParams;
            SiluHalfAct<ElementC, ElementI, ElementD> silu;
            silu.Compute(ubC, ubI, ubD, count, isLeft, eventUbDMTE3V, siluParams);
        }
    }
};

// ---------------------------------------------------------------------------
// Shared MTE2 / V / MTE3 pipeline skeleton for the A5 half-aligned activation
// epilogues. Per epilogue tile: MTE2 loads the fp32 tile into ubC, V computes
// the activation into ubD (delegated to Act), MTE3 writes the result back to
// GM. Per-stage HardEvents keep the engines from overtaking each other on the
// recycled UB buffers; ubListId rotates across the UB_STAGES buffers.
// ---------------------------------------------------------------------------
template <class DispatchPolicy_, class ElementC_, class ElementI_, class ElementD_, class TileShape_, class Act_>
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
    using Act = Act_;
    using Params = typename Act::Params;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr uint32_t TILE_M = TileShape::ROW;
    static constexpr uint32_t TILE_N = TileShape::COLUMN;
    static constexpr uint32_t TILE_COUNT = TileShape::COUNT;
    static constexpr uint32_t ROW_ONCE = 64;

    using EpilogueTileSwizzle = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    // Check the element type of C
    static_assert(std::is_same_v<ElementC, float>, "Element type of C must be float");

    // Epilogue params definition
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

            // V-engine activation; the functor waits MTE3_V before its first
            // overwrite of ubD and leaves the final fp32 result in ubD.
            act_.Compute(ubC, ubI, ubD, count, isLeft, eventUbDMTE3VList[ubListId], params);

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
    Params params;
    Act act_;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementI> ubIList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];

    uint32_t ubListId{0};
};

}  // namespace detail

// SiLU epilogue: gate half x * sigmoid(x), up half precision-aligned.
template <uint32_t UB_STAGES_, class ElementC_, class ElementI_, class ElementD_, class TileShape_>
class BlockEpilogue<EpilogueAtlasA5SiluHalf<UB_STAGES_>, ElementC_, ElementI_, ElementD_, TileShape_>
    : public detail::BlockEpilogueActivationHalfBase<EpilogueAtlasA5SiluHalf<UB_STAGES_>, ElementC_, ElementI_,
                                                     ElementD_, TileShape_,
                                                     detail::SiluHalfAct<ElementC_, ElementI_, ElementD_>>
{
public:
    using Base =
        detail::BlockEpilogueActivationHalfBase<EpilogueAtlasA5SiluHalf<UB_STAGES_>, ElementC_, ElementI_, ElementD_,
                                                TileShape_, detail::SiluHalfAct<ElementC_, ElementI_, ElementD_>>;
    using Params = typename Base::Params;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<Arch::Ascend950> &resource) : Base(resource, Params()) {}
};

// SiTU epilogue: gate half beta * tanh(gate / beta) * sigmoid(gate), up half
// linear_beta * tanh(up / linear_beta) when hasLinearBeta, else precision-aligned.
// The multiplication of the two halves is performed by the existing post-epilogue stage.
template <uint32_t UB_STAGES_, class ElementC_, class ElementI_, class ElementD_, class TileShape_>
class BlockEpilogue<EpilogueAtlasA5SituHalf<UB_STAGES_>, ElementC_, ElementI_, ElementD_, TileShape_>
    : public detail::BlockEpilogueActivationHalfBase<EpilogueAtlasA5SituHalf<UB_STAGES_>, ElementC_, ElementI_,
                                                     ElementD_, TileShape_,
                                                     detail::SituHalfAct<ElementC_, ElementI_, ElementD_>>
{
public:
    using Base =
        detail::BlockEpilogueActivationHalfBase<EpilogueAtlasA5SituHalf<UB_STAGES_>, ElementC_, ElementI_, ElementD_,
                                                TileShape_, detail::SituHalfAct<ElementC_, ElementI_, ElementD_>>;
    using Params = typename Base::Params;

    static_assert(std::is_same_v<ElementD_, float>, "Element type of D must be float");

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<Arch::Ascend950> &resource) : Base(resource, Params()) {}

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<Arch::Ascend950> &resource, Params const &params_) : Base(resource, params_) {}
};

template <uint32_t UB_STAGES_, class ElementC_, class ElementI_, class ElementD_, class TileShape_>
class BlockEpilogue<EpilogueAtlasA5SiluSituHalf<UB_STAGES_>, ElementC_, ElementI_, ElementD_, TileShape_>
    : public detail::BlockEpilogueActivationHalfBase<EpilogueAtlasA5SiluSituHalf<UB_STAGES_>, ElementC_, ElementI_,
                                                     ElementD_, TileShape_,
                                                     detail::SiluSituHalfAct<ElementC_, ElementI_, ElementD_>>
{
public:
    using Base = detail::BlockEpilogueActivationHalfBase<EpilogueAtlasA5SiluSituHalf<UB_STAGES_>, ElementC_, ElementI_,
                                                         ElementD_, TileShape_,
                                                         detail::SiluSituHalfAct<ElementC_, ElementI_, ElementD_>>;
    using Params = typename Base::Params;

    static_assert(std::is_same_v<ElementD_, float>, "Element type of D must be float");

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<Arch::Ascend950> &resource, Params const &params_) : Base(resource, params_) {}
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_SILU_HALF_H
