/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_BLOCK_QUANT_TLA_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_BLOCK_QUANT_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "catlass_a5/epilogue/tile/tile_perblock_quant.hpp"
#include "catlass_a5/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Epilogue::Block {

template <class ElementSrc_, class ElementDst_, class ElementScale_, class TilePerBlockQuant_>
class BlockEpilogue<EpilogueAscend950PerBlockQuantTla<1>, ElementSrc_, ElementDst_, ElementScale_, TilePerBlockQuant_>
{
public:
    using DispatchPolicy = EpilogueAscend950PerBlockQuantTla<1>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = 1;

    using ElementSrc = ElementSrc_;
    using LayoutSrc = detail::TagToLayout_t<ElementSrc, layout::RowMajor>;
    using ElementDst = ElementDst_;
    using LayoutDst = detail::TagToLayout_t<ElementDst, layout::RowMajor>;
    using ElementScale = ElementScale_;
    using LayoutScale = detail::TagToLayout_t<ElementScale, layout::VectorLayout>;

    static_assert(std::is_same_v<ElementSrc, bfloat16_t> && (std::is_same_v<ElementDst, float8_e4m3_t>) &&
                      std::is_same_v<ElementScale, float>,
                  "The element type template parameters of BlockEpilogue are wrong");

    using TilePerBlockQuant = TilePerBlockQuant_;
    using TensorDst =
        tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorUbDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, tla::Coord<tla::_0, tla::_0>,
                                    AscendC::TPosition::VECCALC>;
    using CopyUbToGmDst = Epilogue::Tile::CopyUb2GmTla<ArchTag, TensorUbDst, TensorDst>;

    static constexpr uint32_t FLOAT_ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(float);
    static constexpr uint32_t VL_SIZE = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementSrc);

    struct Params {
        LayoutSrc layoutSrc{};
        LayoutDst layoutDst{};
        LayoutScale layoutScale{};
        uint32_t strideC{0};

        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(LayoutSrc const &layoutSrc_, LayoutDst const &layoutDst_, LayoutScale const &layoutScale_,
               uint32_t strideC_)
            : layoutSrc(layoutSrc_), layoutDst(layoutDst_), layoutScale(layoutScale_), strideC(strideC_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        nLoops = AscendC::CeilDivision(params.strideC, VL_SIZE);
    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
        nLoops = AscendC::CeilDivision(params.strideC, VL_SIZE);
    }

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementSrc> ubC, AscendC::LocalTensor<ElementDst> ubDst,
                    AscendC::LocalTensor<ElementScale> ubScale, AscendC::GlobalTensor<ElementDst> gmDst)
    {
        auto tensorUbC = tla::MakeTensor(ubC, params.layoutSrc, Arch::PositionUB{});
        auto tensorUbDst = tla::MakeTensor(ubDst, params.layoutDst, Arch::PositionUB{});
        auto tensorUbScale = tla::MakeTensor(ubScale, params.layoutScale, Arch::PositionUB{});
        tilePerBlockQuant(tensorUbC, tensorUbDst, tensorUbScale, nLoops);
        auto tensorDst = tla::MakeTensor(gmDst, params.layoutDst, Arch::PositionGM{});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        copyUbToGmDst(tensorDst, tensorUbDst);
    }

private:
    Params params;
    uint16_t nLoops{};
    TilePerBlockQuant tilePerBlockQuant;
    CopyUbToGmDst copyUbToGmDst;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_BLOCK_QUANT_TLA_HPP
