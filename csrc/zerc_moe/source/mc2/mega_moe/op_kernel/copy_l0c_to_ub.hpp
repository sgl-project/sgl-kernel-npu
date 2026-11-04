/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_UB_950_EXPAND_HPP
#define CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_UB_950_EXPAND_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "tla/tensor.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

namespace Catlass::Gemm::Tile {

template <class ArchTag, class TensorSrc, class TensorDst, CopyL0CToUBMode CopyMode = CopyL0CToUBMode::NO_SPLIT,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, uint32_t TargetUBIdx = 0,
          bool ReluEnable = false, class Enable = void>
struct CopyL0CToTargetUBTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to aiv one ub, can not find the specialization.");
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_,
          uint32_t TargetUBIdx_>
struct CopyL0CToTargetUBTla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT, ScaleGranularity::NO_QUANT, TargetUBIdx_, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;
    static constexpr auto targetUBIdx = TargetUBIdx_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor");

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.subBlockId = targetUBIdx;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(dstTensor.data()[dstOffset],
                                                                   srcTensor.data()[srcOffset], intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Tile

#endif
#endif  // CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_UB_950_EXPAND_HPP
