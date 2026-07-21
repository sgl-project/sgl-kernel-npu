/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_ASCEND950_COPY_L1_TO_L0B_HPP
#define CATLASS_CONV_TILE_ASCEND950_COPY_L1_TO_L0B_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Tile {

/// CopyL1ToL0BTla, Ascend950, CI1KHKWCOCI0 in and nZ out.
template <class Element>
struct CopyL1ToL0BTla {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0BTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        uint32_t cin1 = tla::get<0>(srcTensor.shape());
        uint32_t kh = tla::get<1>(srcTensor.shape());
        uint32_t kw = tla::get<2>(srcTensor.shape());
        uint32_t cout = tla::get<3>(srcTensor.shape());
        uint32_t coutOrg = tla::get<2>(srcTensor.stride()) / ELE_NUM_PER_C0;

        AscendC::LoadData2DParamsV2 loadDataParams;

        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilDiv<C0_NUM_PER_FRACTAL>(cout);
        loadDataParams.kStep = cin1 * kh * kw;
        loadDataParams.srcStride = CeilDiv<C0_NUM_PER_FRACTAL>(coutOrg);
        loadDataParams.dstStride = tla::get<1, 1>(dstTensor.shape());
        loadDataParams.ifTranspose = false;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
    }
};

}  // namespace Catlass::Conv::Tile

#endif  // CATLASS_CONV_TILE_ASCEND950_COPY_L1_TO_L0B_HPP
