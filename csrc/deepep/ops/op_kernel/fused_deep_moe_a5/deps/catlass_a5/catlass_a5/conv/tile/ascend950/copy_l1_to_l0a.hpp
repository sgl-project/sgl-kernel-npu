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

#ifndef CATLASS_CONV_TILE_ASCEND950_COPY_L1_TO_L0A_HPP
#define CATLASS_CONV_TILE_ASCEND950_COPY_L1_TO_L0A_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Tile {

/// CopyL1ToL0ATla, Ascend950, NC1HWC0 in and zN out.
template <class Element>
struct CopyL1ToL0ATla {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    Conv2dFilterParams params;

    // Methods
    CATLASS_DEVICE
    CopyL1ToL0ATla(const Conv2dFilterParams &params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t *blockPadList)
    {
        uint32_t hiActual = tla::get<2>(srcTensor.shape());
        uint32_t wiActual = tla::get<3>(srcTensor.shape());
        uint32_t mPartRound = tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        uint32_t kPartActual = tla::get<1, 0>(dstTensor.shape()) * tla::get<1, 1>(dstTensor.shape());
        uint32_t cin1L0Actual = kPartActual / (params.kh() * params.kw() * ELE_NUM_PER_C0);

        AscendC::LoadData3DParamsV2<Element> loadDataParams;

        loadDataParams.padList[0] = blockPadList[0];
        loadDataParams.padList[1] = blockPadList[1];
        loadDataParams.padList[2] = blockPadList[2];
        loadDataParams.padList[3] = blockPadList[3];
        loadDataParams.l1H = static_cast<uint16_t>(hiActual);  // 源操作数 height
        loadDataParams.l1W = static_cast<uint16_t>(wiActual);  // 源操作数 width
        loadDataParams.channelSize =
            static_cast<uint16_t>(cin1L0Actual * ELE_NUM_PER_C0);  // 源操作数的通道数(channelSize为 4, 8, N*16,
                                                                   // N*16+4, N*16+8)
        loadDataParams.kExtension = static_cast<uint16_t>(kPartActual);  // 目的操作数Width维度的传输长度(c0size对齐)
        loadDataParams.mExtension = static_cast<uint16_t>(mPartRound);  // 目的操作数height维度的传输长度(16的倍数)
        loadDataParams.kStartPt = 0;  // 目的操作数Width维度的起点(c0size对齐)
        loadDataParams.mStartPt = 0;  // 目的操作数height维度的起点(16的倍数)
        loadDataParams.strideW = params.strideW();
        loadDataParams.strideH = params.strideH();
        loadDataParams.filterW = params.kw();
        loadDataParams.filterH = params.kh();
        loadDataParams.dilationFilterW = params.dilationW();
        loadDataParams.dilationFilterH = params.dilationH();
        loadDataParams.enTranspose = false;
        loadDataParams.enSmallK = false;
        loadDataParams.padValue = (half)(0);  // Pad填充值的数值

        uint16_t dstStride = CeilDiv<C0_NUM_PER_FRACTAL>(mPartRound);
        AscendC::SetLoadDataRepeatWithStride({0, 1, 0, dstStride});

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadDataWithStride(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
    }
};

}  // namespace Catlass::Conv::Tile

#endif  // CATLASS_CONV_TILE_ASCEND950_COPY_L1_TO_L0A_HPP
