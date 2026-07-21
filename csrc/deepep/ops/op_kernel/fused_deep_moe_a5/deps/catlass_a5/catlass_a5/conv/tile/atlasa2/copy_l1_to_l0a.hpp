/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_ATLASA2_COPY_L1_TO_L0A_HPP
#define CATLASS_CONV_TILE_ATLASA2_COPY_L1_TO_L0A_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Tile {

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Catlass::Gemm::GemmType<Element, layout::NC1HWC0, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::NC1HWC0;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    Conv2dFilterParams params;

    CATLASS_DEVICE
    CopyL1ToL0A(const Conv2dFilterParams &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor,  // (ho, wo, cin1, Kh, Kw, C0)
                    AscendC::LocalTensor<Element> srcTensor,  // (cin1, hi, wi, C0)
                    LayoutDst const &layoutDst,               // {mPartRound, kPartActual} rowsInFractal, rowsByFractal,
                                                              // colsInFractal, colsByFractal
                    LayoutSrc const &layoutSrc,  // {1, FmapL1TileShape::Cin1, hiActual, wiActual, ELE_NUM_A_PER_C0}
                    uint8_t *blockPadList)
    {
        uint32_t hiActual = layoutSrc.shape(2);
        uint32_t wiActual = layoutSrc.shape(3);
        uint32_t mPartRound = layoutDst.orgShape(0);
        uint32_t kPartActual = layoutDst.orgShape(1);
        uint32_t cin1L0Actual = kPartActual / (params.kh() * params.kw() * ELE_NUM_PER_C0);

        AscendC::LoadData(
            dstTensor, srcTensor,
            {
                // load3dv2
                blockPadList,                                          // {padLeft, padRight, padTop, padBottom}
                static_cast<uint16_t>(hiActual),                       // 源操作数 height
                static_cast<uint16_t>(wiActual),                       // 源操作数 width
                static_cast<uint16_t>(cin1L0Actual * ELE_NUM_PER_C0),  // 源操作数的通道数(channelSize为 4, 8, N*16,
                                                                       // N*16+4, N*16+8)
                static_cast<uint16_t>(kPartActual),  // 目的操作数Width维度的传输长度(16的倍数)
                static_cast<uint16_t>(mPartRound),   // 目的操作数height维度的传输长度(16的倍数)
                0,                                   // 目的操作数Width维度的起点
                0,                                   // 目的操作数height维度的起点
                params.strideW(), params.strideH(), params.kw(), params.kh(), params.dilationW(), params.dilationH(),
                false,     // 是否启用转置
                false,     // 是否使能small k特性
                (half)(0)  // Pad填充值的数值
            });
    }
};

///////////////////////////////////////////TileCopyTla//////////////////////////////////////////////////////
/// CopyL1ToL0ATla, NC1HWC0 in and zN out.
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

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::LoadData(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], loadDataParams);
    }
};

}  // namespace Catlass::Conv::Tile

#endif  // CATLASS_CONV_TILE_ATLASA2_COPY_L1_TO_L0A_HPP
