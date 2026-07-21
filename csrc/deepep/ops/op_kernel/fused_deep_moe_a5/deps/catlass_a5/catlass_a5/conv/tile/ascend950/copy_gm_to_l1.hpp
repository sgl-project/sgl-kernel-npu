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

#ifndef CATLASS_CONV_TILE_ASCEND950_COPY_GM_TO_L1_HPP
#define CATLASS_CONV_TILE_ASCEND950_COPY_GM_TO_L1_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/gemm_type.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Conv::Tile {

/// CopyGmToL1ATla, Ascend950, NC1HWC0 in and NC1HWC0 out.
template <class Element>
struct CopyGmToL1ATla {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1ATla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        uint32_t cin1Actual = tla::get<1>(srcTensor.shape());
        uint32_t hiActual = tla::get<2>(srcTensor.shape());
        uint32_t wiActual = tla::get<3>(srcTensor.shape());
        uint32_t srcStrideCin1 = tla::get<1>(srcTensor.stride());  // Hi * Wi * ELE_NUM_PER_C0
        uint32_t dstStrideCin1 = tla::get<1>(dstTensor.stride());  // Hi * Wi * ELE_NUM_PER_C0
        uint32_t wiOrg = tla::get<2>(srcTensor.stride()) / ELE_NUM_PER_C0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        for (int cin1Idx = 0; cin1Idx < cin1Actual; cin1Idx++) {
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset],
                              {
                                  static_cast<uint16_t>(hiActual),  // blockCount 连续传输数据块个数
                                  static_cast<uint16_t>(wiActual),  // blockLen 每个连续传输数据块长度(32Bytes)
                                  static_cast<uint16_t>(wiOrg - wiActual),  // srcStride 相邻连续数据块的间隔(32Bytes)
                                  0  // dstStride 相邻连续数据块间的间隔(32Bytes)
                              });
            srcOffset += srcStrideCin1;
            dstOffset += dstStrideCin1;
        }
    }
};

/// CopyGmToL1BTla, Ascend950, CI1KHKWCOCI0 in and CI1KHKWCOCI0 out.
template <class Element>
struct CopyGmToL1BTla {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1BTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        uint32_t cin1Actual = tla::get<0>(srcTensor.shape());
        uint32_t KhKw = tla::get<1>(srcTensor.shape()) * tla::get<2>(srcTensor.shape());
        uint32_t coutActual = tla::get<3>(srcTensor.shape());
        uint32_t coutRound = tla::get<3>(dstTensor.shape());
        uint32_t coutOrg = tla::get<2>(srcTensor.stride()) / ELE_NUM_PER_C0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset],
                          AscendC::DataCopyParams(cin1Actual * KhKw,  // blockCount 连续传输数据块个数
                                                  coutActual,         // blockLen 每个连续传输数据块长度
                                                  coutOrg - coutActual,  // 源操作数，相邻连续数据块的间隔(32Bytes)
                                                  coutRound - coutActual  // 目的操作数，相邻连续数据块间的间隔(32Bytes)
                                                  ));
    }
};

}  // namespace Catlass::Conv::Tile

#endif  // CATLASS_CONV_TILE_ASCEND950_COPY_GM_TO_L1_HPP
