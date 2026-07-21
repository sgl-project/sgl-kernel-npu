/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_MMAD_HPP
#define CATLASS_GEMM_TILE_TILE_MMAD_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "tla_a5/tensor.hpp"
namespace Catlass::Gemm::Tile {

///////////////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // Methods

    CATLASS_DEVICE
    TileMmad() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
                    AscendC::LocalTensor<ElementA> const &l0ATensor, AscendC::LocalTensor<ElementB> const &l0BTensor,
                    uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> &&
                      (std::is_same_v<typename AType_::Layout, layout::ColumnMajor> ||
                       std::is_same_v<typename AType_::Layout, layout::nZ>)) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if constexpr (std::is_same_v<typename AType_::Layout, layout::VectorLayout>) {
            mmadParams.disableGemv = false;
        } else {
            mmadParams.disableGemv = true;
        }
#endif

        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
                    AscendC::LocalTensor<ElementA> const &l0ATensor, AscendC::LocalTensor<ElementB> const &l0BTensor,
                    AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor, uint32_t m, uint32_t n, uint32_t k,
                    bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        using LayoutL1A = typename helper::L1ATypeSelector<AType_>::L1AType::Layout;
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif

        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

///////////////////////////////////////////TileMmadTla/////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag,
    /// Element for A matrix operand
    class ElementA,
    /// LayoutTag for A matrix operand in L1
    class LayoutTagL1A>
struct TileMmadTla {
    // Methods

    CATLASS_DEVICE
    TileMmadTla() {}

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(TensorC const &l0CTensor, TensorA const &l0ATensor, TensorB const &l0BTensor,
                                   uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if constexpr (std::is_same_v<LayoutTagL1A, layout::VectorLayout>) {
            mmadParams.disableGemv = false;
        } else {
            mmadParams.disableGemv = true;
        }
#endif

        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB, class TensorBias>
    CATLASS_DEVICE void operator()(TensorC const &l0CTensor, TensorA const &l0ATensor, TensorB const &l0BTensor,
                                   TensorBias const &l0BiasTensor, uint32_t m, uint32_t n, uint32_t k,
                                   bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif

        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), l0BiasTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(TensorC const &l0CTensor, TensorA const &l0ATensor, TensorB const &l0BTensor,
                                   uint32_t m, uint32_t n, uint32_t k, uint32_t l0Batch)
    {
        const uint32_t L0AM = tla::get<0, 0>(l0ATensor.shape()) * tla::get<0, 1>(l0ATensor.shape());
        const uint32_t L0AK = tla::get<1, 0>(l0ATensor.shape()) * tla::get<1, 1>(l0ATensor.shape());
        const uint32_t L0BK = tla::get<0, 0>(l0BTensor.shape()) * tla::get<0, 1>(l0BTensor.shape());
        const uint32_t L0BN = tla::get<1, 0>(l0BTensor.shape()) * tla::get<1, 1>(l0BTensor.shape());
        const uint32_t L0CM = tla::get<0, 0>(l0CTensor.shape()) * tla::get<0, 1>(l0CTensor.shape());
        const uint32_t L0CN = tla::get<1, 0>(l0CTensor.shape()) * tla::get<1, 1>(l0CTensor.shape());

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        mmadParams.disableGemv = true;
#endif
        for (uint32_t l0BatchIdx = 0; l0BatchIdx < l0Batch; l0BatchIdx++) {
            AscendC::Mmad(l0CTensor.data()[l0BatchIdx * L0CM * L0CN], l0ATensor.data()[l0BatchIdx * L0AM * L0AK],
                          l0BTensor.data()[l0BatchIdx * L0BK * L0BN], mmadParams);
        }
    }

    // automatically extracts actual sizes from tensor originShape
    template <class TensorC, class TensorA, class TensorB>
    CATLASS_DEVICE void operator()(TensorC const &l0CTensor, TensorA const &l0ATensor, TensorB const &l0BTensor,
                                   bool initC = true, uint8_t unitFlag = 0)
    {
        // Get actual sizes from tensor originShape
        uint32_t m = tla::get<0>(l0CTensor.layout().originShape());
        uint32_t n = tla::get<1>(l0CTensor.layout().originShape());
        uint32_t k = tla::get<1>(l0ATensor.layout().originShape());
        AscendC::MmadParams mmadParams;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201)
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<LayoutTagL1A, layout::nZ>) {
            mmadParams.kDirectionAlign = true;
        }
        if constexpr (!std::is_same_v<LayoutTagL1A, layout::VectorLayout>) {
            if (m == 1) m = 16;  // avoid gemv mode
        }
#endif
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
        if constexpr (std::is_same_v<LayoutTagL1A, layout::VectorLayout>) {
            mmadParams.disableGemv = false;
        } else {
            mmadParams.disableGemv = true;
        }
#endif
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;

        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Gemm::Tile

#endif  // CATLASS_GEMM_TILE_TILE_MMAD_HPP
