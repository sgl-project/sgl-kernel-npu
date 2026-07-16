/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_SPARSE_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_SPARSE_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm/tile/tile_copy.hpp"
#include "catlass_a5/gemm/tile/tile_mmad.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class ArchTag_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_
>
struct BlockMmadSparseTla <
    SparseMatmulMultiBlockOnKAxis<ArchTag_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_
> {
public:
    using DispatchPolicy = SparseMatmulMultiBlockOnKAxis<ArchTag_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy_::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy_::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy_::LayoutC;
    using ElementSparseIndex = uint8_t;
    using ElementAccumulator = typename TileCopy_::ElementAccumulator;

    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;

    using L1ALayout = typename TileCopy_::LayoutTagL1A;
    using L1BLayout = typename TileCopy_::LayoutTagL1B;
    using L1IdxLayout = typename TileCopy_::LayoutTagL1BIdx; // sparse index fractal 16*8
    using L0ALayout = detail::TagToLayout_t<ElementA, layout::zZ>;
    using L0BLayout = detail::TagToLayout_t<ElementB, layout::nZ>;
    using L0IdxLayout = detail::TagToLayout_t<uint8_t, layout::nZ>;

    using L1ATlaTensor = tla::Tensor<AscendC::LocalTensor<ElementA>, L1ALayout,
                                    tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using L1BTlaTensor = tla::Tensor<AscendC::LocalTensor<ElementB>, L1BLayout,
                                    tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    static constexpr uint16_t LIMIT_MNSIZE = 10;
    static constexpr uint16_t ALIGN_NUM = 16;
    static constexpr int32_t DENSE_MATRIX_B_OFFSET = 2;
    static constexpr int32_t INDEX_MATRIX_OFFSET = 8;
    static constexpr int32_t SHARE_LENS_COUNT = 3;
    static constexpr auto L1_M = tla::get<0>(L1Shape{});
    static constexpr auto L1_N = tla::get<1>(L1Shape{});
    static constexpr auto L1_K = tla::get<2>(L1Shape{});
    static constexpr auto L0_M = tla::get<0>(L0Shape{});
    static constexpr auto L0_N = tla::get<1>(L0Shape{});
    static constexpr auto L0_K = tla::get<2>(L0Shape{});
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_M * L1_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_N * L1_K * sizeof(ElementB) / DENSE_MATRIX_B_OFFSET;
    static constexpr uint32_t L1B_IDX_TILE_SIZE = L1_N * L1_K * sizeof(ElementB) / INDEX_MATRIX_OFFSET;
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_M * L0_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_K * L0_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_M * L0_N * sizeof(ElementAccumulator);

    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    static constexpr bool TRANS_A = tla::detail::isColumnMajor<LayoutA>::value;
    static constexpr bool TRANS_B = tla::detail::isColumnMajor<LayoutB>::value;

    // Check L1TileShape
    static_assert((L1A_TILE_SIZE + L1B_TILE_SIZE + L1B_IDX_TILE_SIZE) * STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1_M == L0_M && L1_N == L0_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_K <= L1_K, "L0TileShape::K cannot exceed L1TileShape::K");

    static_assert((std::is_same_v<ElementA, int8_t> && std::is_same_v<ElementB, int8_t> &&
                std::is_same_v<ElementC, int32_t>), "Unsupported dtype");
    static_assert(tla::detail::isColumnMajor<LayoutB>::value, "Only support MatrixB transpose");

    CATLASS_DEVICE
    BlockMmadSparseTla()
    {
        Init();
    }

    CATLASS_DEVICE
    ~BlockMmadSparseTla()
    {
        End();
    }

    /**
 * @brief Initialization function
 * 
 * This function initializes the buffers and parameters required for matrix multiplication operations
 */
    CATLASS_DEVICE
    void Init()
    {
        constexpr static int32_t aL0MatrixByteSize =
            L0_M * L0_K * sizeof(ElementA);
        constexpr static int32_t bL0MatrixByteSize =
            L0_N * L0_K * sizeof(ElementB);
        constexpr static int32_t cMatrixByteSize =
            L0_M * L0_N * sizeof(ElementAccumulator);

        constexpr static int dbL0AFlag = (aL0MatrixByteSize * STAGES > ArchTag::L0A_SIZE) ? 1 : STAGES;
        constexpr static int dbL0BFlag = (bL0MatrixByteSize * STAGES > ArchTag::L0B_SIZE) ? 1 : STAGES;

        uint32_t l1AOffset = 0;
        uint32_t l1BOffset = L1A_TILE_SIZE * STAGES;
        uint32_t l1BIdxOffset = l1BOffset + L1B_TILE_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList_[i] = l1Buf_[l1AOffset + L1A_TILE_SIZE * i].template ReinterpretCast<ElementA>();
            l1BTensorList_[i] = l1Buf_[l1BOffset + L1B_TILE_SIZE * i].template ReinterpretCast<ElementB>();
            l1BIdxTensorList_[i] = l1Buf_[l1BIdxOffset + L1B_IDX_TILE_SIZE * i].template ReinterpretCast<ElementSparseIndex>();
            l0ATensorList_[i] = l0ABuf_[L0A_PINGPONG_BUF_SIZE * i].template ReinterpretCast<ElementA>();
            l0BTensorList_[i] = l0BBuf_[L0B_PINGPONG_BUF_SIZE * i].template ReinterpretCast<ElementB>();

            // Assign event ID for each stages
            l1AEventList_[i] = i;
            l1BEventList_[i] = i + STAGES;
            l1BIdxEventList_[i] = i + STAGES * 2;
            l0AEventList_[i] = i;
            l0BEventList_[i] = i + STAGES;

            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BIdxEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
        }
        l0CTensor_ =  l0CBuf_.template ReinterpretCast<ElementAccumulator>();
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /**
 * @brief Perform the matrix multiplication operations
 * @param [in] DstTensor: destination tensor type
 * @param [in] SrcATensor: matrix A source tensor type
 * @param [in] SrcBTensor: matrix B source tensor type
 * @param [in] SparseIdxTensor: the type of the sparse index tensor
 * @param [in] Shape: shape type
 * @param [in] c: destination tensor
 * @param [in] a: matrix A source tensor
 * @param [in] b: matrix B source tensor
 * @param [in] sparseIdx: the sparse index tensor
 * @param [in] actualShape: actual shape
 */
    template <class DstTensor, class SrcATensor, class SrcBTensor, class SparseIdxTensor, class Shape>
    CATLASS_DEVICE
    void operator()(DstTensor& c, const SrcATensor& a, const SrcBTensor& b,
                                    const SparseIdxTensor& sparseIdx, const Shape& actualShape)
    {
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<DstTensor>;
        CopyL0CToGm copyL0CToGm;

        int32_t mIter = CeilDiv(tla::get<0>(actualShape), L0_M);
        int32_t nIter = CeilDiv(tla::get<1>(actualShape), L0_N);
        int32_t kOuterIter = CeilDiv(tla::get<2>(actualShape), L1_K);
        int32_t tailBaseM = (tla::get<0>(actualShape) % L0_M) == 0 ? L0_M : (tla::get<0>(actualShape) % L0_M);
        int32_t tailBaseN = (tla::get<1>(actualShape) % L0_N) == 0 ? L0_N : (tla::get<1>(actualShape) % L0_N);
        int32_t tailBaseK = (tla::get<2>(actualShape) % L0_K) == 0 ? L0_K : (tla::get<2>(actualShape) % L0_K);
        int32_t tailL1K = (tla::get<2>(actualShape) % L1_K) == 0 ? L1_K : (tla::get<2>(actualShape) % L1_K);

        for (auto mIndex = 0; mIndex < mIter; ++mIndex) {
            for (auto nIndex = 0; nIndex < nIter; ++nIndex) {
                int mPartActual = (mIndex + 1 == mIter) ? tailBaseM : L0_M;
                int nPartActual = (nIndex + 1 == nIter) ? tailBaseN : L0_N;
                auto l0CLayout = tla::MakeLayoutL0C(mPartActual, nPartActual);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                auto l0CTlaTensor = tla::MakeTensor(l0CTensor_, l0CLayout, Arch::PositionL0C{});

                for (auto kOuterIndex = 0; kOuterIndex < kOuterIter; ++kOuterIndex) {
                    uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
                    int l1K = (kOuterIndex + 1 == kOuterIter) ? tailL1K : L1_K;
                    // -----------------Step1: GM -> L1 -----------------
                    auto aTileHeight = mPartActual;
                    auto aTileWidth = l1K;
                    auto aL1Row = mIndex;
                    auto aL1Col = kOuterIndex;

                    auto l1ALayout = tla::MakeLayout<ElementA, L1ALayout>(aTileHeight, aTileWidth);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[l1ListIdNext]);
                    using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<SrcATensor>;

                    auto l1ATensor = tla::MakeTensor(l1ATensorList_[l1ListIdNext], l1ALayout, Arch::PositionL1{});
                    auto SrcATlaTensor = tla::MakeTensor(a.data(), a.layout(),
                                                        tla::MakeCoord(aL1Row * L0_M,
                                                                        aL1Col * L1_K),
                                                        Arch::PositionGM{});
                    CopyGmToL1A copyGmToL1A;
                    copyGmToL1A(l1ATensor, SrcATlaTensor);
                    
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList_[l1ListIdNext]);

                    auto bTileHeight = l1K;
                    auto bTileWidth = nPartActual;
                    auto bL1Row = kOuterIndex;
                    auto bL1Col = nIndex;
                    using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<SrcBTensor>;
                    using CopyGmToL1BIdx = typename TileCopy_::template CopyGmToL1BIdx<SparseIdxTensor>;
                    CopyGmToL1B copyGmToL1B;
                    CopyGmToL1BIdx copyGmToL1BIdx;

                    auto l1BLayout = tla::MakeLayout<ElementB, L1BLayout>(bTileHeight / DENSE_MATRIX_B_OFFSET, bTileWidth);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListIdNext]);

                    auto l1BTensor = tla::MakeTensor(l1BTensorList_[l1ListIdNext], l1BLayout, Arch::PositionL1{});
                    auto srcBTlaTensor = GetTile(b, tla::MakeCoord(bL1Row * L1_K / DENSE_MATRIX_B_OFFSET, bL1Col * L0_N),
                            tla::MakeShape(l1K / DENSE_MATRIX_B_OFFSET, nPartActual));

                    auto l1BIndexLayout =
                        tla::MakeLayout<int32_t, L1IdxLayout>(bTileHeight / INDEX_MATRIX_OFFSET, bTileWidth);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BIdxEventList_[l1ListIdNext]);

                    auto l1BIdxTensor = tla::MakeTensor(l1BIdxTensorList_[l1ListIdNext], l1BIndexLayout, Arch::PositionL1{});
                    auto sparseIdxTlaTensor = tla::MakeTensor(sparseIdx.data(), sparseIdx.layout(),
                                                            tla::MakeCoord(bL1Row * L1_K / INDEX_MATRIX_OFFSET, bL1Col * L0_N),
                                                            Arch::PositionGM{});

                    copyGmToL1B(l1BTensor, srcBTlaTensor);

                    copyGmToL1BIdx(l1BIdxTensor, sparseIdxTlaTensor);

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListIdNext]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BIdxEventList_[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList_[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BIdxEventList_[l1ListIdNext]);

                    int32_t kInnerIter =
                        (kOuterIndex + 1 == kOuterIter) ? CeilDiv(tailL1K, L0_K) : CeilDiv(L1_K, L0_K);
                    for (auto kInnerIndex = 0; kInnerIndex < kInnerIter; ++kInnerIndex) {
                        int baseK =
                            (kOuterIndex + 1 == kOuterIter) && (kInnerIndex + 1 == kInnerIter) ? tailBaseK : L0_K;

                        // -----------------Step2: L1 -> L0-----------------
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0AListId]);

                        auto aBaseHeight = mPartActual;
                        auto aBaseWidth = baseK;
                        auto aL0Row = mIndex;
                        auto aL0Col = kInnerIndex;
                        auto l0ALayout = tla::MakeLayout<ElementA, layout::zZ>(aBaseHeight, aBaseWidth);
                        auto l0ATensor = tla::MakeTensor(l0ATensorList_[l0AListId], l0ALayout, Arch::PositionL0A{});

                        auto l1ATileTensor = tla::MakeTensor(l1ATensor.data(), l1ATensor.layout(),
                                                        tla::MakeCoord(aL0Row * 0,
                                                                        aL0Col * L0_K),
                                                        Arch::PositionL1{});
                        copyL1ToL0A_(l0ATensor, l1ATileTensor);

                        auto bBaseHeight = baseK;
                        auto bBaseWidth = nPartActual;
                        auto bL0Row = kInnerIndex;
                        auto bL0Col = nIndex;
                        auto l0BLayout = tla::MakeLayout<ElementB, layout::nZ>(
                                        bBaseHeight / DENSE_MATRIX_B_OFFSET, bBaseWidth);

                        auto l0BTensor = tla::MakeTensor(l0BTensorList_[l0BListId], l0BLayout, Arch::PositionL0B{});
                        auto l1BTileTensor = tla::MakeTensor(l1BTensor.data(), l1BTensor.layout(),
                                                        tla::MakeCoord(bL0Row * L0_K / DENSE_MATRIX_B_OFFSET, 0),
                                                        Arch::PositionL1{});

                        copyL1ToL0B_(l0BTensor, l1BTileTensor, l1BIdxTensor);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[l0AListId]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList_[l0AListId]);

                        //  -----------------Step3: compute-----------------
                        AscendC::MmadParams mmadParams;
                        // GEMV is automatically enabled when setting M=1 in normal mode
                        mmadParams.m = (mPartActual == 1 ? ALIGN_NUM : mPartActual);
                        mmadParams.k = baseK;
                        mmadParams.n = nPartActual;
                        mmadParams.unitFlag = 0;
                        mmadParams.cmatrixInitVal = (kOuterIndex == 0) && (kInnerIndex == 0) ? true : false;
                        MmadWithSparse(l0CTensor_, l0ATensorList_[l0AListId], l0BTensorList_[l0BListId], mmadParams);
                        if ((mPartActual / ALIGN_NUM) * (nPartActual / ALIGN_NUM) < LIMIT_MNSIZE) {
                            AscendC::PipeBarrier<PIPE_M>();
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0AListId]);
                        l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[l1ListIdNext]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1ListIdNext]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BIdxEventList_[l1ListIdNext]);
                    l1ListId = l1ListIdNext;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                // -----------------Step4: L0C -> GM-----------------
                auto l0CSrcTensor = tla::MakeTensor(l0CTlaTensor.data(), l0CTlaTensor.layout(),
                                    tla::MakeCoord(mIndex * L0_M, nIndex * L0_N),
                                    Arch::PositionL0C{});
                copyL0CToGm(c, l0CSrcTensor);

                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            }
        }
    }

private:
    /**
     * @brief End function, release all events
     */
    CATLASS_DEVICE
    void End()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BIdxEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;

    AscendC::LocalTensor<uint8_t> l1Buf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, 0 , ArchTag::L1_SIZE)};
    AscendC::LocalTensor<uint8_t> l0ABuf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A2, 0 , ArchTag::L0A_SIZE)};
    AscendC::LocalTensor<uint8_t> l0BBuf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B2, 0 , ArchTag::L0B_SIZE)};
    AscendC::LocalTensor<uint8_t> l0CBuf_{AscendC::LocalTensor<uint8_t>(AscendC::TPosition::CO1, 0 , ArchTag::L0C_SIZE)};
    AscendC::LocalTensor<ElementA> l1ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList_[STAGES];
    AscendC::LocalTensor<ElementSparseIndex> l1BIdxTensorList_[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList_[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_;

    // Multi-stage event id list
    int32_t l1AEventList_[STAGES];
    int32_t l1BEventList_[STAGES];
    int32_t l1BIdxEventList_[STAGES];
    int32_t l0AEventList_[STAGES];
    int32_t l0BEventList_[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
};
} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_SPARSE_TLA_HPP