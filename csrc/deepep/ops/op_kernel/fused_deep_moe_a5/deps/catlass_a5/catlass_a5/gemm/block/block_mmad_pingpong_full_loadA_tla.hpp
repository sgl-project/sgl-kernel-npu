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

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_FULL_LOADA_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_FULL_LOADA_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class ArchTag_,
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    bool USE_HF32_MODE_,
    uint32_t L0C_STAGES_,
    bool ENABLE_L1_RESIDENT_,
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadTla <
    MmadFullLoadA<ArchTag_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_, USE_HF32_MODE_, L0C_STAGES_, ENABLE_L1_RESIDENT_, L1A_STAGES_, 
        L1B_STAGES_, L0A_STAGES_, L0B_STAGES_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadFullLoadA<ArchTag_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_, USE_HF32_MODE_, L0C_STAGES_, ENABLE_L1_RESIDENT_, 
        L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr bool USE_HF32_MODE = DispatchPolicy::USE_HF32_MODE;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = ArchTag::L1_SIZE / 2;
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // Check HF32_MODE
    static_assert(
        !USE_HF32_MODE || (USE_HF32_MODE && std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>),
        "HF32 MODE only supports in float!"
    );

    // Check L0C_STAGES
    static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1), "L0C_STAGES must be 1 when UnitFlag is true!");

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                          std::is_same_v<ElementC, float>) && tla::detail::iszN<ElementC, LayoutC>::value),
        "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // Check L1TileShape
    static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");

    static_assert((!HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 8) || (HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 7), 
        "L1 Buffer overflow: Exceeds the supported range of EVENT(0~7)");

    static_assert((!HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 8) || (HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 7), 
        "L0 Buffer overflow: Exceeds the supported range of EVENT_ID(0~7)");

    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    // When enableing L1 resident mode, restore the pointer and coordinates that record the last state
    // to the initial state. if tow blockmmad instances need to be consecutively invoked at the kernel layer,
    // RestoreStatus() must be inserted between them.
    CATLASS_DEVICE
    void RestoreStatus()
    {
        for (int i = 0; i < L1A_STAGES; ++i) {
            lastAddrA[i] = nullptr;
            lastCoordA[i] = MatrixCoord{0U, 0U};
        }
        for (int i = 0; i < L1B_STAGES; ++i) {
            lastAddrB[i] = nullptr;
            lastCoordB[i] = MatrixCoord{0U, 0U};
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        if ASCEND_IS_AIC {
            // use HF32 when USE_HF32_MODE is true
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(true);
            } else {
                AscendC::SetHF32Mode(false);
            }
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(true);
            }
            uint32_t l1AOffset = l1BufAddrStart;
            uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1A_STAGES;
            // Init buffers
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
                // Assign event ID for each stages
                l1AEventList[i] = i;
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
                // Assign event ID for each stages
                l1BEventList[i] = i + L1A_STAGES;
                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
                // Assign event ID for each stages
                l0AEventList[i] = i;
                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
                // Assign event ID for each stages
                l0BEventList[i] = i + L0A_STAGES;
                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            if constexpr(!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
                    l0CEventList[i] = i;
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
                }
            } else {
                l0CTensorList[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            }
            if constexpr (HAS_BIAS) {
                uint32_t l1BiasOffset = l1BOffset + L1B_TILE_SIZE * L1B_STAGES;
                l1BiasTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(l1BiasOffset);
                l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }

            if constexpr (ENABLE_L1_RESIDENT) {
                RestoreStatus();
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(false);
            }
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(false);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            if constexpr(!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
                }
            }
            if constexpr (HAS_BIAS) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorB, class TensorC, class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, GemmCoord const &actualShape,
        bool needLoadL1, TensorBias const &tensorBias = {})
    {
        // Check L1TileShape
        if constexpr (HAS_BIAS) {
            static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
            static constexpr uint32_t L1_BIAS_SIZE = L1_TILE_N * sizeof(ElementBias);
            static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE,
                "BIAS_BUF_SIZE exceeding the BT space! Reduce L0_TILE_N");
            static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES + L1_BIAS_SIZE <= ArchTag::L1_SIZE,
                "L1TileShape exceeding the L1 space!");
        }

        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL0CToGm copyL0CToGm;

        // load first matrix A tile from GM to L1
        auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, actualShape.k());
        if (needLoadL1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[0]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[0], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorTileA = GetTileA(tensorA, 0, 0, actualShape.m(), actualShape.k());
            copyGmToL1A(tensorL1A, tensorTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]);
        }

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            // Avoid using the gemv mode in mmad
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        // uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);
        uint32_t kTileCount = CeilDiv<L1_TILE_K>(kBlockActual);
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
        uint32_t kL1Actual =
            (firstTileIdx < kTileCount - 1) ? L1_TILE_K : (kBlockActual - firstTileIdx * L1_TILE_K);

        // load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(firstTileIdx * L1_TILE_K, 0), tla::MakeShape(kL1Actual, nBlockActual));
        if constexpr (ENABLE_L1_RESIDENT) {
            if (lastAddrB[l1BListId] != tensorTileB.data().GetPhyAddr()
                || tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListId].row()
                || tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListId].column()) {
                copyGmToL1B(tensorL1B, tensorTileB);
                lastCoordB[l1BListId] = MatrixCoord{tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                lastAddrB[l1BListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                    tensorTileB.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1B(tensorL1B, tensorTileB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
            using CopyGmToL1Bias = typename TileCopy::template CopyGmToL1Bias<TensorBias>;
            CopyGmToL1Bias copyGmToL1Bias;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
            auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
            copyGmToL1Bias(tensorL1Bias, tensorBias);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        }

        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mL1Actual);
        uint32_t nL0Loop = CeilDiv<L0_TILE_N>(nL1Actual);

        // main loop
        for (uint32_t kL1Idx = 0; kL1Idx < kTileCount; kL1Idx++) {
            uint32_t shuffleKIdx = (startTileIdx + kL1Idx) % kTileCount;
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
            uint32_t kL1ActualNext{0};
            // preload next tile from GM to L1
            if (kL1Idx < kTileCount - 1) {
                uint32_t kL1IdxNext = (startTileIdx + kL1Idx + 1) % kTileCount;
                kL1ActualNext = (kL1IdxNext < kTileCount - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);

                // Get L1 tensor for next stage
                auto l1BTensor = l1BTensorList[l1BListIdNext];
                auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
                // Get GM tile for next stage
                auto tensorTileB = GetTile(tensorB, tla::MakeCoord(kL1IdxNext * L1_TILE_K, 0),
                    tla::MakeShape(kL1ActualNext, nBlockActual));

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                if constexpr (ENABLE_L1_RESIDENT) {
                    if (lastAddrB[l1BListIdNext] != tensorTileB.data().GetPhyAddr()
                        || tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListIdNext].row()
                        || tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListIdNext].column()) {
                        copyGmToL1B(tensorL1B, tensorTileB);
                        lastCoordB[l1BListIdNext] =
                            MatrixCoord{tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                        lastAddrB[l1BListIdNext] =
                            const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                                tensorTileB.data().GetPhyAddr()
                            );
                    }
                } else {
                    copyGmToL1B(tensorL1B, tensorTileB);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[0];
            auto l1BTensor = l1BTensorList[l1BListId];
            auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
            tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
            // Get the loop nums on L0
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

            for (int mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mL1Actual - mL0Idx * L0_TILE_M);

                for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mL0Actual, kL0Actual);
                    auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
                    // Locate the current tile of matrix A on L1
                    auto tensorTileL1A = GetTileA(tensorL1A, mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K + shuffleKIdx * L1_TILE_K,
                                                  mL0Actual, kL0Actual);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    bool initC = ((kL1Idx == 0) && (kL0Idx == 0));
                    for (int nL0Idx = 0; nL0Idx < nL0Loop; nL0Idx++) {
                        uint32_t nL0Actual = (nL0Idx < nL0Loop - 1) ? L0_TILE_N : (nL1Actual - nL0Idx * L0_TILE_N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL0Actual);
                        auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
                        // Locate the current tile of matrix B on L1
                        auto tensorTileL1B = GetTile(tensorL1B,
                                                     tla::MakeCoord(kL0Idx * L0_TILE_K, nL0Idx * L0_TILE_N),
                                                     tla::MakeShape(kL0Actual, nL0Actual));

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((mL0Idx == 0) && (kL0Idx == 0) && (nL0Idx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(tensorL0B, tensorTileL1B);
                        
                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                        }

                        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                            if (initC) {
                                if (nL0Idx == 0) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                                auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
                                auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
                                auto tensorTileL1Bias = GetTile(tensorL1Bias,
                                                                tla::MakeCoord(nL0Idx * L0_TILE_N),
                                                                tla::MakeShape(nL0Actual));
                                // Load bias to l0 biastable
                                copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                                if (nL0Idx == nL0Loop - 1) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                                }
                            }
                        }

                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                        // Locate the current tile on L0C
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                     tla::MakeCoord(mL0Idx * L0_TILE_M, nL0Idx * L0_TILE_N),
                                                     tla::MakeShape(mL0Actual, nL0Actual));

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kL1Idx == kTileCount - 1) && (mL0Idx == mL0Loop - 1) &&
                                (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }

                        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                            if (initC) {
                                tileMmad(tensorTileL0C, tensorL0A, tensorL0B, tensorL0Bias,
                                    mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                            } else {
                                tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                                    mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                            }
                        } else {
                            tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                                mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                        }

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1BListId = l1BListIdNext;
            kL1Actual = kL1ActualNext;
        }

        // copy block out
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            copyL0CToGm(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        } else {
            copyL0CToGm(tensorC, tensorL0C, 0b11);
        }
    }

protected:
    template<class TensorA>
    CATLASS_DEVICE auto GetTileA(TensorA &tensorA, uint32_t mIndex, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        if constexpr(tla::detail::isVector<LayoutA>::value) {
            return GetTile(tensorA, tla::MakeCoord(kIndex), tla::MakeShape(kSize));
        } else {
            return GetTile(tensorA, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
        }
    }

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    AscendC::LocalTensor<uint8_t> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType* lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType* lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];
    
    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_FULL_LOADA_TLA_HPP
