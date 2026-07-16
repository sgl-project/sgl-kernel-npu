/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PRELOAD_ASYNC_WITH_CALLBACK_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PRELOAD_ASYNC_WITH_CALLBACK_TLA_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/detail/callback.hpp"
#include "catlass_a5/gemm/dispatch_policy.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "tla_a5/layout.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Block {

template <
    class ArchTag_,
    uint32_t PRELOAD_STAGES_,
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    bool ENABLE_L1_RESIDENT_,
    bool USE_HF32_MODE_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadPreloadAsyncWithCallback<
        ArchTag_,
        PRELOAD_STAGES_,
        L1A_STAGES_,
        L1B_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_,
        USE_HF32_MODE_,
        ENABLE_L1_RESIDENT_>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadPreloadAsyncWithCallback<
        ArchTag_,
        PRELOAD_STAGES_,
        L1A_STAGES_,
        L1B_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_,
        USE_HF32_MODE_,
        ENABLE_L1_RESIDENT_>;
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

    static_assert(
        tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!"
    );
    static_assert(
        tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!"
    );

    static constexpr uint32_t PRELOAD_STAGES = DispatchPolicy::PRELOAD_STAGES;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t MIN_L1_STAGES = L1A_STAGES > L1B_STAGES ? L1B_STAGES : L1A_STAGES;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr bool USE_HF32_MODE = DispatchPolicy::USE_HF32_MODE;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
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

    // Check LayoutC
    static_assert(
        tla::detail::isRowMajor<LayoutC>::value
            || ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t>
                 || std::is_same_v<ElementC, float>)
                && tla::detail::iszN<ElementC, LayoutC>::value),
        "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!"
    );

    // Check L1TileShape
    static_assert(
        L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!"
    );

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet"
    );
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");
    static_assert(PRELOAD_STAGES <= MIN_L1_STAGES, 
        "PRELOAD_STAGES must not exceed the smaller of L1A_STAGES and L1B_STAGES!");

    static_assert(
        (!HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 8) || (HAS_BIAS && (L1A_STAGES + L1B_STAGES + MIN_L1_STAGES) <= 8),
        "L1 Buffer overflow: Exceeds the supported range of EVENT(0~7)"
    );
    static_assert(
        (!HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 8) || (HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 7), 
        "L0 Buffer overflow: Exceeds the supported range of EVENT_ID(0~7)");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // use HF32 when USE_HF32_MODE is true
        if constexpr (USE_HF32_MODE) {
            AscendC::SetHF32Mode(true);
        } else {
            AscendC::SetHF32Mode(false);
        }
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(true);
        }
        InitL1(resource, l1BufAddrStart);
        InitL0A(resource);
        InitL0B(resource);
        InitL0C(resource);
        if constexpr (HAS_BIAS) {
            l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
        }
        if constexpr (ENABLE_L1_RESIDENT) {
            RestoreStatus();
        }
    }

    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        if constexpr (USE_HF32_MODE) {
            AscendC::SetHF32Mode(false);
        }
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(false);
        }
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
        if constexpr (HAS_BIAS) {
            for (uint32_t i = 0; i < MIN_L1_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BiasEventList[i]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
        }
    }

    template <class TensorA, class TensorB, class TensorC, class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(
        TensorA &tensorA,
        TensorB &tensorB,
        TensorC &tensorC,
        GemmCoord const &actualShape,
        TensorBias const &tensorBias = {},
        Callback const &callbackBeforeFixpipe = {},
        Callback const &callbackAfterFixpipe = {}
    )
    {
        // Check L1TileShape
        if constexpr (HAS_BIAS) {
            static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
            static constexpr uint32_t L1BIAS_SIZE = L1_TILE_N * MIN_L1_STAGES * sizeof(ElementBias);
            static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE, "BIAS_BUF_SIZE exceeding the BT space! Reduce L0_TILE_N");
            static_assert(
                L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES + L1BIAS_SIZE <= ArchTag::L1_SIZE,
                "L1TileShape exceeding the L1 space!"
            );
        }

        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            // Avoid using the gemv mode in mmad
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() % kL1Loop;
        }

        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; ++kL1Idx) {
            uint32_t kL1TileIdx = (startTileIdx + kL1Idx < kL1Loop) ? (startTileIdx + kL1Idx)
                                                                    : (startTileIdx + kL1Idx - kL1Loop);

            uint32_t kL1Actual = (kL1TileIdx < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1TileIdx * L1_TILE_K);

            // Load matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorTileA =
                GetTile(tensorA, tla::MakeCoord(0, kL1TileIdx * L1_TILE_K), tla::MakeShape(mBlockActual, kL1Actual));
            if constexpr (ENABLE_L1_RESIDENT) {
                // If the currently loaded GM pointer and block coordinates are the same as the last loaded ones,
                // skip this loading.
                if (lastAddrA[l1AListId] != tensorTileA.data().GetPhyAddr()
                    || tla::get<0>(tensorTileA.coord()) != lastCoordA[l1AListId].row()
                    || tla::get<1>(tensorTileA.coord()) != lastCoordA[l1AListId].column()) {
                    copyGmToL1A(tensorL1A, tensorTileA);
                    lastCoordA[l1AListId] = MatrixCoord{
                        tla::get<0>(tensorTileA.coord()), tla::get<1>(tensorTileA.coord())};
                    lastAddrA[l1AListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                        tensorTileA.data().GetPhyAddr()
                    );
                }
            } else {
                copyGmToL1A(tensorL1A, tensorTileA);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
            // Load matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorTileB =
                GetTile(tensorB, tla::MakeCoord(kL1TileIdx * L1_TILE_K, 0), tla::MakeShape(kL1Actual, nBlockActual));
            if constexpr (ENABLE_L1_RESIDENT) {
                if (lastAddrB[l1BListId] != tensorTileB.data().GetPhyAddr()
                    || tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListId].row()
                    || tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListId].column()) {
                    copyGmToL1B(tensorL1B, tensorTileB);
                    lastCoordB[l1BListId] = MatrixCoord{
                        tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                    lastAddrB[l1BListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                        tensorTileB.data().GetPhyAddr()
                    );
                }
            } else {
                copyGmToL1B(tensorL1B, tensorTileB);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

            if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                if (kL1Idx == 0) {
                    using CopyGmToL1Bias = typename TileCopy::template CopyGmToL1Bias<TensorBias>;
                    CopyGmToL1Bias copyGmToL1Bias;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BiasEventList[l1BiasListId]);
                    auto l1Bias = l1BiasTensor[l1BiasListId].template ReinterpretCast<ElementBias>();
                    auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
                    copyGmToL1Bias(tensorL1Bias, tensorBias);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BiasEventList[l1BiasListId]);
                }
            }

            // If the number of preload instructions reaches the upper limit, perform an mmad calculation on L1 tile
            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad<TensorC>(l1TileMmadParamsList[l1TileMmadParamsId]);
            }

            // Store the current load status
            uint32_t preloadL1TileMmadParamsId = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES)
                                                     ? (l1TileMmadParamsId + preloadCount)
                                                     : (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
            auto &l1TileMmadParams = l1TileMmadParamsList[preloadL1TileMmadParamsId];
            l1TileMmadParams.l1AListId = l1AListId;
            l1TileMmadParams.l1BListId = l1BListId;
            l1TileMmadParams.l1BiasListId = l1BiasListId;
            l1TileMmadParams.mL1Actual = mL1Actual;
            l1TileMmadParams.nL1Actual = nL1Actual;
            l1TileMmadParams.kL1Actual = kL1Actual;
            l1TileMmadParams.isKLoopFirst = (kL1Idx == 0);
            l1TileMmadParams.isKLoopLast = (kL1Idx == kL1Loop - 1);
            l1TileMmadParams.isNeedAddBias = !std::is_same_v<TensorBias, EmptyClass>;
            if (kL1Idx == kL1Loop - 1) {
                l1TileMmadParams.gmBlockC = tensorC.data();
                l1TileMmadParams.layoutCInGm = tensorC.layout();
                l1TileMmadParams.coord = tensorC.coord();
                l1TileMmadParams.callbackBeforeFixpipe = callbackBeforeFixpipe;
                l1TileMmadParams.callbackAfterFixpipe = callbackAfterFixpipe;
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            }
            l1AListId = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            l1BListId = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
            if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                if (kL1Idx == 0) {
                    l1BiasListId = (l1BiasListId + 1 < MIN_L1_STAGES) ? (l1BiasListId + 1) : 0;
                }
            }
        }
    }

    template <class TensorC>
    CATLASS_DEVICE void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad<TensorC>(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

    // When enabling L1 resident mode, restore the pointer and coordinates that record the last state
    // to the initial state. if two blockmmad instances need to be consecutively invoked at the kernel layer,
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

private:
    struct L1TileMmadParams {
        uint32_t l1AListId;
        uint32_t l1BListId;
        uint32_t l1BiasListId;
        uint32_t mL1Actual;
        uint32_t nL1Actual;
        uint32_t kL1Actual;
        bool isKLoopFirst;
        bool isKLoopLast;
        bool isNeedAddBias;
        AscendC::GlobalTensor<ElementC> gmBlockC;
        LayoutC layoutCInGm;
        tla::tuple<uint32_t, uint32_t> coord;
        Callback callbackBeforeFixpipe;
        Callback callbackAfterFixpipe;

        CATLASS_DEVICE
        L1TileMmadParams() = default;
    };

    CATLASS_DEVICE
    void InitL1(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1A_STAGES;
        uint32_t l1BiasOffset = l1BOffset + L1B_TILE_SIZE * L1B_STAGES;
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l1BEventList[i] = i + L1A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < MIN_L1_STAGES; ++i) {
            if constexpr (HAS_BIAS) {
                l1BiasTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(
                    l1BiasOffset + L1_TILE_N * sizeof(ElementBias) * i
                );
                l1BiasEventList[i] = i + L1A_STAGES + L1B_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BiasEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    void InitL0A(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
    }

    CATLASS_DEVICE
    void InitL0B(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    CATLASS_DEVICE
    void InitL0C(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    template <class TensorC>
    CATLASS_DEVICE void L1TileMmad(L1TileMmadParams const &params)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;

        uint32_t kL0Loop = CeilDiv<L0_TILE_K>(params.kL1Actual);
        auto &l1ATensor = l1ATensorList[params.l1AListId];
        auto &l1BTensor = l1BTensorList[params.l1BListId];
        auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});

        auto &l0CTensor = l0CTensorList[l0CListId];
        auto layoutInL0C = tla::MakeLayoutL0C(params.mL1Actual, params.nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (params.isKLoopFirst) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; ++kL0Idx) {
            uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (params.kL1Actual - kL0Idx * L0_TILE_K);

            auto &l0ATile = l0ATensorList[l0AListId];
            auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(params.mL1Actual, kL0Actual);
            auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
            auto tensorTileL1A =
                GetTile(tensorL1A, tla::MakeCoord(0, kL0Idx * L0_TILE_K), tla::MakeShape(params.mL1Actual, kL0Actual));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            if (kL0Idx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[params.l1AListId]);
            }
            copyL1ToL0A(tensorL0A, tensorTileL1A);
            if (kL0Idx == kL0Loop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[params.l1AListId]);
            }

            auto &l0BTile = l0BTensorList[l0BListId];
            auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, params.nL1Actual);
            auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
            // Locate the current tile of matrix B on L1
            auto tensorTileL1B =
                GetTile(tensorL1B, tla::MakeCoord(kL0Idx * L0_TILE_K, 0), tla::MakeShape(kL0Actual, params.nL1Actual));

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            if (kL0Idx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1BListId]);
            }
            copyL1ToL0B(tensorL0B, tensorTileL1B);
            if (kL0Idx == kL0Loop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1BListId]);
            }

            bool initC = (params.isKLoopFirst && (kL0Idx == 0));
            if constexpr (HAS_BIAS) {
                if (params.isNeedAddBias && initC) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BiasEventList[params.l1BiasListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                    auto l1Bias = l1BiasTensor[params.l1BiasListId].template ReinterpretCast<ElementBias>();
                    auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
                    auto tensorTileL1Bias = GetTile(tensorL1Bias, tla::MakeCoord(0), tla::MakeShape(params.nL1Actual));
                    // Load bias to l0 biasTable
                    copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BiasEventList[params.l1BiasListId]);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

            // If the unit flag is enabled, the unit flag is set according to the calculation progress
            uint8_t unitFlag = 0b00;
            if constexpr (ENABLE_UNIT_FLAG) {
                if (params.isKLoopLast && (kL0Idx == kL0Loop - 1)) {
                    unitFlag = 0b11;
                } else {
                    unitFlag = 0b10;
                }
            }

            if constexpr (HAS_BIAS) {
                if (params.isNeedAddBias && initC) {
                    tileMmad(
                        tensorL0C, tensorL0A, tensorL0B, tensorL0Bias, params.mL1Actual, params.nL1Actual, kL0Actual,
                        initC, unitFlag
                    );
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                } else {
                    tileMmad(
                        tensorL0C, tensorL0A, tensorL0B, params.mL1Actual, params.nL1Actual, kL0Actual, initC, unitFlag
                    );
                }
            } else {
                tileMmad(
                    tensorL0C, tensorL0A, tensorL0B, params.mL1Actual, params.nL1Actual, kL0Actual, initC, unitFlag
                );
            }

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
            l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
        }

        if (params.isKLoopLast) {
            auto layoutCInGm = params.layoutCInGm;
            auto tensorTileC =
                tla::MakeTensor(params.gmBlockC, layoutCInGm, params.coord, Arch::PositionType<TensorC::position>{});

            params.callbackBeforeFixpipe();

            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                copyL0CToDst(tensorTileC, tensorL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                copyL0CToDst(tensorTileC, tensorL0C, 0b11);
            }
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;

            params.callbackAfterFixpipe();
        }
    }

    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l1BiasListId{0};

    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    AscendC::LocalTensor<uint8_t> l1BiasTensor[MIN_L1_STAGES];
    int32_t l1BiasEventList[MIN_L1_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};
    uint32_t preloadCount{0};

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType* lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType* lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PRELOAD_ASYNC_WITH_CALLBACK_TLA_HPP
