/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DYNAMIC_LOCAL_PADDING_C_PADDING_COMMON_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_DYNAMIC_LOCAL_PADDING_C_PADDING_COMMON_MATMUL_HPP

#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm/kernel/padding_matmul.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class PrologueA_,
    class PrologueB_,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class RemovePaddingC_>
class DynamicLocalPaddingCPaddingCommonMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using RemovePaddingC = RemovePaddingC_;

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutC = typename BlockMmad::LayoutC;

    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::
        conditional_t<std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::
        conditional_t<std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord l1TileShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrWC;
        uint32_t swizzleOffset;
        uint32_t swizzleDirection;
        uint32_t m1Factor;
        uint32_t n1Factor;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GemmCoord const &l1TileShape_,
            GM_ADDR ptrA_,
            LayoutA &layoutA_,
            GM_ADDR ptrB_,
            LayoutB &layoutB_,
            GM_ADDR ptrC_,
            LayoutC &layoutC_,
            GM_ADDR ptrWA_,
            GM_ADDR ptrWB_,
            GM_ADDR ptrWC_,
            uint32_t swizzleOffset_,
            uint32_t swizzleDirection_,
            uint32_t m1Factor_,
            uint32_t n1Factor_
        )
            : problemShape(problemShape_)
            , l1TileShape(l1TileShape_)
            , ptrA(ptrA_)
            , layoutA(layoutA_)
            , ptrB(ptrB_)
            , layoutB(layoutB_)
            , ptrC(ptrC_)
            , layoutC(layoutC_)
            , ptrWA(ptrWA_)
            , ptrWB(ptrWB_)
            , ptrWC(ptrWC_)
            , swizzleOffset(swizzleOffset_)
            , swizzleDirection(swizzleDirection_)
            , m1Factor(m1Factor_)
            , n1Factor(n1Factor_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    DynamicLocalPaddingCPaddingCommonMatmul()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            typename BlockMmad::LayoutA layoutWA;
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            typename BlockMmad::LayoutB layoutWB;
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        if constexpr (!std::is_void_v<RemovePaddingC>) {
            for (uint32_t i = 0; i < GMWC_BUFFER_NUM; ++i) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(cvSyncFlagList[i]);
            }

            MatrixCoord gmTileShape(params.l1TileShape.m() * params.m1Factor, params.l1TileShape.n() * params.n1Factor);

            BlockScheduler matmulBlockScheduler(
                params.problemShape, gmTileShape, params.swizzleOffset, params.swizzleDirection
            );
            AscendC::GlobalTensor<ElementC> gmC;
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
            AscendC::GlobalTensor<ElementC> gmWC;
            gmWC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWC);
            uint32_t aiCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            uint32_t aiCoreNum = AscendC::GetBlockNum();
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            RemovePaddingC removePaddingC(resource);
            for (uint32_t loopIdx = aiCoreIdx; loopIdx < coreLoops; loopIdx += aiCoreNum) {
                Catlass::Arch::CrossCoreWaitFlag(cvSyncFlagList[cvSyncId]);
                // Compute initial location in logical coordinates
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                uint64_t gmOffsetWC = static_cast<size_t>(aiCoreIdx * GMWC_BUFFER_NUM + cvSyncId) * gmTileShape.row()
                                      * gmTileShape.column();
                LayoutC layoutWC = LayoutC{actualBlockShape.m(), actualBlockShape.n(), gmTileShape.column()};
                MatrixCoord coordC{blockCoord.m() * gmTileShape.row(), blockCoord.n() * gmTileShape.column()};
                int64_t gmOffsetC = params.layoutC.GetOffset(coordC);
                removePaddingC(gmC[gmOffsetC], gmWC[gmOffsetWC], params.layoutC, layoutWC, true);

                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(cvSyncFlagList[cvSyncId]);
                cvSyncId = (cvSyncId + 1 < GMWC_BUFFER_NUM) ? (cvSyncId + 1) : 0;
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        typename BlockMmad::LayoutA layoutA;
        typename BlockMmad::LayoutB layoutB;
        typename BlockMmad::LayoutC layoutC;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        if constexpr (std::is_void_v<PrologueA>) {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            layoutA = params.layoutA;
        } else {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
        }
        AscendC::GlobalTensor<ElementB> gmB;
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
        }
        AscendC::GlobalTensor<ElementC> gmC;
        if constexpr (std::is_void_v<RemovePaddingC>) {
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
            layoutC = params.layoutC;
        } else {
            gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWC);
        }

        MatrixCoord gmTileShape(params.l1TileShape.m() * params.m1Factor, params.l1TileShape.n() * params.n1Factor);

        BlockScheduler matmulBlockScheduler(
            params.problemShape, gmTileShape, params.swizzleOffset, params.swizzleDirection
        );
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(params.l1TileShape, resource);

        GemmCoord gmBlockCoord;
        GemmCoord gmActualBlockShape;
        GemmCoord gmNextBlockCoord;
        GemmCoord gmNextActualBlockShape;

        uint32_t aiCoreIdx = AscendC::GetBlockIdx();
        uint32_t aiCoreNum = AscendC::GetBlockNum();

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {

            if constexpr (!std::is_void_v<RemovePaddingC>) {
                Catlass::Arch::CrossCoreWaitFlag(cvSyncFlagList[cvSyncId]);
            }

            bool isGmFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            if (isGmFirstBlock) {
                gmBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                gmActualBlockShape = matmulBlockScheduler.GetActualBlockShape(gmBlockCoord);
                gmBlockCoord = gmBlockCoord * GemmCoord(params.m1Factor, params.n1Factor, 0);
            } else {
                gmBlockCoord = gmNextBlockCoord;
                gmActualBlockShape = gmNextActualBlockShape;
            }

            bool hasGmNextBlock = false;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasGmNextBlock = true;
                gmNextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                gmNextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(gmNextBlockCoord);
                gmNextBlockCoord = gmNextBlockCoord * GemmCoord(params.m1Factor, params.n1Factor, 0);
            }

            MatrixCoord innerLoopsMN = CeilDiv(
                MakeCoord(gmActualBlockShape.m(), gmActualBlockShape.n()),
                MakeCoord(params.l1TileShape.m(), params.l1TileShape.n())
            );
            uint32_t innerLoops = innerLoopsMN.row() * innerLoopsMN.column();

            GemmCoord blockCoord;
            GemmCoord actualBlockShape;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            bool hasNextBlock = false;
            bool isFirstBlock = false;

            if constexpr (!std::is_void_v<RemovePaddingC>) {
                layoutC = LayoutC{gmActualBlockShape.m(), gmActualBlockShape.n(), gmTileShape.column()};
            }

            // Get actual block shape for mmad
            auto GetActualBlockShape = [](const GemmCoord &problemShape, const GemmCoord &l1TileShape,
                                          const GemmCoord &blockCoord) {
                GemmCoord loopsMNK = CeilDiv(problemShape, l1TileShape);
                uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1))
                                       ? (problemShape.m() - blockCoord.m() * l1TileShape.m())
                                       : l1TileShape.m();
                uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1))
                                       ? (problemShape.n() - blockCoord.n() * l1TileShape.n())
                                       : l1TileShape.n();
                return GemmCoord{mActual, nActual, problemShape.k()};
            };

            for (uint32_t innerLoopIdx = 0; innerLoopIdx < innerLoops; ++innerLoopIdx) {
                uint32_t mIdx = innerLoopIdx / innerLoopsMN.column();
                uint32_t nIdx = innerLoopIdx % innerLoopsMN.column();

                if (innerLoopIdx == 0 && isGmFirstBlock) {
                    isFirstBlock = true;
                } else {
                    isFirstBlock = false;
                }

                if (innerLoopIdx + 1 < innerLoops || hasGmNextBlock) {
                    hasNextBlock = true;
                } else {
                    hasNextBlock = false;
                }

                if (innerLoopIdx + 1 < innerLoops) {
                    nextBlockCoord = gmBlockCoord
                                     + GemmCoord(
                                         (innerLoopIdx + 1) / innerLoopsMN.column(),
                                         (innerLoopIdx + 1) % innerLoopsMN.column(), 0
                                     );

                } else if (hasGmNextBlock) {
                    nextBlockCoord = gmNextBlockCoord;
                }
                blockCoord = gmBlockCoord + GemmCoord(mIdx, nIdx, 0);
                MatrixCoord coordA{blockCoord.m() * params.l1TileShape.m(), blockCoord.k() * params.l1TileShape.k()};
                MatrixCoord coordB{blockCoord.k() * params.l1TileShape.k(), blockCoord.n() * params.l1TileShape.n()};
                MatrixCoord coordC{blockCoord.m() * params.l1TileShape.m(), blockCoord.n() * params.l1TileShape.n()};
                int64_t gmOffsetA = layoutA.GetOffset(coordA);
                int64_t gmOffsetB = layoutB.GetOffset(coordB);
                actualBlockShape = GetActualBlockShape(params.problemShape, params.l1TileShape, blockCoord);

                MatrixCoord coordNextA{
                    nextBlockCoord.m() * params.l1TileShape.m(), nextBlockCoord.k() * params.l1TileShape.k()
                };
                MatrixCoord coordNextB{
                    nextBlockCoord.k() * params.l1TileShape.k(), nextBlockCoord.n() * params.l1TileShape.n()
                };
                int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
                int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);
                nextActualBlockShape = GetActualBlockShape(params.problemShape, params.l1TileShape, nextBlockCoord);

                int64_t gmOffsetC{0};
                if constexpr (!std::is_void_v<RemovePaddingC>) {
                    gmOffsetC = static_cast<size_t>(aiCoreIdx * GMWC_BUFFER_NUM + cvSyncId) * gmTileShape.row()
                                    * gmTileShape.column()
                                + layoutC.GetOffset(
                                    MakeCoord(mIdx * params.l1TileShape.m(), nIdx * params.l1TileShape.n())
                                );
                } else {
                    gmOffsetC = layoutC.GetOffset(coordC);
                }

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], layoutC, gmA[gmOffsetNextA],
                    gmB[gmOffsetNextB], actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock
                );
            }

            if constexpr (!std::is_void_v<RemovePaddingC>) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cvSyncFlagList[cvSyncId]);
                cvSyncId = (cvSyncId + 1 < GMWC_BUFFER_NUM) ? (cvSyncId + 1) : 0;
            }
        }
        if constexpr (!std::is_void_v<RemovePaddingC>) {
            for (uint32_t i = 0; i < GMWC_BUFFER_NUM; ++i) {
                Catlass::Arch::CrossCoreWaitFlag(cvSyncFlagList[i]);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr uint32_t GMWC_BUFFER_NUM = 4;
    Arch::CrossCoreFlag cvSyncFlagList[GMWC_BUFFER_NUM] =
        {Arch::FlagID{0}, Arch::FlagID{1}, Arch::FlagID{2}, Arch::FlagID{3}};
    uint32_t cvSyncId{0};
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_DYNAMIC_LOCAL_PADDING_C_PADDING_COMMON_MATMUL_HPP