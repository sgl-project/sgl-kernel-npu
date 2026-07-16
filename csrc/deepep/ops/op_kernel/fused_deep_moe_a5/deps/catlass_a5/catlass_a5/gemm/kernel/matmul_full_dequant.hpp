/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef CATLASS_GEMM_KERNEL_MATMUL_FULL_DEQUANT_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_FULL_DEQUANT_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/arch/cross_core_sync.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/status.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class KernelMatmulDequant{
public:
    CATLASS_DEVICE 
    KernelMatmulDequant() = default;
    CATLASS_DEVICE 
    ~KernelMatmulDequant() = default;

    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;
    using OutputType = typename BlockEpilogue::DataTypeOut;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    static constexpr int64_t l1M = tla::get<0>(L1TileShape{});
    static constexpr int64_t l1N = tla::get<1>(L1TileShape{});
    static constexpr int64_t l1K = tla::get<2>(L1TileShape{});

    using TupleShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = tla::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    
    // ND layout
    using NDLayout = tla::Layout<tla::Shape<int64_t, int64_t>, tla::Stride<int64_t, tla::Int<1>>>; 

    struct DequantMatmulArguments {
        GM_ADDR x1;
        GM_ADDR x2;
        GM_ADDR y;
        DequantMatmulArguments() = default;
    };

    struct Arguments {
        ProblemShape problemShape;
        DequantMatmulArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        Arguments() = default;
    };

    using DequantMatmulParams = DequantMatmulArguments;
    struct Params {
        ProblemShape problemShape;
        DequantMatmulParams mmadParams;
        BlockEpilogueParams epilogueParams;
        Params() = default;
    };

private:
    using AGlobalTensorType = AscendC::GlobalTensor<ElementA>;
    using BGlobalTensorType = AscendC::GlobalTensor<ElementB>;
    using CLocalTensorType = AscendC::LocalTensor<ElementC>;
    using OutputTensorType = AscendC::GlobalTensor<OutputType>;

    using AGmTensor = tla::Tensor<AGlobalTensorType, NDLayout, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::GM>;
    using BGmTensor = tla::Tensor<BGlobalTensorType, NDLayout, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::GM>;
    using CUbTensor = tla::Tensor<CLocalTensorType, NDLayout, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::VECCALC>;
    using OutputTensor = tla::Tensor<OutputTensorType, NDLayout, tla::Coord<tla::_0, tla::_0>,
                                     AscendC::TPosition::GM>;
    AGlobalTensorType aGm_;
    BGlobalTensorType bGm_;
    CLocalTensorType cUb_;
    OutputTensorType outputGm_;
    AGmTensor aGmTensor_;
    BGmTensor bGmTensor_;
    CUbTensor cUbTensor_;
    OutputTensor outputTensor_;

    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static int16_t AIV_SYNC_AIC_FLAG = 5;
    constexpr static int16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static int16_t FLAG_ID_MAX = 16;
    constexpr static int16_t COUNT_ID_MAX = 15;
    constexpr static int16_t COUNT_FLAG = 3;

public:
    CATLASS_DEVICE 
    void Init(Params const& params) 
    {
        int64_t m = params.problemShape.m;
        int64_t n = params.problemShape.n;
        int64_t k = params.problemShape.k;

        AscendC::GlobalTensor<ElementA> tempA;
        tempA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.mmadParams.x1), m * k);
        aGm_.address_ = tempA.address_;
        AscendC::GlobalTensor<ElementB> tempB;
        tempB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.mmadParams.x2), k * n);
        bGm_.address_ = tempB.address_;
        AscendC::GlobalTensor<OutputType> tempOut;
        tempOut.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(params.mmadParams.y), m * n);
        outputGm_.address_ = tempOut.address_;
        
        NDLayout aLayout = tla::MakeLayout(tla::MakeShape(m, k),tla::MakeStride(k, tla::Int<1>{}));
        aGmTensor_ = tla::MakeTensor(aGm_, aLayout, Arch::PositionGM{});

        NDLayout bLayout = tla::MakeLayout(tla::MakeShape(k, n),tla::MakeStride(n, tla::Int<1>{}));
        bGmTensor_ = tla::MakeTensor(bGm_, bLayout, Arch::PositionGM{});

        NDLayout outLayout = tla::MakeLayout(tla::MakeShape(m, n),tla::MakeStride(n, tla::Int<1>{}));
        outputTensor_ = tla::MakeTensor(outputGm_, outLayout, Arch::PositionGM{});
    }

    CATLASS_HOST_DEVICE 
    static Status CheckShape(ProblemShape const &shape)
    {
        return Status::kSuccess;
    }

    CATLASS_HOST_DEVICE 
    static bool CanImplement(Arguments const& args)
    {
        return true;
    }

    CATLASS_HOST_DEVICE 
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE 
    static Params ToUnderlyingArguments(Arguments const& args, GM_ADDR workspace)
    {
        Params params = {
            args.problemShape,
            args.mmadArgs,
            BlockEpilogue::InitParams(args.epilogueArgs),
        };
        return params;
    }
    
    CATLASS_DEVICE void operator()(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        BlockEpilogue epilogue;
        
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        constexpr bool enable2UB = (AscendC::IsSameType<ElementC, ElementAccumulator>::value);
        if ASCEND_IS_AIV {
            if constexpr (!enable2UB && AscendC::GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        if (curBlockIdx >= blockNum) {
            return;
        }
        
        GemmCoord ps {
            params.problemShape.m,
            params.problemShape.n,
            params.problemShape.k
        };

        Init(params);
        int64_t calM = enable2UB ? CeilDiv(l1M, AscendC::GetTaskRation()) : l1M;
        epilogue.Init(resource, params.epilogueParams, ps);
        BlockScheduler bs(curBlockIdx, blockNum, ps);

        if (bs.endBlockIdx_ + 1 <= blockNum / 2) {
            bs.UpdateTailTile();
        }
        bool enableCVSync = false;
        int count = 0;
        int64_t count_id = 0;
        uint32_t coreLoops = bs.round_;
        for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
            bool isLastLoop = (loopIdx == coreLoops - 1 && curBlockIdx <= bs.endBlockIdx_);
            bs.UpdateMNTileIdx(loopIdx, isLastLoop);
            bs.UpdateBlockShape(loopIdx, isLastLoop);

            auto blockShape = bs.GetBlockShape();
            auto blkElemCoord = bs.GetBlockCoordByElement();

            uint32_t mCoord = blkElemCoord.m();
            uint32_t nCoord = blkElemCoord.n();
            uint32_t kCoord = blkElemCoord.k();

            uint32_t blockM = blockShape.m();
            uint32_t blockN = blockShape.n();
            uint32_t blockK = blockShape.k();

            GemmCoord actualBlockShape = GemmCoord{blockM, blockN, blockK};

            if (blockM <= 0 || blockN <= 0) {
                break;
            }

            auto ubLocalTemp = epilogue.GetL0c2UbTensor();

            auto tensorA = tla::MakeTensor(aGm_, aGmTensor_.layout(), Arch::PositionGM{});
            auto tensorB = tla::MakeTensor(bGm_, bGmTensor_.layout(), Arch::PositionGM{});

            cUb_.SetAddr(ubLocalTemp.address_);
            int64_t alignN = RoundUp(blockN, static_cast<int64_t>(Catlass::BYTE_PER_BLK / sizeof(ElementC)));
            auto cUbLayout = tla::MakeLayout<ElementC, layout::RowMajor>(blockM, alignN);
            auto tensorC = tla::MakeTensor(cUb_, cUbLayout, Arch::PositionUB{});

            if ASCEND_IS_AIC {
                if (enableCVSync) {
                    count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + count_id);
                    if constexpr (enable2UB) {
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                            AIV_SYNC_AIC_FLAG + count_id + FLAG_ID_MAX);
                    }
                }
                
                auto aInputTlaTensor = GetTile(aGmTensor_,
                                               tla::MakeCoord(mCoord, kCoord),
                                               tla::MakeShape(blockM, blockK));
                auto bInputTlaTensor = GetTile(bGmTensor_,
                                               tla::MakeCoord(kCoord, nCoord),
                                               tla::MakeShape(blockK, blockN));
                auto cOutputTlaTensor = GetTile(tensorC,
                                               tla::MakeCoord(0, 0),
                                               tla::MakeShape(blockM, alignN));
                
                blockMmad(
                    aInputTlaTensor,
                    bInputTlaTensor,
                    cOutputTlaTensor,
                    actualBlockShape
                );
                // Compute block-level mmad with epilogue
                enableCVSync = true;
                count++;
                count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIC_SYNC_AIV_FLAG + count_id);
                if constexpr (enable2UB) {
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIC_SYNC_AIV_FLAG + count_id + FLAG_ID_MAX);
                }
            }
            // AIV Process
            if ASCEND_IS_AIV {
                // Synchronize with aic
                count++;
                count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(
                    AIC_SYNC_AIV_FLAG + count_id);
                // Calulate epilogue
                auto cUbTlaTensor = GetTile(tensorC,
                                            tla::MakeCoord(mCoord, nCoord),
                                            tla::MakeShape(blockM, blockN));
                epilogue(cUbTlaTensor);
                // Notify aic
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(
                    AIV_SYNC_AIC_FLAG + count_id);
            }
        }
        // Match extra event after aic process finished
        if ASCEND_IS_AIC {
            if (enableCVSync) {
                count_id = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIV_SYNC_AIC_FLAG + count_id);
                if constexpr (enable2UB) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + count_id + FLAG_ID_MAX);
                }
            }
        }
    }
};

} // namespace Catlass::Gemm::Kernel
#endif //CATLASS_GEMM_KERNEL_MATMUL_FULL_DEQUANT_HPP