/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_SPARSE_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_SPARSE_MATMUL_TLA_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/coord.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "catlass_a5/status.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"
#include "catlass_a5/gemm/block/block_scheduler_iterateK.hpp"

namespace Catlass::Gemm::Kernel {


template <
    class ProblemShape_,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class KernelSparseMatmul {
public:
    CATLASS_DEVICE
    KernelSparseMatmul() {}
    CATLASS_DEVICE
    ~KernelSparseMatmul() {}

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using L1Shape = typename BlockMmad::L1Shape;
    using L0Shape = typename BlockMmad::L0Shape;

    static constexpr uint32_t L1_M = tla::get<0>(L1Shape{});
    static constexpr uint32_t L1_N = tla::get<1>(L1Shape{});
    static constexpr uint32_t L1_K = tla::get<2>(L1Shape{});
    static constexpr uint32_t L0_M = tla::get<0>(L0Shape{});
    static constexpr uint32_t L0_N = tla::get<1>(L0Shape{});
    static constexpr uint32_t L0_K = tla::get<2>(L0Shape{});
    static constexpr uint32_t L0A_SIZE = BlockMmad::DispatchPolicy::ArchTag::AtlasA2::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = BlockMmad::DispatchPolicy::ArchTag::AtlasA2::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = BlockMmad::DispatchPolicy::ArchTag::AtlasA2::L0C_SIZE;
    static constexpr uint32_t L1_SIZE = BlockMmad::DispatchPolicy::ArchTag::AtlasA2::L1_SIZE;
    static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t DENSE_MATRIX_B_OFFSET = 2;
    static constexpr uint32_t INDEX_MATRIX_OFFSET = 8;
    static constexpr uint32_t MATRIX_INNER_DIM_LIMIT_SIZE = 65536;

    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

    /**
     * @struct BlockMmadArguments
     * @brief Kernel arguments for the host side
     */
    struct BlockMmadArguments {
        GM_ADDR aGmAddr{nullptr};       ///< The global memory address of matrix A
        GM_ADDR bGmAddr{nullptr};       ///< The global memory address of matrix B
        GM_ADDR cGmAddr{nullptr};       ///< The global memory address of matrix C
        GM_ADDR biasGmAddr{nullptr};    ///< The global memory address of bias
        GM_ADDR indexGmAddr{nullptr};   ///< The global memory address of index
        LayoutA layoutA;
        LayoutB layoutB;
        LayoutC layoutC;
    };
    
    /**
     * @struct BlockSchedulerArguments
     * @brief Kernel arguments for the host side
     */
    struct BlockSchedulerArguments {
        int64_t aicoreNum;
    };

    // schedulerOp
    using BlockSchedulerOp = BlockScheduler;

    // mmadOp
    using BlockMmadOp = BlockMmad;
    using BlockMmadParams = BlockMmadArguments;
    using BlockSchedulerArguments = BlockSchedulerArguments;
    using BlockSchedulerParams = BlockSchedulerArguments;
    using AType = typename BlockMmad::ElementA;
    using BType = typename BlockMmad::ElementB;
    using CType = typename BlockMmad::ElementC;
    using IndexType = uint8_t;

    using TupleShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = tla::Coord<int64_t, int64_t, int64_t, int64_t>;

    using AGlobalTensorType = AscendC::GlobalTensor<AType>;
    using BGlobalTensorType = AscendC::GlobalTensor<BType>;
    using CGlobalTensorType = AscendC::GlobalTensor<CType>;
    using ATlaTensor = tla::Tensor<AGlobalTensorType, LayoutA, tla::Coord<tla::_0, tla::_0>,
                                AscendC::TPosition::GM>;
    using BTlaTensor = tla::Tensor<BGlobalTensorType, LayoutB, tla::Coord<tla::_0, tla::_0>,
                                AscendC::TPosition::GM>;
    using CTlaTensor = tla::Tensor<CGlobalTensorType, LayoutC, tla::Coord<tla::_0, tla::_0>,
                                AscendC::TPosition::GM>;

    // NZ layout for index (B transpose)
    using BIndexLayout = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<8>, int64_t>, tla::Shape<tla::Int<16>, int64_t> >, // 8 = 32 / (sizeof(uint8) * 4)
        tla::Stride<tla::Stride<tla::Int<1>, int64_t>, tla::Stride<tla::Int<8>, tla::_128>>   // 128 = 16 * 8
    >;

    using IndexGlobalTensorType = AscendC::GlobalTensor<IndexType>;
    using IndexTensor = tla::Tensor<IndexGlobalTensorType, BIndexLayout, tla::Coord<tla::_0, tla::_0>,
                                    AscendC::TPosition::GM>;

    // attribute
    AGlobalTensorType aGlobal_;
    BGlobalTensorType bGlobal_;
    CGlobalTensorType cGlobal_;
    IndexGlobalTensorType indexGlobal_;
    ATlaTensor aTlaTensor_;
    BTlaTensor bTlaTensor_;
    CTlaTensor cTlaTensor_;
    IndexTensor indexTlaTensor_;

    // mmad
    BlockMmadParams blockMmadParams_{};
    // shape
    TupleShape problemShape_{};

    /**
     * @struct Arguments
     * @brief Structure to hold arguments for the problem
     */
    struct Arguments {
        ProblemShape problemShape;              ///< Problem shape
        BlockMmadArguments mmadArgs;            ///< MMAD parameters
        BlockSchedulerArguments schedulerArgs;  ///< Scheduler parameters
        Arguments() = default;                  ///< Default constructor
    };

    /**
     * @struct Params
     * @brief Structure to hold parameters for the problem
     */
    struct Params {
        ProblemShape problemShape;              ///< Problem shape
        BlockMmadParams mmadParams;             ///< MMAD parameters
        BlockSchedulerParams schedulerParams;   ///< Scheduler parameters
        Params() = default;                     ///< Default constructor
    };

    /**
     * @brief Convert ProblemShape to TupleShape
     * @param [in] shape: ProblemShape to be converted
     * @return TupleShape representation of the input ProblemShape
     */
    CATLASS_DEVICE
    static TupleShape ToShapeTuple(ProblemShape const& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    /**
     * @brief Initialize the parameters for the problem
     * @param [in] params: parameters to be initialized
     */
    CATLASS_DEVICE
    void Init(Params const& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        blockMmadParams_ = params.mmadParams;
        int64_t m = tla::get<0>(problemShape_);
        int64_t n = tla::get<1>(problemShape_);
        int64_t k = tla::get<2>(problemShape_);
        // Init Tensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr), m * k);
        aTlaTensor_ = tla::MakeTensor(aGlobal_, blockMmadParams_.layoutA, Arch::PositionGM{});

        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr), (k / DENSE_MATRIX_B_OFFSET) * n);
        bTlaTensor_ = tla::MakeTensor(bGlobal_, blockMmadParams_.layoutB, Arch::PositionGM{});

        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr), m * n);
        cTlaTensor_ = tla::MakeTensor(cGlobal_, blockMmadParams_.layoutC, Arch::PositionGM{});

        
        BIndexLayout indexLayout = tla::MakeLayout(
            tla::MakeShape(tla::MakeShape(tla::Int<INDEX_MATRIX_OFFSET>{},
                                          CeilDiv<int64_t>(k / INDEX_MATRIX_OFFSET, INDEX_MATRIX_OFFSET)),
                           tla::MakeShape(tla::Int<C0_NUM_PER_FRACTAL>{},
                                          CeilDiv<int64_t>(n, C0_NUM_PER_FRACTAL))),
            tla::MakeStride(tla::MakeStride(tla::Int<1>{}, RoundUp<int64_t>(n, C0_NUM_PER_FRACTAL) * INDEX_MATRIX_OFFSET),
                            tla::MakeStride(tla::Int<INDEX_MATRIX_OFFSET>{}, tla::Int<INDEX_MATRIX_OFFSET * C0_NUM_PER_FRACTAL>{})));
        indexGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ IndexType*>(blockMmadParams_.indexGmAddr),
                                    n * k / INDEX_MATRIX_OFFSET);
        indexTlaTensor_ = tla::MakeTensor(indexGlobal_, indexLayout, Arch::PositionGM{});
    }

    /**
     * @brief Get the offset for the block
     * @param [in] BlockCoord: type of the block coordinate
     * @param [in] blockCoord: block coordinate
     * @return Tuple of offsets for A, B, C, and index
     */
    template <class BlockCoord>
    CATLASS_DEVICE
    tla::Coord<int64_t, int64_t, int64_t, int64_t> GetOffset(const BlockCoord& blockCoord)
    {
        int64_t m = tla::get<0>(problemShape_);
        int64_t n = tla::get<1>(problemShape_);
        int64_t k = tla::get<2>(problemShape_);
        tla::Coord<int64_t, int64_t> aCoord = tla::MakeCoord(tla::get<0>(blockCoord), tla::get<2>(blockCoord));
        tla::Coord<int64_t, int64_t> bCoord = tla::MakeCoord(tla::get<2>(blockCoord), tla::get<1>(blockCoord));
        tla::Coord<int64_t, int64_t> cCoord = tla::MakeCoord(tla::get<0>(blockCoord), tla::get<1>(blockCoord));

        int64_t offsetA = aTlaTensor_.layout()(aCoord) + tla::get<3>(blockCoord) * m * k;
        int64_t offsetB =
            bTlaTensor_.layout()(bCoord) + tla::get<3>(blockCoord) * n * k / DENSE_MATRIX_B_OFFSET;
        int64_t offsetC = cTlaTensor_.layout()(cCoord) + tla::get<3>(blockCoord) * m * n;
        int64_t offsetIndex =
            indexTlaTensor_.layout()(bCoord) + tla::get<3>(blockCoord) * n * k / INDEX_MATRIX_OFFSET;

        return {offsetA, offsetB, offsetC, offsetIndex};
    }

    /**
     * @brief Check the shape of the problem
     * @param [in] shape: problem shape to be checked
     * @return Status of the check
     */
    CATLASS_HOST_DEVICE
    static Status CheckShape(ProblemShape const& shape)
    {
        uint32_t m = shape.m;
        uint32_t n = shape.n;
        uint32_t k = shape.k;
        uint32_t b = shape.b;
        if (b > 1) { // Sparse only support batch 1
            return Status::kInvalid;
        }
        if (k % 8 != 0) { // 8: Sparse k must be multiple of 8
            return Status::kInvalid;
        }
        // Check matrix size exceeds limit
        if (!tla::detail::isColumnMajor<LayoutA>::value && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // mk matrix k limit
            return Status::kInvalid;
        }

        if (tla::detail::isColumnMajor<LayoutA>::value && m > MATRIX_INNER_DIM_LIMIT_SIZE) { // km matrix m limit
            return Status::kInvalid;
        }
        if (!tla::detail::isColumnMajor<LayoutB>::value && n > MATRIX_INNER_DIM_LIMIT_SIZE) { // kn matrix n limit
            return Status::kInvalid;
        }

        if (tla::detail::isColumnMajor<LayoutB>::value && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // nk matrix k limit
            return Status::kInvalid;
        }
        return Status::kSuccess;
    }

    /**
     * @brief Check if the problem can be implemented
     * @param [in] args: arguments for the problem
     * @return Status of the check
     */
    CATLASS_HOST_DEVICE
    static bool CanImplement(Arguments const& args)
    {
        // Check mmad args
        Status BlockMmadCanImplement;
        if (L0_M * L0_K * sizeof(AType) * STAGES > L0A_SIZE ||
            L0_N * L0_K * sizeof(BType) * STAGES > L0B_SIZE || L0_M * L0_N * sizeof(CType) > L0C_SIZE ||
            (L1_M * L1_K * sizeof(AType) + L1_K * L1_N * sizeof(BType)) * STAGES > L1_SIZE) {
            BlockMmadCanImplement = Status::kInvalid;
        } else {
            BlockMmadCanImplement = Status::kSuccess;
        }
        if (CheckShape(args.problemShape) == Status::kInvalid || BlockMmadCanImplement == Status::kInvalid) {
            return false;
        }
        return true;
    }

    /**
     * @brief Get the workspace size for the problem
     * @param [in] args: arguments for the problem
     * @return Workspace size
     */
    CATLASS_HOST_DEVICE
    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    /**
     * @brief Initialize the parameters for the problem
     * @param [in] args: arguments for the problem
     * @param [out] workspace: the address of the work space
     * @return Initialized parameters
     */
    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(Arguments const& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = {args.mmadArgs.aGmAddr, args.mmadArgs.bGmAddr,
                                    args.mmadArgs.cGmAddr, args.mmadArgs.biasGmAddr, args.mmadArgs.indexGmAddr,
                                    args.mmadArgs.layoutA, args.mmadArgs.layoutB, args.mmadArgs.layoutC};
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, args.schedulerArgs};
        return params;
    }

    /**
     * @brief Overloaded operator() for the problem
     * @param [in] params: parameters for the problem
     */
    CATLASS_DEVICE
    void operator()(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // Get blockIdx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if (curBlockIdx >= blockNum) {
            return;
        }
        // Init
        Init(params);
        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schedulerParams.aicoreNum);

        int64_t tileNum = bs.GetTileNum();
        // Send event when using aiv_1

        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset = GetOffset(blockCoord);

            // calculate block-level offset
            if (tla::get<0>(blockShape) <= 0 || tla::get<1>(blockShape) <= 0) {
                return;
            }
            int64_t offsetA = tla::get<0>(blockOffset);
            int64_t offsetB = tla::get<1>(blockOffset);
            int64_t offsetC = tla::get<2>(blockOffset);
            int64_t offsetIndex = tla::get<3>(blockOffset);

            auto aGlobalT = aGlobal_[offsetA];
            auto bGlobalT = bGlobal_[offsetB];
            auto cGlobalT = cGlobal_[offsetC];
            auto indexGlobalT = indexGlobal_[offsetIndex];

            auto aGlobalTensor = tla::MakeTensor(aGlobalT, aTlaTensor_.layout(), Arch::PositionGM{});
            auto bGlobalTensor = tla::MakeTensor(bGlobalT, bTlaTensor_.layout(), Arch::PositionGM{});
            auto cGlobalTensor = tla::MakeTensor(cGlobalT, cTlaTensor_.layout(), Arch::PositionGM{});
            auto indexGlobalTensor = tla::MakeTensor(indexGlobalT, indexTlaTensor_.layout(), Arch::PositionGM{});
            blockMmadOp(cGlobalTensor, aGlobalTensor, bGlobalTensor, indexGlobalTensor, blockShape);
        }
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_SPARSE_MATMUL_TLA_HPP