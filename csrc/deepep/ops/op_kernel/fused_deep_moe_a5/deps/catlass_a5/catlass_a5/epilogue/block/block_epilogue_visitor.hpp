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

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/dispatch_policy.hpp"
#include "catlass_a5/epilogue/fusion/visitor_impl_base.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Epilogue::Block {

template <
    bool USE_UB_WORKSPACE_,
    class ArchTag_,
    class ComputeLength_,
    class EVG_,
    class ElementC_
>
class BlockEpilogue<
    EpilogueVisitor<USE_UB_WORKSPACE_>,
    ArchTag_,
    ComputeLength_,
    EVG_,
    ElementC_
> {
public:
    static constexpr bool USE_UB_WORKSPACE = USE_UB_WORKSPACE_;
    static constexpr uint32_t COMPUTE_LENGTH = ComputeLength_::value;
    static_assert(COMPUTE_LENGTH % BYTE_PER_C0 == 0, "COMPUTE_LENGTH must be divisible by BYTE_PER_C0");
    using EVG = EVG_;
    using ArchTag = ArchTag_;
    using ElementC = ElementC_;

    struct Params {
        typename EVG::Params evg_params;

        Params() {}

        Params(typename EVG::Params const& evg_params_)
            : evg_params(evg_params_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag>& resource, Params const& params)
        : params(params), evg(params.evg_params), resource_(resource)
    {
        if ASCEND_IS_AIV {
            int32_t evVMTE2 = 0;   // V_MTE2
            int32_t evMTE2V = 0;   // MTE2_V
            int32_t evMTE3V = 0;   // MTE3_V
            int32_t evVMTE3 = 0;   // V_MTE3

            for (int i = 0; i < 2; ++i) {
                eventVMTE2[i] = evVMTE2++;
                eventMTE2V[i] = evMTE2V++;
                eventMTE3V[i] = evMTE3V++;
                eventVMTE3[i] = evVMTE3++;
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[0]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[1]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[0]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[1]);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[1]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[1]);
        }
    }

    // 仅 UB Visitor 路径使用，用于UB 中构建 MMAD 目标 tensor
    CATLASS_DEVICE auto GetMmadUbTensor(GemmCoord const& actualBlockShape)
    {
        static_assert(USE_UB_WORKSPACE,
                      "GetMmadUbTensor is only valid when using UB workspace (EpilogueVisitor<true>)");

        uint32_t m = actualBlockShape.m();
        uint32_t n = actualBlockShape.n();
        uint32_t strideC = RoundUp<BYTE_PER_C0>(n);
        auto layoutC = tla::MakeLayout(
            tla::MakeShape(m, n), tla::MakeStride(strideC, tla::Int<1>{})
        );
        auto ubAcc = resource_.ubBuf.template GetBufferByByte<ElementC>(0);
        return tla::MakeTensor(ubAcc, layoutC, Arch::PositionUB{});
    }

    template <class TensorC>
    CATLASS_DEVICE
    void operator()(
        GemmCoord const& blockShapeMNK,
        GemmCoord const& blockCoordMNK,
        GemmCoord const& actualBlockShapeMNK,
        TensorC const& tensorBlockC
    )
    {
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        MatrixCoord subblockShape{
            CeilDiv(actualBlockShape.row(), static_cast<uint32_t>(AscendC::GetSubBlockNum())),
            actualBlockShape.column()
        };
        MatrixCoord subblockCoord{AscendC::GetSubBlockIdx(), 0};
        MatrixCoord actualSubblockShape = MatrixCoord::Min(
            subblockShape, actualBlockShape - subblockCoord * subblockShape);
        MatrixCoord subblockOffset = subblockCoord * subblockShape;

        // 对 GM Visitor：从 GM C 中切出对应 subblock；
        // 对 UB Visitor ：fixpipe后，有效结果在上半部分，按 (0, 0) 开始处理当前 block 的 C。
        auto tensorSubblockC = GetTile(tensorBlockC,
            tla::MakeCoord(
                USE_UB_WORKSPACE ? 0 : subblockOffset.row(),
                USE_UB_WORKSPACE ? 0 : subblockOffset.column()
            ),
            tla::MakeShape(actualSubblockShape.row(), actualSubblockShape.column()));

        // 分配 UB 空间并获取两套 callbacks（双缓冲）
        // 根据 DispatchPolicy 决定 UB 起始偏移：
        // - EpilogueVisitor<false> : 使用 GM workspace，UB 从 0 开始
        // - EpilogueVisitor<true>  : 使用 UB workspace，UB 从 L0C_SIZE / 2 开始，[0, L0C_SIZE / 2)分配给tensorC
        uint32_t ub_offset0 = USE_UB_WORKSPACE ? (ArchTag::L0C_SIZE / 2) : 0;
        auto callbacks0 = evg.get_callbacks(
            resource_, ub_offset0, COMPUTE_LENGTH
        );
        uint32_t ub_offset1 = ub_offset0;
        auto callbacks1 = evg.get_callbacks(
            resource_, ub_offset1, COMPUTE_LENGTH
        );

        uint32_t rows = actualSubblockShape.row();
        uint32_t cols = actualSubblockShape.column();

        // 遍历所有 tile，实现双缓冲流水
        uint32_t ubListId = 0;  // 0或1，交替使用
        
        for (uint32_t r = 0; r < rows; ) {
            auto& cbs = ((ubListId & 1) ? callbacks1 : callbacks0);

            // 检查是否需要列分块
            if (cols <= COMPUTE_LENGTH) {
                // 列宽 <= COMPUTE_LENGTH，可以处理完整行宽，一次做多行
                uint32_t colsAligned = RoundUp<BYTE_PER_C0>(cols);
                uint32_t maxRowsPerTile = COMPUTE_LENGTH / colsAligned;
                if (maxRowsPerTile == 0) maxRowsPerTile = 1;  // 防止除零
                
                uint32_t remainRows = rows - r;
                uint32_t tileRows = (remainRows < maxRowsPerTile) ? remainRows : maxRowsPerTile;
                
                MatrixCoord tileShape{tileRows, cols};
                MatrixCoord localTileOffset{r, 0};
                
                // 计算对齐的 tile shape
                MatrixCoord alignedTileShape{
                    tileShape.row(),
                    colsAligned
                };
                
                // 统一流水：执行一次 tile 的 Load-Compute-Store
                // 从tensorSubblockC获取信息，只传递必要参数
                MatrixCoord globalOffset = blockOffset + subblockOffset + localTileOffset;
                run_tile(cbs, tensorSubblockC, localTileOffset, tileShape, alignedTileShape, ubListId, globalOffset);
                r += tileRows;
            } else { 
                // 列宽 > COMPUTE_LENGTH，需要列分块，每次处理1行
                for (uint32_t c = 0; c < cols; ) {
                    uint32_t remainCols = cols - c;
                    uint32_t tileCols = (remainCols < COMPUTE_LENGTH) ? remainCols : COMPUTE_LENGTH;
                    
                    uint32_t colsAligned = RoundUp<BYTE_PER_C0>(tileCols);

                    MatrixCoord tileShape{1, tileCols};
                    MatrixCoord localTileOffset{r, c};
                    
                    // 计算对齐的 tile shape
                    MatrixCoord alignedTileShape{
                        tileShape.row(),
                        colsAligned
                    };
                    
                    // 统一流水：执行一次 tile 的 Load-Compute-Store
                    // 从tensorSubblockC获取信息，只传递必要参数
                    MatrixCoord globalOffset = blockOffset + subblockOffset + localTileOffset;
                    run_tile(cbs, tensorSubblockC, localTileOffset, tileShape, alignedTileShape, ubListId, globalOffset);
                    c += tileCols;
                }
                
                r += 1;  // 处理完一行
            }

            ubListId = 1 - ubListId; // Buffer 轮转
        }
    }

    private:
    template <class Callbacks, class TensorC>
    CATLASS_DEVICE void run_tile(
        Callbacks& cbs,
        TensorC const& tensorSubblockC,
        MatrixCoord const& localTileOffset,
        MatrixCoord const& actualTileShape,
        MatrixCoord const& alignedTileShape,
        uint32_t ubListId,
        MatrixCoord const& globalOffset
    ) {
        // 创建acc tile tensor
        auto tensorTile = GetTile(tensorSubblockC,
            tla::MakeCoord(localTileOffset.row(), localTileOffset.column()),
            tla::MakeShape(actualTileShape.row(), actualTileShape.column()));
        
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[ubListId]);
        cbs.template visit<Epilogue::Fusion::VisitStage::LOAD, ArchTag>(tensorTile, alignedTileShape, globalOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMTE2V[ubListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[ubListId]);
        cbs.template visit<Epilogue::Fusion::VisitStage::COMPUTE, ArchTag>(tensorTile, alignedTileShape, globalOffset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventVMTE2[ubListId]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3[ubListId]);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVMTE3[ubListId]);
        cbs.template visit<Epilogue::Fusion::VisitStage::STORE, ArchTag>(tensorTile, alignedTileShape, globalOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventMTE3V[ubListId]);
    }

    Params params;
    EVG evg;
    Arch::Resource<ArchTag>& resource_;
    int32_t eventVMTE2[2];
    int32_t eventMTE2V[2];
    int32_t eventMTE3V[2];
    int32_t eventVMTE3[2];
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_EPILOGUE_VISITOR_HPP
