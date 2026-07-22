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

#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_ROW_BROADCAST_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_ROW_BROADCAST_HPP

#include "catlass_a5/epilogue/fusion/visitor_impl.hpp"
#include "catlass_a5/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"
#include "catlass_a5/layout/layout.hpp"

namespace Catlass::Epilogue::Fusion {

// Row-broadcast a 1xN GM vector to an MxN UB tile for the current tile.
// LOAD stage: load GM row segment [0, globalOffset.column() : cols] -> UB (1 x cols),
// COMPUTE stage: replicate across rows into an tile shape UB buffer and return it.
// STORE stage: no-op, just return the cached UB buffer.
template <class Element, class Layout>
struct VisitorRowBroadcast : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    using ElementOutput = Element;

    struct Arguments {
        GM_ADDR ptr_row = nullptr;  // GM address of 1 x N row vector
        Layout layout = {};         // layout over (1, N)
    };

    struct Params {
        GM_ADDR ptr_row;
        Layout layout;

        Params() {}

        Params(GM_ADDR ptr_row_, Layout const &layout_) : ptr_row(ptr_row_), layout(layout_) {}
    };

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const &, Arguments const &args, void *)
    {
        return Params(args.ptr_row, args.layout);
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const &, Arguments const &)
    {
        return 0;
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const &, Arguments const &args)
    {
        return args.ptr_row != nullptr;
    }

    VisitorRowBroadcast() {}

    VisitorRowBroadcast(Params const &params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<Element> ubOut;
        Params const *params_ptr;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<Element> ubOut_, Params const *params_ptr_, uint32_t compute_length_)
            : ubOut(ubOut_), params_ptr(params_ptr_), compute_length(compute_length_)
        {}

        template <VisitStage Stage, class ArchTag, class TensorC, typename... Args>
        CATLASS_DEVICE AscendC::LocalTensor<Element> const &visit(TensorC const &tensorTile,
                                                                  MatrixCoord const &alignedTileShape,
                                                                  MatrixCoord const &globalOffset,
                                                                  Args const &.../*unused*/
        )
        {
            // 从tensor获取actualTileShape
            auto &actualRows = tla::get<0>(tensorTile.shape());
            auto &actualCols = tla::get<1>(tensorTile.shape());
            auto &alignedCols = alignedTileShape.column();

            if constexpr (Stage == VisitStage::LOAD) {
                // 创建UB tensor用于存储第一行（1 x actualCols）
                auto layoutUbRow = tla::MakeLayout(tla::MakeShape(tla::Int<1>{}, actualCols),
                                                   tla::MakeStride(alignedCols, tla::Int<1>{}));
                auto tensorUbRow = tla::MakeTensor(ubOut, layoutUbRow, Arch::PositionUB{});

                // TLA Layout: 创建GM row tensor并使用GetTile创建tile视图
                AscendC::GlobalTensor<Element> gmRow;
                gmRow.SetGlobalBuffer((__gm__ Element *)(params_ptr->ptr_row));
                auto tensorRow = tla::MakeTensor(gmRow, params_ptr->layout, Arch::PositionGM{});

                auto tensorTileRow = GetTile(tensorRow, tla::MakeCoord(uint32_t(0), globalOffset.column()),
                                             tla::MakeShape(tla::Int<1>{}, actualCols));

                // 使用TLA tile copy
                using CopyGm2UbTlaT =
                    Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(tensorTileRow), decltype(tensorUbRow)>;
                CopyGm2UbTlaT copyGm2UbTla{};
                copyGm2UbTla(tensorUbRow, tensorTileRow);
            }
            if constexpr (Stage == VisitStage::COMPUTE) {
                for (uint32_t r = 1; r < actualRows; ++r) {
                    // copy row 0 -> row r，使用 aligned stride
                    AscendC::DataCopy(ubOut[r * alignedCols],  // dst start at offset r*alignedCols
                                      ubOut[0],                // src start at offset 0
                                      alignedCols);
                }
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(Arch::Resource<ArchTag> &resource, uint32_t &ub_offset, uint32_t compute_length)
    {
        auto ubOut = resource.ubBuf.template GetBufferByByte<Element>(ub_offset);
        ub_offset += compute_length * sizeof(Element);
        assert(ub_offset <= ArchTag::UB_SIZE);
        return Callbacks(ubOut, &params, compute_length);
    }

    Params params;
};

}  // namespace Catlass::Epilogue::Fusion

#endif
