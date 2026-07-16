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

#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_AUX_STORE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_AUX_STORE_HPP

#include "catlass_a5/epilogue/fusion/visitor_impl.hpp"
#include "catlass_a5/epilogue/fusion/operations.hpp"
#include "catlass_a5/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "tla_a5/tensor.hpp"
#include "tla_a5/layout.hpp"
#include "catlass_a5/layout/layout.hpp"

namespace Catlass::Epilogue::Fusion {

template<
  class Element,
  class Layout
>
struct VisitorAuxStore : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = Element;

    struct Arguments {
        GM_ADDR ptr_aux = nullptr;
        Layout layout = {};
    };

    struct Params {
        GM_ADDR ptr_aux;
        Layout layout;

            Params() {}

            Params(GM_ADDR ptr_aux_, Layout const& layout_)
            : ptr_aux(ptr_aux_), layout(layout_) {}
    };

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
        return Params(args.ptr_aux, args.layout);
    }

    template <class ProblemShape>
    static size_t
    get_workspace_size(ProblemShape const&, Arguments const&) {
        return 0;
    }

    template <class ProblemShape>
    static bool
    can_implement(ProblemShape const&, Arguments const& args) {
        return args.ptr_aux != nullptr;
    }

    VisitorAuxStore() {}

    VisitorAuxStore(Params const& params_) : params(params_) {}

    struct Callbacks : EmptyCallbacks {
        Params const* params_ptr;

        CATLASS_DEVICE
        Callbacks(Params const* params_ptr_)
            : params_ptr(params_ptr_) {}

        template <VisitStage Stage, class ArchTag, class TensorC, typename ElementInput>
        CATLASS_DEVICE AscendC::LocalTensor<ElementInput> const& visit(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            AscendC::LocalTensor<ElementInput> const& input
        ) {
            static_assert(std::is_same_v<ElementInput, Element>,
                          "VisitorAuxStore: element type mismatch. Insert VisitorCast<...> before store.");
            if constexpr (Stage == VisitStage::STORE) {
                // 写回 GM（使用全局坐标，tile 封装处理跨距）
                if (params_ptr->ptr_aux != nullptr) {
                    // 从tensor获取actualTileShape
                    auto actualRows = tla::get<0>(tensorTile.shape());
                    auto actualCols = tla::get<1>(tensorTile.shape());
                    
                    // TLA Layout: 创建aux tensor并使用GetTile创建tile视图
                    AscendC::GlobalTensor<Element> gmAux;
                    gmAux.SetGlobalBuffer((__gm__ Element*)(params_ptr->ptr_aux));
                    auto tensorAux = tla::MakeTensor(gmAux, params_ptr->layout, Arch::PositionGM{});
                    
                    // 使用globalOffset创建aux tensor的tile视图
                    auto tensorTileAux = GetTile(tensorAux,
                        tla::MakeCoord(globalOffset.row(), globalOffset.column()),
                        tla::MakeShape(actualRows, actualCols));
                    
                    // 创建UB tensor
                    auto layoutUb = tla::MakeLayout(
                        tla::MakeShape(actualRows, actualCols),
                        tla::MakeStride(alignedTileShape.column(), tla::Int<1>{})
                    );
                    auto tensorUb = tla::MakeTensor(input, layoutUb, Arch::PositionUB{});
                    
                    // 使用TLA tile copy
                    using CopyUb2GmTlaT = Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(tensorUb), decltype(tensorTileAux)>;
                    CopyUb2GmTlaT copyUb2GmTla{};
                    copyUb2GmTla(tensorTileAux, tensorUb);
                }
            }
            // 透传返回输入以便继续参与 EVG 组合
            return input;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        return Callbacks(&params);
    }

    Params params;
};

} // namespace Catlass::Epilogue::Fusion

#endif



