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

#ifndef CATLASS_EPILOGUE_FUSION_TREE_VISITOR_HPP
#define CATLASS_EPILOGUE_FUSION_TREE_VISITOR_HPP

#include "catlass_a5/epilogue/fusion/visitor_impl.hpp"

namespace Catlass::Epilogue::Fusion {

template <class NodeOp, class... ChildOps>
struct TreeVisitor : VisitorImpl<ChildOps..., NodeOp> {
    using VisitorImpl<ChildOps..., NodeOp>::VisitorImpl;

    // Visitor 的输出类型与父节点算子的输出类型保持一致，便于作为其他 TopologicalVisitor 的子节点使用
    using ElementOutput = typename NodeOp::ElementOutput;

    template<class CallbacksImpl>
    struct Callbacks : CallbacksImpl {
        CATLASS_DEVICE
        Callbacks(CallbacksImpl&& impl)
            : CallbacksImpl(static_cast<CallbacksImpl&&>(impl)) {}

        using CallbacksImpl::callbacks_tuple;

        // 辅助函数：收集子节点输出
        template <VisitStage Stage, class ArchTag, class TensorC, typename... Args, int... Is>
        CATLASS_DEVICE auto collect_child_outputs(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            tla::seq<Is...>,
            Args const&... args
        ) {
            return tla::tuple<decltype(tla::get<Is>(callbacks_tuple).template visit<Stage, ArchTag>(
                tensorTile, alignedTileShape, globalOffset, args...))...>(
                tla::get<Is>(callbacks_tuple).template visit<Stage, ArchTag>(
                    tensorTile, alignedTileShape, globalOffset, args...)...
            );
        }

        // 辅助函数：调用父节点
        template <VisitStage Stage, class ArchTag, class TensorC, typename ChildOutputs, int... Is>
        CATLASS_DEVICE auto call_parent_with_outputs(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            ChildOutputs const& child_outputs,
            tla::seq<Is...>
        ) {
            constexpr int Rm1 = sizeof...(ChildOps);
            return tla::get<Rm1>(callbacks_tuple).template visit<Stage, ArchTag>(
                tensorTile, alignedTileShape, globalOffset,
                tla::get<Is>(child_outputs)...
            );
        }

        // 统一的 visit 签名：可变参数
        template <VisitStage Stage, class ArchTag, class TensorC, typename... Args>
        CATLASS_DEVICE auto visit(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            Args const&... args
        ) {
            constexpr int Rm1 = sizeof...(ChildOps);
            
            // 访问所有子节点，收集输出
            auto child_outputs = collect_child_outputs<Stage, ArchTag>(
                tensorTile, alignedTileShape, globalOffset,
                tla::make_seq<Rm1>{},
                args...
            );
            
            // 将子节点输出传给父节点
            return call_parent_with_outputs<Stage, ArchTag>(
                tensorTile, alignedTileShape, globalOffset,
                child_outputs,
                tla::make_seq<Rm1>{}
            );
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        auto base_callbacks = this->VisitorImpl<ChildOps..., NodeOp>::get_callbacks(
            resource, ub_offset, compute_length
        );
        return Callbacks<decltype(base_callbacks)>(
            static_cast<decltype(base_callbacks)&&>(base_callbacks)
        );
    }
};

template <class NodeOp, class... ChildOps>
using EVG = TreeVisitor<NodeOp, ChildOps...>;

} // namespace Catlass::Epilogue::Fusion

#endif
