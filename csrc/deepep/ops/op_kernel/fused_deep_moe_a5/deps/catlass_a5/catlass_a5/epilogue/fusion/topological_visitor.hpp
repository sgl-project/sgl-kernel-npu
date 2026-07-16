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

#ifndef CATLASS_EPILOGUE_FUSION_TOPOLOGICAL_VISITOR_HPP
#define CATLASS_EPILOGUE_FUSION_TOPOLOGICAL_VISITOR_HPP

#include "catlass_a5/epilogue/fusion/visitor_impl.hpp"

namespace Catlass::Epilogue::Fusion {

namespace detail {

// 获取参数包中的最后一个类型，用于推导组合 Visitor 的输出类型
template <class T, class... Rest>
struct last_type {
    using type = typename last_type<Rest...>::type;
};

template <class T>
struct last_type<T> {
    using type = T;
};

} // namespace detail

// 拓扑顺序访问器：按照 EdgeTuple 指定的依赖顺序访问各节点。
// EdgeTuple 形如 tla::tuple<tla::seq<child_idx...>, ...>，长度等于 Ops 数量，
// 约定最后一个 Op（索引 R-1）为根节点（产生输出）。
template <class EdgeTuple, class... Ops>
struct TopologicalVisitor : VisitorImpl<Ops...> {
    using VisitorImpl<Ops...>::VisitorImpl;

    // Visitor 的输出类型等于根节点（最后一个 Op）的输出类型，
    // 这样 TopologicalVisitor 也可以作为其他 TopologicalVisitor 的子节点参与嵌套
    using ElementOutput = typename detail::last_type<Ops...>::type::ElementOutput;

    template<class CallbacksImpl>
    struct Callbacks : CallbacksImpl {
        CATLASS_DEVICE
        Callbacks(CallbacksImpl&& impl)
            : CallbacksImpl(static_cast<CallbacksImpl&&>(impl)) {}

        using CallbacksImpl::callbacks_tuple;

        // 缓存：按节点索引缓存节点的输出，避免重复访问
        tla::tuple<AscendC::LocalTensor<typename Ops::ElementOutput>...> cache_;
        bool visited_[sizeof...(Ops)] = {false};

        // 重置访问标记（在每个 tile 开始时调用）
        template <int... Is>
        CATLASS_DEVICE void reset_flags_impl(tla::seq<Is...>) {
            // 展开赋值
            auto _ = {(visited_[Is] = false, 0)...};
        }
        CATLASS_DEVICE void reset_flags() {
            reset_flags_impl(tla::make_seq<sizeof...(Ops)>{});
        }

        // 访问节点 i：先访问其子节点，按 EdgeTuple 指定顺序收集输出，再调用第 i 个回调
        template <VisitStage Stage, int I, class ArchTag, class TensorC, class Seq, class... Args>
        CATLASS_DEVICE auto visit_node(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            Seq /*edges_seq*/,  // 形如 tla::seq<child...>
            Args const&... args
        ) {
            if (visited_[I]) {
                return tla::get<I>(cache_);
            }

            // 收集子节点输出为一个 tla::tuple<ChildOutputs...>
            auto child_outputs = collect_children<Stage, I, ArchTag, TensorC>(
                tensorTile, alignedTileShape, globalOffset, Seq{}, args...
            );

            // 将子节点输出按照 Seq 指定顺序展开，调用第 I 个回调
            auto ret = call_current<Stage, I, ArchTag, TensorC>(tensorTile, alignedTileShape, globalOffset, child_outputs);

            // 缓存
            tla::get<I>(cache_) = ret;
            visited_[I] = true;

            return ret;
        }

        // 收集子节点输出的实现：针对 tla::seq 索引序列展开
        template <VisitStage Stage, int I, class ArchTag, class TensorC, int... ChildIs, class... Args>
        CATLASS_DEVICE auto collect_children(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            tla::seq<ChildIs...> /*seq*/,
            Args const&... args
        ) {
            return tla::tuple<decltype(
                this->template visit_node<Stage, ChildIs, ArchTag, TensorC>(
                    tensorTile, alignedTileShape, globalOffset,
                    decltype(tla::get<ChildIs>(EdgeTuple{})){}, args...
                )
            )...>(
                this->template visit_node<Stage, ChildIs, ArchTag, TensorC>(
                    tensorTile, alignedTileShape, globalOffset,
                    decltype(tla::get<ChildIs>(EdgeTuple{})){}, args...
                )...
            );
        }

        // 展开 child_outputs 元组并调用第 I 个算子
        template <VisitStage Stage, int I, class ArchTag, class TensorC, class ChildOutputs, int... Js>
        CATLASS_DEVICE auto call_current_expand(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            ChildOutputs const& child_outputs,
            tla::seq<Js...>
        ) {
            return tla::get<I>(this->callbacks_tuple).template visit<Stage, ArchTag>(
                tensorTile, alignedTileShape, globalOffset,
                tla::get<Js>(child_outputs)...
            );
        }

        template <VisitStage Stage, int I, class ArchTag, class TensorC, class ChildOutputs>
        CATLASS_DEVICE auto call_current(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            ChildOutputs const& child_outputs
        ) {
            constexpr int Num = tla::tuple_size<ChildOutputs>::value;
            return call_current_expand<Stage, I, ArchTag, TensorC>(
                tensorTile, alignedTileShape, globalOffset,
                child_outputs, tla::make_seq<Num>{}
            );
        }

        // 通过 collect_children 直接递归 visit_node

        // 统一入口：以根节点 R-1 开始访问
        template <VisitStage Stage, class ArchTag, class TensorC, typename... Args>
        CATLASS_DEVICE auto visit(
            TensorC const& tensorTile,
            MatrixCoord const& alignedTileShape,
            MatrixCoord const& globalOffset,
            Args const&... args
        ) {
            //每次visit需要重置缓存标记，防止重复访问一个节点多次
            reset_flags(); 
            constexpr int R = sizeof...(Ops);
            using RootEdges = decltype(tla::get<R - 1>(EdgeTuple{}));
            return visit_node<Stage, R - 1, ArchTag, TensorC>(tensorTile, alignedTileShape, globalOffset, RootEdges{}, args...);
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        auto base_callbacks = this->VisitorImpl<Ops...>::get_callbacks(
            resource, ub_offset, compute_length
        );
        return Callbacks<decltype(base_callbacks)>(static_cast<decltype(base_callbacks)&&>(base_callbacks));
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif


