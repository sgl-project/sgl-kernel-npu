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

#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_HPP

#include "catlass_a5/arch/resource.hpp"
#include "catlass_a5/epilogue/fusion/visitor_impl_base.hpp"
#include "catlass_a5/layout/layout.hpp"
#include "tla_a5/int_tuple.hpp"

namespace Catlass::Epilogue::Fusion {

// 空回调（用于叶子节点）
struct EmptyCallbacks {
    CATLASS_DEVICE void begin_epilogue() {}
    CATLASS_DEVICE void end_epilogue() {}
};

template <class... Ops>
struct VisitorImpl : VisitorImplBase<Ops...> {
    using VisitorImplBase<Ops...>::VisitorImplBase;
    using VisitorImplBase<Ops...>::ops;

    // 为所有节点提供一个标准的空实现：按需可被覆盖
    template <class ProblemShape, class OpArgs>
    static Catlass::Status initialize_workspace(
        ProblemShape const&, OpArgs const&, void*
    ) {
        return Catlass::Status::kSuccess;
    }

    template <class CallbacksTuple>
    struct Callbacks {
        CallbacksTuple callbacks_tuple;

        CATLASS_DEVICE
        Callbacks(CallbacksTuple&& cbs) : callbacks_tuple(static_cast<CallbacksTuple&&>(cbs)) {}

        CATLASS_DEVICE void begin_epilogue() {
            tla::for_each(callbacks_tuple, [](auto& cb) { cb.begin_epilogue(); });
        }

        CATLASS_DEVICE void end_epilogue() {
            tla::for_each(callbacks_tuple, [](auto& cb) { cb.end_epilogue(); });
        }
    };

    template <class ArchTag, int... Is>
    CATLASS_DEVICE auto get_callbacks_impl(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length,
        tla::seq<Is...>
    ) {
        auto tuple_cbs = tla::tuple<
            decltype(tla::get<Is>(ops).get_callbacks(resource, ub_offset, compute_length))...
        >(
            tla::get<Is>(ops).get_callbacks(resource, ub_offset, compute_length)...
        );
        return Callbacks<decltype(tuple_cbs)>(static_cast<decltype(tuple_cbs)&&>(tuple_cbs));
    }

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(
        Arch::Resource<ArchTag>& resource,
        uint32_t& ub_offset,
        uint32_t compute_length
    ) {
        return get_callbacks_impl(
            resource, ub_offset, compute_length,
            tla::make_seq<sizeof...(Ops)>{}
        );
    }
};

} // namespace Catlass::Epilogue::Fusion

#endif
