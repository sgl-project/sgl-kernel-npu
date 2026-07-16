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

#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_BASE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_IMPL_BASE_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/status.hpp"
#include "tla_a5/int_tuple.hpp"

namespace Catlass::Epilogue::Fusion {

enum class VisitStage : uint8_t {
    LOAD = 0,      
    COMPUTE = 1,   
    STORE = 2,
};

template <class... Ops>
struct VisitorImplBase {
    using Arguments = tla::tuple<typename Ops::Arguments...>;
    using Params = tla::tuple<typename Ops::Params...>;

    template <class ProblemShape>
    static constexpr Params
    to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                auto ret = Op::to_underlying_arguments(problem_shape, op_args, op_workspace);
                if (op_workspace != nullptr) {
                    size_t sz = Op::get_workspace_size(problem_shape, op_args);
                    op_workspace += (sz + 31) & ~31; // 32 字节对齐
                }
                return ret;
            },
            [](auto&&... op_params) -> tla::tuple<tla::remove_cvref_t<decltype(op_params)>...> { 
                return tla::tuple<tla::remove_cvref_t<decltype(op_params)>...>(op_params...); 
            }
        );
    }

    template <class ProblemShape>
    static bool
    can_implement(ProblemShape const& problem_shape, Arguments const& args) {
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                return Op::can_implement(problem_shape, op_args);
            },
            [](auto&&... ok) { return (true && ... && ok); }
        );
    }

    template <class ProblemShape>
    static size_t
    get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                size_t sz = Op::get_workspace_size(problem_shape, op_args);
                return (sz + 31) & ~31; // 32 字节对齐
            },
            [](auto&&... rounded_sizes) { return (size_t{0} + ... + rounded_sizes); }
        );
    }

    // 为每个节点按树序进行工作区初始化，分配并前进指针。
    template <class ProblemShape>
    static Status
    initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
        Status status = Status::kSuccess;
        uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);

        return tla::transform_apply(
            tla::tuple<Ops...>{}, args,
            [&](auto&& op_tag, auto const& op_args) {
                if (status != Status::kSuccess) {
                    return status;
                }
                using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                status = Op::initialize_workspace(problem_shape, op_args, op_workspace);
                if (op_workspace != nullptr) {
                    size_t sz = Op::get_workspace_size(problem_shape, op_args);
                    op_workspace += (sz + 31) & ~size_t(31);
                }
                return status;
            },
            [&](auto const&... ) { return status; }
        );
    }

    CATLASS_HOST_DEVICE
    VisitorImplBase() {}

    CATLASS_HOST_DEVICE
    VisitorImplBase(Params const& params)
        : ops(
            tla::transform_apply(
                tla::tuple<Ops...>{}, params,
                [](auto&& op_tag, auto const& op_params) {
                    using Op = typename tla::remove_cvref_t<decltype(op_tag)>;
                    return Op(op_params);
                },
                [](auto&&... built_ops) -> tla::tuple<tla::remove_cvref_t<decltype(built_ops)>...> { 
                    return tla::tuple<tla::remove_cvref_t<decltype(built_ops)>...>(built_ops...); 
                }
            )
        )
    {}

    tla::tuple<Ops...> ops;
};

} // namespace Catlass::Epilogue::Fusion

#endif
