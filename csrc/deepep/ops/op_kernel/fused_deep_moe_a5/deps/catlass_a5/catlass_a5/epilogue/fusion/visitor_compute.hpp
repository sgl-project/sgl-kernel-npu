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

#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_COMPUTE_HPP

#include "catlass_a5/epilogue/fusion/visitor_impl.hpp"
#include "catlass_a5/epilogue/fusion/operations.hpp"

namespace Catlass::Epilogue::Fusion {

// 辅助函数：使用index_sequence展开tuple来构造OP对象（聚合初始化）
namespace detail {
template <template <class> class Op, class ElementCompute, typename... Scalars, size_t... Is>
CATLASS_DEVICE Op<ElementCompute> construct_op_from_tuple(tla::tuple<Scalars...> const &scalars,
                                                          std::index_sequence<Is...>)
{
    // 使用花括号聚合初始化，要求操作类型必须是聚合类型（无构造函数）
    return Op<ElementCompute>{tla::get<Is>(scalars)...};
}
}  // namespace detail

template <template <class> class ComputeFn, class ElementCompute, typename... Scalars>
struct VisitorCompute : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = ElementCompute;

    // 定义scalar类型别名（Scalars...可以为空）
    using ScalarsTuple = tla::tuple<Scalars...>;

    // Arguments 支持聚合初始化：直接用 {scalar1, scalar2, ...} 初始化
    // 当 Scalars... 为空时，使用 {} 初始化
    struct Arguments {
        ScalarsTuple scalars;
    };

    struct Params {
        ScalarsTuple scalars;

        Params() = default;

        Params(Arguments const &args) : scalars(args.scalars) {}
    };

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const &, Arguments const &args, void *)
    {
        return Params(args);
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const &, Arguments const &)
    {
        return 0;
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const &, Arguments const &)
    {
        return true;
    }

    VisitorCompute() : params() {}

    VisitorCompute(Params const &params_) : params(params_) {}

    Params params;

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<ElementCompute> ubOut;
        uint32_t compute_length;
        ScalarsTuple scalars_tuple;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<ElementCompute> ubOut_, uint32_t compute_length_, Params const &params_)
            : ubOut(ubOut_), compute_length(compute_length_), scalars_tuple(params_.scalars)
        {}

        template <VisitStage Stage, class ArchTag, class TensorC, typename... ElementInputs>
        CATLASS_DEVICE AscendC::LocalTensor<ElementCompute> const &
        visit(TensorC const & /*tensorTile*/, MatrixCoord const & /*alignedTileShape*/,
              MatrixCoord const & /*globalOffset*/, AscendC::LocalTensor<ElementInputs> const &...inputs)
        {
            if constexpr (Stage == VisitStage::COMPUTE) {
                constexpr bool all_inputs_match = (std::is_same_v<ElementInputs, ElementCompute> && ...);
                static_assert(
                    all_inputs_match,
                    "VisitorCompute: input element types must equal ElementCompute. Insert VisitorCast if needed.");

                using Op = ComputeFn<ElementCompute>;

                // 统一处理：使用花括号聚合初始化（Scalars...为空时，展开为空参数，等价于Op{}）
                Op compute_fn = detail::construct_op_from_tuple<ComputeFn, ElementCompute, Scalars...>(
                    scalars_tuple, std::make_index_sequence<sizeof...(Scalars)>{});
                compute_fn(ubOut, compute_length, inputs...);
                AscendC::PipeBarrier<PIPE_V>();
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(Arch::Resource<ArchTag> &resource, uint32_t &ub_offset, uint32_t compute_length)
    {
        auto ubOut = resource.ubBuf.template GetBufferByByte<ElementCompute>(ub_offset);
        ub_offset += compute_length * sizeof(ElementCompute);
        assert(ub_offset <= ArchTag::UB_SIZE);
        return Callbacks(ubOut, compute_length, params);
    }
};

}  // namespace Catlass::Epilogue::Fusion

#endif
