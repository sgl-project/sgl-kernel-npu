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

#ifndef CATLASS_EPILOGUE_FUSION_VISITOR_CAST_HPP
#define CATLASS_EPILOGUE_FUSION_VISITOR_CAST_HPP

#include "catlass_a5/epilogue/fusion/visitor_impl.hpp"
#include "catlass_a5/epilogue/fusion/operations.hpp"

namespace Catlass::Epilogue::Fusion {

template <class ElementTo, class ElementFrom, AscendC::RoundMode RoundStyle = AscendC::RoundMode::CAST_NONE>
struct VisitorCast : VisitorImpl<> {
    using VisitorImpl<>::VisitorImpl;

    // 输出元素类型与输出阶段元信息
    using ElementOutput = ElementTo;

    struct Arguments {};
    struct Params {};

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const &, Arguments const &, void *)
    {
        return Params();
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

    VisitorCast() {}

    VisitorCast(Params const &) {}

    struct Callbacks : EmptyCallbacks {
        AscendC::LocalTensor<ElementTo> ubOut;
        uint32_t compute_length;

        CATLASS_DEVICE
        Callbacks(AscendC::LocalTensor<ElementTo> ubOut_, uint32_t compute_length_)
            : ubOut(ubOut_), compute_length(compute_length_)
        {}

        template <VisitStage Stage, class ArchTag, class TensorC, typename ElementInput>
        CATLASS_DEVICE AscendC::LocalTensor<ElementTo> const &visit(TensorC const & /*tensorTile*/,  // 不使用
                                                                    MatrixCoord const & /*alignedTileShape*/,  // 不使用
                                                                    MatrixCoord const & /*globalOffset*/,
                                                                    AscendC::LocalTensor<ElementInput> const &input)
        {
            static_assert(std::is_same_v<ElementInput, ElementFrom>, "VisitorCast: input type mismatch");
            static_assert(!std::is_same_v<ElementInput, ElementTo>, "VisitorCast: no need to cast");
            if constexpr (Stage == VisitStage::COMPUTE) {
                Cast<ElementTo, ElementInput, RoundStyle>{}(ubOut, compute_length, input);
                AscendC::PipeBarrier<PIPE_V>();
            }
            return ubOut;
        }
    };

    template <class ArchTag>
    CATLASS_DEVICE auto get_callbacks(Arch::Resource<ArchTag> &resource, uint32_t &ub_offset, uint32_t compute_length)
    {
        auto ubOut = resource.ubBuf.template GetBufferByByte<ElementTo>(ub_offset);
        ub_offset += compute_length * sizeof(ElementTo);
        assert(ub_offset <= ArchTag::UB_SIZE);
        return Callbacks(ubOut, compute_length);
    }

    Params params;
};

}  // namespace Catlass::Epilogue::Fusion

#endif
