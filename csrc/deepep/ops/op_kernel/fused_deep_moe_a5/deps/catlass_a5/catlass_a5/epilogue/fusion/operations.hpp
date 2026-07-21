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

#ifndef CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP
#define CATLASS_EPILOGUE_FUSION_OPERATIONS_HPP

#include "catlass_a5/catlass.hpp"

namespace Catlass::Epilogue::Fusion {

// 一元
template <typename T>
struct Exp {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Exp(dst, src, compute_length);
    }
};

template <typename T>
struct Relu {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Relu(dst, src, compute_length);
    }
};

template <typename T>
struct Sqrt {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Sqrt(dst, src, compute_length);
    }
};

template <typename T>
struct RsqrtFast {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Rsqrt(dst, src, compute_length);
    }
};

template <typename T>
struct LeakyRelu {
    T scalar;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::LeakyRelu(dst, src, scalar, compute_length);
    }
};

template <typename T, typename S, AscendC::RoundMode RoundMode = AscendC::RoundMode::CAST_NONE>
struct Cast {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<S> const &src) const
    {
        static_assert(!std::is_same_v<T, S>, "Cast: input type mismatch");
        AscendC::Cast(dst, src, RoundMode, compute_length);
    }
};

template <typename T>
struct Silu {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Silu(dst, src, compute_length);
    }
};

// 二元
template <typename T>
struct Mul {
    template <typename... Inputs>
    CATLASS_DEVICE void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length,
                                   AscendC::LocalTensor<T> const &src0, AscendC::LocalTensor<T> const &src1,
                                   Inputs const &...rest) const
    {
        AscendC::Mul(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Muls {
    T scalarValue;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Muls(dst, src, scalarValue, compute_length);
    }
};

template <typename T>
struct Maxs {
    T scalarValue;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Maxs(dst, src, scalarValue, compute_length);
    }
};

template <typename T>
struct Mins {
    T scalarValue;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Mins(dst, src, scalarValue, compute_length);
    }
};

template <typename T>
struct Add {
    template <typename... Inputs>
    CATLASS_DEVICE void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length,
                                   AscendC::LocalTensor<T> const &src0, AscendC::LocalTensor<T> const &src1,
                                   Inputs const &...rest) const
    {
        AscendC::Add(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Adds {
    T scalarValue;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Adds(dst, src, scalarValue, compute_length);
    }
};

template <typename T>
struct Sub {
    template <typename... Inputs>
    CATLASS_DEVICE void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length,
                                   AscendC::LocalTensor<T> const &src0, AscendC::LocalTensor<T> const &src1,
                                   Inputs const &...rest) const
    {
        AscendC::Sub(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Div {
    template <typename... Inputs>
    CATLASS_DEVICE void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length,
                                   AscendC::LocalTensor<T> const &src0, AscendC::LocalTensor<T> const &src1,
                                   Inputs const &...rest) const
    {
        AscendC::Div(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Max {
    template <typename... Inputs>
    CATLASS_DEVICE void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length,
                                   AscendC::LocalTensor<T> const &src0, AscendC::LocalTensor<T> const &src1,
                                   Inputs const &...rest) const
    {
        AscendC::Max(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

template <typename T>
struct Min {
    template <typename... Inputs>
    CATLASS_DEVICE void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length,
                                   AscendC::LocalTensor<T> const &src0, AscendC::LocalTensor<T> const &src1,
                                   Inputs const &...rest) const
    {
        AscendC::Min(dst, src0, src1, compute_length);
        if constexpr (sizeof...(rest) > 0) {
            AscendC::PipeBarrier<PIPE_V>();
            operator()(dst, compute_length, dst, rest...);
        }
    }
};

// 其他类op
template <typename T>
struct AddRelu {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src0,
                    AscendC::LocalTensor<T> const &src1) const
    {
        AscendC::Add(dst, src0, src1, compute_length);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Relu(dst, dst, compute_length);
    }
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
// Prelu
template <typename T>
struct Prelu {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src0,
                    AscendC::LocalTensor<T> const &src1) const
    {
        AscendC::Prelu(dst, src0, src1, compute_length);
    }
};

template <typename T>
struct Reciprocal {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Divs(dst, T(1), src, compute_length);
    }
};

template <typename T>
struct Rsqrt {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Sqrt(dst, src, compute_length);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Divs(dst, T(1), dst, compute_length);
    }
};

template <typename T>
struct Sigmoid {
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<T> &dst, uint32_t compute_length, AscendC::LocalTensor<T> const &src) const
    {
        AscendC::Muls(dst, src, T(-1), compute_length);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dst, dst, compute_length);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(dst, dst, T(1), compute_length);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Divs(dst, T(1), dst, compute_length);
    }
};
#endif

}  // namespace Catlass::Epilogue::Fusion

#endif
