/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DETAIL_TAG_TO_LAYOUT_HPP
#define CATLASS_DETAIL_TAG_TO_LAYOUT_HPP

#include "catlass_a5/layout/layout.hpp"
#include "catlass_a5/numeric_size.hpp"
#include "tla_a5/layout.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Catlass::detail {
////////////////////////////////////////////////////////////////////////////////////////////////////
// For each Catlass::layout, provides its corresponding tla layout types
template <class Element, class LayoutTag>
struct TagToLayout {
    using type = LayoutTag;
};

template <class Element>
struct TagToLayout<Element, layout::RowMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
};

template <class Element>
struct TagToLayout<Element, layout::ColumnMajor> {
    using type = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<tla::Int<1>, int64_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::VectorLayout> {
    using type = tla::Layout<tla::Shape<uint32_t>, tla::Stride<tla::Int<1>>>;
};

template <class Element>
struct TagToLayout<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>,
            tla::Stride<tla::Int<1>, int64_t>>>;
};

template <class Element>
struct TagToLayout<Element, layout::zZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<ELE_NUM_PER_C0>, int64_t>,
            tla::Stride<tla::Int<1>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

template <class Element>
struct TagToLayout<Element, layout::nZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<1>, int64_t>,
            tla::Stride<tla::Int<ELE_NUM_PER_C0>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

template <class Element>
struct TagToLayout<Element, layout::NC1HWC0> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    using type = tla::Layout<
        tla::Shape<uint32_t, uint32_t, uint32_t, uint32_t, tla::Int<ELE_NUM_PER_C0>>, 
        tla::Stride<int64_t, int64_t, int64_t, tla::Int<ELE_NUM_PER_C0>, tla::Int<1>>,
        tla::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>;
};

template <class Element>
struct TagToLayout<Element, layout::CI1KHKWCOCI0> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    using type = tla::Layout<
        tla::Shape<uint32_t, uint32_t, uint32_t, uint32_t, tla::Int<ELE_NUM_PER_C0>>, 
        tla::Stride<int64_t, int64_t, int64_t, tla::Int<ELE_NUM_PER_C0>, tla::Int<1>>,
        tla::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>>;
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <>
struct TagToLayout<float8_e8m0_t, layout::zZ> {
    static constexpr uint32_t ELE_NUM_PER_C0 = 2;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = 32;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<ELE_NUM_PER_C0>, int64_t>,
            tla::Stride<tla::Int<1>, tla::Int<ELE_NUM_PER_FRACTAL>>>>;
};

template <>
struct TagToLayout<float8_e8m0_t, layout::nN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = 2;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = 32;
    using type = tla::Layout<
        tla::Shape<tla::Shape<tla::Int<ELE_NUM_PER_C0>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
        tla::Stride<tla::Stride<tla::Int<1>, tla::Int<ELE_NUM_PER_FRACTAL>>,
            tla::Stride<tla::Int<ELE_NUM_PER_C0>, int64_t>>>;
};
#endif

// Convenience aliases
template <class Element, class LayoutTag>
using TagToLayout_t = typename TagToLayout<Element, LayoutTag>::type;

constexpr uint32_t ELE_NUM_PER_FRACTAL_L0C = 256;
using LayoutL0C = tla::Layout<
    tla::Shape<tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>, tla::Shape<tla::Int<C0_NUM_PER_FRACTAL>, uint32_t>>,
    tla::Stride<tla::Stride<tla::Int<C0_NUM_PER_FRACTAL>, tla::Int<ELE_NUM_PER_FRACTAL_L0C>>,
        tla::Stride<tla::Int<1>, int64_t>>>;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::detail

#endif // CATLASS_DETAIL_TAG_TO_LAYOUT_HPP