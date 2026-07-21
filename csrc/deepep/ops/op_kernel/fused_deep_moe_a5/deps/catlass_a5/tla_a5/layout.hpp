/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_LAYOUT_HPP
#define TLA_LAYOUT_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/numeric_size.hpp"
#include "tla_a5/numeric/integral_constant.hpp"
#include "tla_a5/numeric/math.hpp"
#include "tla_a5/tuple.hpp"
#include "tla_a5/int_tuple.hpp"
#include "catlass_a5/layout/layout.hpp"

namespace tla {

// Aliases

template <class... Shapes>
using Shape = tla::tuple<Shapes...>;

template <class... Strides>
using Stride = tla::tuple<Strides...>;

template <class... Coords>
using Coord = tla::tuple<Coords...>;

template <class... Ts>
CATLASS_HOST_DEVICE constexpr Shape<Ts...> MakeShape(Ts const &...t)
{
    return {t...};
}
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Stride<Ts...> MakeStride(Ts const &...t)
{
    return {t...};
}
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Coord<Ts...> MakeCoord(Ts const &...t)
{
    return {t...};
}

namespace detail {

// Type trait to generate OriginShape type with rank=R, depth=1, element type=uint32_t
template <int Rank, class Sequence = void>
struct MakeOriginShapeTypeImpl;

template <int Rank, size_t... Is>
struct MakeOriginShapeTypeImpl<Rank, tla::index_sequence<Is...>> {
    template <size_t>
    using repeat_type = uint32_t;
    using type = Shape<repeat_type<Is>...>;
};

template <class Stride>
using MakeOriginShapeType =
    typename MakeOriginShapeTypeImpl<rank_v<Stride>, tla::make_index_sequence<rank_v<Stride>>>::type;

struct UnpackedMakeOriginShapeU32 {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr auto operator()(T const &...a) const
    {
        return MakeShape(static_cast<uint32_t>(a)...);
    }
};

}  // namespace detail

//
// Layout
//
// 自动推导 OriginShape 类型为 Shape<uint32_t...>，rank=rank_v<Stride>
template <class Shape, class Stride, class OriginShape = detail::MakeOriginShapeType<Stride>>
struct Layout : private tla::tuple<Shape, Stride, OriginShape> {
    // NOTE: This defaults static Shapes/Strides correctly, but not dynamic
    CATLASS_HOST_DEVICE constexpr Layout(Shape const &shape = {}, Stride const &stride = {},
                                         OriginShape const &originShape = {})
        : tla::tuple<Shape, Stride, OriginShape>(shape, stride, originShape)
    {}

    //
    // Accessors
    //

    static constexpr int rank = rank_v<Stride>;
    static constexpr int depth = depth_v<Stride>;

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) shape()
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> &>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) shape() const
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const &>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) stride()
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> &>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) stride() const
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const &>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) originShape()
    {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> &>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) originShape() const
    {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const &>(*this));
    }

    template <class Coord>
    CATLASS_HOST_DEVICE constexpr auto operator()(Coord const &coord) const
    {
        return crd2offset(coord, shape(), stride());
    }
};

// Layout construction

template <class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(Shape const &shape, Stride const &stride, OriginShape const &originShape)
{
    static_assert(is_tuple<Shape>::value || is_integral<Shape>::value);
    static_assert(is_tuple<Stride>::value || is_integral<Stride>::value);
    static_assert(is_tuple<OriginShape>::value || is_integral<OriginShape>::value);
    return Layout<Shape, Stride, OriginShape>(shape, stride, originShape);
}

template <class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(Shape const &shape, Stride const &stride)
{
    static_assert(is_tuple<Shape>::value || is_integral<Shape>::value);
    static_assert(is_tuple<Stride>::value || is_integral<Stride>::value);
    // 计算默认的 originShape：将 shape 扁平化为 depth=1，并将每个维度归一化为 uint32_t
    return MakeLayout(shape, stride, tla::transform_apply(shape, Product{}, detail::UnpackedMakeOriginShapeU32{}));
}

// Convenience tags for common layouts

template <class LayoutTag>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutFromTag(LayoutTag const &tag)
{
    static_assert(std::is_same_v<LayoutTag, Catlass::layout::RowMajor> ||
                      std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
                      std::is_same_v<LayoutTag, Catlass::layout::VectorLayout> ||
                      std::is_same_v<LayoutTag, Catlass::layout::zN> ||
                      std::is_same_v<LayoutTag, Catlass::layout::nZ> || std::is_same_v<LayoutTag, Catlass::layout::L0C>,
                  "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::RowMajor or"
                  "Catlass::layout::ColumnMajor or Catlass::layout::VectorLayout or Catlass::layout::zN or "
                  "Catlass::layout::nZ or Catlass::layout::L0C");

    if constexpr (std::is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        return MakeLayout(MakeShape(tag.shape(0)), MakeStride(tag.stride(0)), MakeShape(tag.shape(0)));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        return MakeLayout(MakeShape(tag.shape(0), tag.shape(1)), MakeStride(tag.stride(0), Int<1>{}),
                          MakeShape(tag.shape(0), tag.shape(1)));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        return MakeLayout(MakeShape(tag.shape(0), tag.shape(1)), MakeStride(Int<1>{}, tag.stride(1)),
                          MakeShape(tag.shape(0), tag.shape(1)));
    } else {  // zN or nZ or L0C
        return MakeLayout(
            MakeShape(MakeShape(tag.shape(0), tag.shape(1)), MakeShape(tag.shape(2), tag.shape(3))),
            MakeStride(MakeStride(tag.stride(0), tag.stride(1)), MakeStride(tag.stride(2), tag.stride(3))),
            MakeShape(tag.orgShape(0), tag.orgShape(1)));
    }
}

// Return the shape of a mode
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape> &layout)
{
    return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape> const &layout)
{
    return layout.template shape<Is...>();
}

// Return the stride of a mode
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape> &layout)
{
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape> const &layout)
{
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape> &layout)
{
    return layout.template originShape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape> const &layout)
{
    return layout.template originShape<Is...>();
}

// Return the rank of layout
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto rank(Layout<Shape, Stride, OriginShape> const &layout)
{
    return rank(shape<Is...>(layout));
}

// Return the depth of the layout
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto depth(Layout<Shape, Stride, OriginShape> const &layout)
{
    return depth(shape<Is...>(layout));
}

// Return the offset of coord
template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2offset(Coord const &coord, Shape const &shape, Stride const &stride);

namespace detail {

template <class Coord, class Shape, class Stride, int... Is>
CATLASS_HOST_DEVICE constexpr auto crd2offset_ttt(Coord const &coord, Shape const &shape, Stride const &stride,
                                                  seq<Is...>)
{
    return (... + crd2offset(get<Is>(coord), get<Is>(shape), get<Is>(stride)));
}

template <class CInt, class STuple, class DTuple, int I0, int... Is>
CATLASS_HOST_DEVICE constexpr auto crd2offset_itt(CInt const &coord, STuple const &shape, DTuple const &stride,
                                                  seq<I0, Is...>)
{
    if constexpr (sizeof...(Is) == 0) {  // Avoid recursion and mod on single/last iter
        return crd2offset(coord, get<I0>(shape), get<I0>(stride));
    } else if constexpr (is_constant<0, CInt>::value) {
        return crd2offset(_0{}, get<I0>(shape), get<I0>(stride)) +
               (_0{} + ... + crd2offset(_0{}, get<Is>(shape), get<Is>(stride)));
    } else {  // General case
        return crd2offset(coord % Product{}(get<I0>(shape)), get<I0>(shape), get<I0>(stride)) +
               crd2offset_itt(coord / Product{}(get<I0>(shape)), shape, stride, seq<Is...>{});
    }
}

}  // end namespace detail

template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2offset(Coord const &coord, Shape const &shape, Stride const &stride)
{
    if constexpr (is_tuple<Coord>::value) {
        if constexpr (is_tuple<Shape>::value) {  // tuple tuple tuple
            static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched Ranks");
            static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2offset_ttt(coord, shape, stride, tuple_seq<Coord>{});
        } else {  // tuple "int" "int"
            static_assert(sizeof(Coord) == 0, "Invalid parameters");
        }
    } else {
        if constexpr (is_tuple<Shape>::value) {  // "int" tuple tuple
            static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2offset_itt(coord, shape, stride, tuple_seq<Shape>{});
        } else {  // "int" "int" "int"
            return coord * stride;
        }
    }
}

template <class Layout>
struct is_layout : false_type {};
template <class Shape, class Stride, class OriginShape>
struct is_layout<Layout<Shape, Stride, OriginShape>> : true_type {};

// Layout Check
namespace detail {

template <class Layout, class Enable = void>
struct isVector {
    static bool const value = false;
};

template <class Layout>
struct isVector<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 1>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Layout, class Enable = void>
struct isRowMajor {
    static bool const value = false;
};

template <class Layout>
struct isRowMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<1>(Layout{}) == 1);
};

template <class Layout, class Enable = void>
struct isColumnMajor {
    static bool const value = false;
};

template <class Layout>
struct isColumnMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct iszN {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszN<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
                     rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<1, 0>(Layout{}) == 1 && stride<0, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

/*
For matmul m axis is not c0 Aligned.
Exp: oriShape(m, k) : (127, 256)
zNUnAlign shape:((127, 1), (16, 256/16))  zN shape: ((16, Ceil(127/16)), (16, 256/16))
*/
template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct iszNUnAlign {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszNUnAlign<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<Element>::value;
    static bool const value = (shape<0, 1>(Layout{}) == 1 && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
                               stride<0, 0>(Layout{}) == ELE_NUM_PER_C0 && stride<1, 0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable = void>
struct iszZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszZ<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<1, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isnZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct isnZ<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == ELE_NUM_PER_C0 && shape<1, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL &&
         stride<0, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleForRowMajorA {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForRowMajorA<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 1 && rank_v<decltype(stride<1>(Layout{}))> == 2 &&
                     !is_constant<2, decltype(stride<0>(Layout{}))>::value &&
                     ((rank_v<decltype(shape<0>(Layout{}))> == 1 && rank_v<decltype(shape<1>(Layout{}))> == 2) ||
                      (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleForColumnMajorA {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForColumnMajorA<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 1 && rank_v<decltype(stride<1>(Layout{}))> == 2 &&
                     is_constant<2, decltype(stride<0>(Layout{}))>::value &&
                     ((rank_v<decltype(shape<0>(Layout{}))> == 1 && rank_v<decltype(shape<1>(Layout{}))> == 2) ||
                      (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void, class Enable3 = void>
struct isMxScaleForRowMajorB {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForRowMajorB<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 1>,
    std::enable_if_t<!is_constant<2, decltype(stride<0, 1>(Layout{}))>::value &&
                     ((rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 1) ||
                      (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void, class Enable3 = void>
struct isMxScaleForColumnMajorB {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForColumnMajorB<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 1>,
    std::enable_if_t<is_constant<2, decltype(stride<0, 1>(Layout{}))>::value &&
                     ((rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 1) ||
                      (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleForzZ {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForzZ<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
                     rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    // TagToLayout zZ for e8m0 uses BYTE_PER_C0 (32) in shape, not MX MakeMxScaleLayout's 2; match by rank only.
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleFornN {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleFornN<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
                     rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static bool const value = true;
};
#endif

}  // end namespace detail

// Advanced Layout constructions

// Make a vector layout.
template <class T>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const &len)
{
    return MakeLayout(MakeShape(len), MakeStride(Int<1>{}), MakeShape(len));
}

// Make a inner layout with Rows and Cols.
template <class Element, class LayoutTag, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const &rows, U const &cols)
{
    static_assert(std::is_same_v<LayoutTag, Catlass::layout::RowMajor> ||
                      std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
                      std::is_same_v<LayoutTag, Catlass::layout::VectorLayout> ||
                      std::is_same_v<LayoutTag, Catlass::layout::zN> ||
                      std::is_same_v<LayoutTag, Catlass::layout::nZ> ||
                      std::is_same_v<LayoutTag, Catlass::layout::zZ> || std::is_same_v<LayoutTag, Catlass::layout::L0C>,
                  "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::RowMajor or"
                  "Catlass::layout::ColumnMajor or Catlass::layout::zN or Catlass::layout::nZ or Catlass::layout::zZ "
                  "or Catlass::layout::L0C");

    constexpr uint32_t ELE_NUM_PER_C0 =
        Catlass::BytesToBits(Catlass::BYTE_PER_C0) / Catlass::SizeOfBits<Element>::value;
    constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;

    if constexpr (std::is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        return MakeLayout(MakeShape(cols), MakeStride(Int<1>{}), MakeShape(cols));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        if constexpr (std::is_same_v<Element, float4_e2m1x2_t> || std::is_same_v<Element, float4_e1m2x2_t>) {
            return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)RoundUp(cols, 2), Int<1>{}),
                              MakeShape(rows, cols));
        }
#endif
        return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)cols, Int<1>{}), MakeShape(rows, cols));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        if constexpr (std::is_same_v<Element, float4_e2m1x2_t> || std::is_same_v<Element, float4_e1m2x2_t>) {
            return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)RoundUp(rows, 2)),
                              MakeShape(rows, cols));
        }
#endif
        return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)rows), MakeShape(rows, cols));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::zN>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                      MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0)),
            MakeShape(rows, cols));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::zZ>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                      MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{},
                                  RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                       MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::L0C>) {
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
        return MakeLayout(
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                      MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                       MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) *
                                                Catlass::C0_NUM_PER_FRACTAL)),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                      MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    }
}

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <class Element, class LayoutTag, bool isMxScaleB, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeMxScaleLayout(T const &rows, U const &cols)
{
    static_assert(std::is_same_v<Element, float8_e8m0_t> && (std::is_same_v<LayoutTag, Catlass::layout::RowMajor> ||
                                                             std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
                                                             std::is_same_v<LayoutTag, Catlass::layout::zZ> ||
                                                             std::is_same_v<LayoutTag, Catlass::layout::nN>),
                  "only support RowMajor, ColumnMajor, zZ, nN in fp8_e8m0_t dtype");

    constexpr uint32_t ELE_NUM_PER_C0 = 2;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 32;

    if constexpr (std::is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(
                MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                MakeStride(RoundUp(cols, Int<ELE_NUM_PER_C0>{}), MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        } else {
            return MakeLayout(MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                              MakeStride(MakeStride(Int<1>{}, cols * ELE_NUM_PER_C0), Int<ELE_NUM_PER_C0>{}),
                              MakeShape(rows, cols));
        }
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                              MakeStride(Int<ELE_NUM_PER_C0>{}, MakeStride(Int<1>{}, rows * ELE_NUM_PER_C0)),
                              MakeShape(rows, cols));
        } else {
            return MakeLayout(
                MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                MakeStride(MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{}), RoundUp(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        }
    } else if constexpr (std::is_same_v<LayoutTag, Catlass::layout::zZ>) {
        return MakeLayout(
            MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                      MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(MakeStride(Int<ELE_NUM_PER_C0>{},
                                  RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                       MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                      MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                       MakeStride(Int<ELE_NUM_PER_C0>{},
                                  RoundUp((int64_t)rows, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL)),
            MakeShape(rows, cols));
    }
}
#endif

namespace detail {

template <class OriginBase, class TileShape, class Coord, int... Is>
CATLASS_HOST_DEVICE constexpr auto CropOriginShape(OriginBase const &originBase, TileShape const &tileShape,
                                                   Coord const &coord, seq<Is...>)
{
    return MakeShape(tla::min(static_cast<uint32_t>(get<Is>(tileShape)),
                              (static_cast<uint32_t>(get<Is>(coord)) < static_cast<uint32_t>(get<Is>(originBase)))
                                  ? (static_cast<uint32_t>(get<Is>(originBase)) - static_cast<uint32_t>(get<Is>(coord)))
                                  : 0u)...);
}

}  // namespace detail

/// 创建 tile layout：使用指定的 tile 尺寸用于内存布局计算，同时携带实际逻辑尺寸（origin_shape）。
/// coord 是元素坐标，用于计算实际的 originShape（处理边界情况）。
/// Supports layouts of any rank (rank >= 1) for depth==1 layouts.
/// For depth>1 (fractal) layouts, currently only rank-2 is supported.
template <class Layout, class TileShape, class Coord>
CATLASS_HOST_DEVICE constexpr auto GetTileLayout(Layout const &layout, TileShape const &tileShape, Coord const &coord)
{
    static_assert(is_tuple<TileShape>::value && depth_v<TileShape> == 1 && rank_v<TileShape> >= 1,
                  "GetTileLayout: TileShape must be a flat tuple with rank >= 1.");
    static_assert(is_tuple<Coord>::value && depth_v<Coord> == 1 && rank_v<Coord> == rank_v<TileShape>,
                  "GetTileLayout: Coord must have the same rank as TileShape.");

    // 统一计算 tail tile 的逻辑尺寸（originShape 裁剪）
    auto tileOriginShape = detail::CropOriginShape(layout.originShape(), tileShape, coord, tuple_seq<TileShape>{});

    // depth==1 的布局（vector/matrix/tensor）：tile shape 直接作为 memory-layout shape
    // 支持任意 rank >= 1（但必须与 layout.rank 匹配）
    if constexpr (Layout::depth == 1) {
        static_assert(Layout::rank == rank_v<TileShape>,
                      "GetTileLayout: for depth==1 layouts, TileShape rank must match layout rank.");
        return MakeLayout(tileShape, layout.stride(), tileOriginShape);
    } else {
        // depth>1 的布局（fractal layout）：目前只支持 rank=2
        // 因为 fractal layout 通常用于矩阵（rank-2），需要把 (rows, cols) 转为同结构嵌套 shape
        static_assert(rank_v<TileShape> == 2,
                      "GetTileLayout: for depth>1 (fractal) layouts, TileShape must be rank-2 (rows, cols).");

        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);

        // MakeMxScaleLayout RowMajor A 等：第一维为行长度，第二维为 (C0, ceil(cols/C0))；与 catlass_dev
        // `MakeLayoutTile` 中 rank(shape<0>)==1 && rank(shape<1>)==2 分支一致。
        if constexpr (Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 1 &&
                      rank_v<decltype(shape<1>(Layout{}))> == 2) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                              layout.stride(), tileOriginShape);
        }
        // MakeMxScaleLayout B 侧等：shape 为 ((C0, ceil(rows/C0)), cols)；与 catlass_dev `MakeLayoutTile` 中
        // rank(shape<0>)==2 && rank(shape<1>)==1 分支一致。
        else if constexpr (Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 2 &&
                           rank_v<decltype(shape<1>(Layout{}))> == 1) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<0, 0>(layout))::value;
            return MakeLayout(MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                              layout.stride(), tileOriginShape);
        }
        // 典型 fractal（zN/nZ 等）：shape<0,0>、shape<1,0> 均为编译期常量；嵌套 ((r0,...),(c0,...))。
        else if constexpr (is_static<decltype(shape<0, 0>(layout))>::value &&
                           is_static<decltype(shape<1, 0>(layout))>::value) {
            constexpr uint32_t dstInnerShapeRow = decltype(shape<0, 0>(layout))::value;
            constexpr uint32_t dstInnerShapeCol = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(MakeShape(MakeShape(Int<dstInnerShapeRow>{}, CeilDiv<dstInnerShapeRow>(rows)),
                                        MakeShape(Int<dstInnerShapeCol>{}, CeilDiv<dstInnerShapeCol>(cols))),
                              layout.stride(), tileOriginShape);
        }
        // 内层块尺寸非编译期常量：运行期从 layout 读取再分块。
        else {
            const uint32_t dstInnerShapeRow = shape<0, 0>(layout);
            const uint32_t dstInnerShapeCol = shape<1, 0>(layout);
            return MakeLayout(MakeShape(MakeShape(dstInnerShapeRow, CeilDiv(rows, dstInnerShapeRow)),
                                        MakeShape(dstInnerShapeCol, CeilDiv(cols, dstInnerShapeCol))),
                              layout.stride(), tileOriginShape);
        }
    }
}

template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutL0C(T const &rows, U const &cols)
{
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
    return MakeLayout(
        MakeShape(MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                  MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
        MakeStride(MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                   MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) *
                                            Catlass::C0_NUM_PER_FRACTAL)),
        MakeShape(rows, cols));
}

template <class Element, class T1, class T2, class T3, class T4>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutFmap(T1 const &batch, T2 const &cin1, T3 const &h, T4 const &w)
{
    constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    const int64_t strideH = w * ELE_NUM_PER_C0;
    const int64_t strideCin1 = h * strideH;
    const int64_t strideBatch = cin1 * strideCin1;
    return MakeLayout(MakeShape(static_cast<uint32_t>(batch), static_cast<uint32_t>(cin1), static_cast<uint32_t>(h),
                                static_cast<uint32_t>(w), Int<ELE_NUM_PER_C0>{}),
                      MakeStride(strideBatch, strideCin1, strideH, Int<ELE_NUM_PER_C0>{}, Int<1>{}),
                      MakeShape(static_cast<uint32_t>(batch), static_cast<uint32_t>(cin1), static_cast<uint32_t>(h),
                                static_cast<uint32_t>(w), ELE_NUM_PER_C0));
}

template <class Element, class PositionType, class T1, class T2, class T3, class T4>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutFilter(T1 const &cin1, T2 const &kh, T3 const &kw, T4 const &cout)
{
    constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    const uint32_t coutRound =
        std::is_same_v<PositionType, Catlass::Arch::PositionL1> ? RoundUp(cout, Catlass::C0_NUM_PER_FRACTAL) : cout;
    const int64_t strideKw = coutRound * ELE_NUM_PER_C0;
    const int64_t strideKh = kw * strideKw;
    const int64_t strideCin1 = kh * strideKh;
    return MakeLayout(MakeShape(static_cast<uint32_t>(cin1), static_cast<uint32_t>(kh), static_cast<uint32_t>(kw),
                                static_cast<uint32_t>(coutRound), Int<ELE_NUM_PER_C0>{}),
                      MakeStride(strideCin1, strideKh, strideKw, Int<ELE_NUM_PER_C0>{}, Int<1>{}),
                      MakeShape(static_cast<uint32_t>(cin1), static_cast<uint32_t>(kh), static_cast<uint32_t>(kw),
                                static_cast<uint32_t>(coutRound), ELE_NUM_PER_C0));
}

//
// Layout transforms
//
namespace detail {

// Prepend one leading dimension to a layout type
// This is the general form of "make batched layout":
// - Given a base layout type of rank R, create a new layout of rank R+1 by
//   prefixing (batchShape, batchStride, batchOrigin) to the existing shape/stride/originShape.
//   intentionally preserve the *types* of each stride element from the original layout
template <class Layout, class NewShapeT = uint32_t, class NewStrideT = int64_t, class NewOriginT = uint32_t,
          class Seq = tla::make_seq<Layout::rank>>
struct PrependDimLayout;

template <class Layout, class NewShapeT, class NewStrideT, class NewOriginT, int... Is>
struct PrependDimLayout<Layout, NewShapeT, NewStrideT, NewOriginT, tla::seq<Is...>> {
    using ShapeOld = tla::remove_cvref_t<decltype(std::declval<Layout const &>().shape())>;
    using StrideOld = tla::remove_cvref_t<decltype(std::declval<Layout const &>().stride())>;
    using OriginOld = tla::remove_cvref_t<decltype(std::declval<Layout const &>().originShape())>;

    using ShapeNew = tla::Shape<NewShapeT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<ShapeOld>()))>...>;
    using StrideNew =
        tla::Stride<NewStrideT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<StrideOld>()))>...>;
    using OriginNew = tla::Shape<NewOriginT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<OriginOld>()))>...>;

    using type = tla::Layout<ShapeNew, StrideNew, OriginNew>;
};

}  // namespace detail

template <class Layout, class NewShapeT = uint32_t, class NewStrideT = int64_t, class NewOriginT = uint32_t>
using PrependDimLayout_t = typename detail::PrependDimLayout<Layout, NewShapeT, NewStrideT, NewOriginT>::type;

template <class Layout>
using MakeBatchedLayout_t = PrependDimLayout_t<Layout>;

}  // end namespace tla

#endif  // TLA_LAYOUT_HPP
