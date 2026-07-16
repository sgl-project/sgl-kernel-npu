/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TENSOR_HPP
#define TLA_TENSOR_HPP

#include "catlass_a5/arch/arch.hpp"
#include "tla_a5/layout.hpp"                     // tla::Shape
#include "tla_a5/numeric/integral_constant.hpp"  // tla::is_integral
#include "tla_a5/int_tuple.hpp"

namespace tla {

//
// Underscore slicing utilities
//
namespace detail {

// Safe "element type at index I" helper that avoids instantiating get<I> for invalid indices.
template <class Coord, int I, class Enable = void>
struct coord_elem_type {
    using type = void;
};
template <class Coord, int I>
struct coord_elem_type<Coord, I,
    std::enable_if_t<(I >= 0) && (I < (int)tla::tuple_size<tla::remove_cvref_t<Coord>>::value)>> {
    using type = tla::remove_cvref_t<decltype(tla::get<I>(std::declval<Coord>()))>;
};

template <class Coord, int I>
struct coord_elem_is_underscore : tla::is_underscore<typename coord_elem_type<Coord, I>::type> {};

// Count underscores.
template <class Coord, int I, class Enable = void>
struct underscore_count_from : tla::integral_constant<int, 0> {};
template <class Coord, int I>
struct underscore_count_from<Coord, I, std::enable_if_t<(I >= 0)>> {
    static constexpr int value =
        (coord_elem_is_underscore<Coord, I>::value ? 1 : 0) + underscore_count_from<Coord, I - 1>::value;
};
template <class Coord>
struct underscore_count
    : tla::integral_constant<int,
        underscore_count_from<Coord, (int)tla::tuple_size<tla::remove_cvref_t<Coord>>::value - 1>::value> {};

// Build index sequences for underscore dims (stable 0..R-1 recursion).
template <class Coord, int I, int R, int... Is>
struct underscore_indices_impl;
template <class Coord, int R, int... Is>
struct underscore_indices_impl<Coord, R, R, Is...> {
    using type = seq<Is...>;
};
template <class Coord, int I, int R, int... Is>
struct underscore_indices_impl
    : std::conditional_t<coord_elem_is_underscore<Coord, I>::value,
                         underscore_indices_impl<Coord, I + 1, R, Is..., I>,
                         underscore_indices_impl<Coord, I + 1, R, Is...>> {};

template <class Coord>
using underscore_indices =
    typename underscore_indices_impl<Coord, 0, (int)tla::tuple_size<tla::remove_cvref_t<Coord>>::value>::type;

// Replace every tla::_ with 0 for offset computation.
template <class T>
CATLASS_HOST_DEVICE constexpr decltype(auto) underscore_to_zero(T const& x) {
    if constexpr (tla::is_underscore<T>::value) {
        return tla::_0{};
    } else {
        return x;
    }
}

template <class Coord, int... I>
CATLASS_HOST_DEVICE constexpr auto replace_underscore_with_zero_impl(Coord const& c, seq<I...>) {
    return tla::MakeCoord(underscore_to_zero(tla::get<I>(c))...);
}

template <class Coord>
CATLASS_HOST_DEVICE constexpr auto replace_underscore_with_zero(Coord const& c) {
    static_assert(tla::is_tuple<tla::remove_cvref_t<Coord>>::value,
                  "Coord must be tla::tuple for underscore slicing.");
    return replace_underscore_with_zero_impl(c, tuple_seq<Coord>{});
}

// Build a layout from selected top-level indices of `layout`.
template <class Layout, int... Is>
CATLASS_HOST_DEVICE constexpr auto select_layout(Layout const& layout, seq<Is...>) {
    auto shape_new  = tla::MakeTuple(tla::get<Is>(layout.shape())...);
    auto stride_new = tla::MakeTuple(tla::get<Is>(layout.stride())...);
    auto origin_new = tla::MakeTuple(tla::get<Is>(layout.originShape())...);
    return tla::MakeLayout(shape_new, stride_new, origin_new);
}

} // namespace detail

//
// slice_and_offset
//
// A lightweight helper that factors underscore slicing into:
// - a projected layout (keeping only underscored dimensions, in-order), and
// - a base offset computed at the fixed indices (underscored dims treated as 0).
//
// Notes:
// - `coord_arg` must be a one-level `tla::tuple` and contain at least one `tla::_`.
// - This function does not perform runtime bounds checks against originShape(); out-of-bounds is undefined behavior.
// - Returned `offset` is an element offset intended for `BuiltinTensor::operator[](offset)` view creation.
template <class CoordArg, class Layout, class BaseCoord>
CATLASS_HOST_DEVICE constexpr auto slice_and_offset(CoordArg const& coord_arg,
                                                   Layout const& layout,
                                                   BaseCoord const& base_coord)
{
    static_assert(tla::is_tuple<tla::remove_cvref_t<CoordArg>>::value, "slice_and_offset expects a tuple CoordArg.");
    static_assert(depth_v<CoordArg> == 1, "slice_and_offset only supports one-level CoordArg (no nested tuples).");
    static_assert((int)tla::tuple_size<tla::remove_cvref_t<CoordArg>>::value == (int)Layout::rank,
                  "slice_and_offset requires CoordArg rank == Layout::rank.");
    static_assert(tla::is_tuple<tla::remove_cvref_t<BaseCoord>>::value, "slice_and_offset expects a tuple BaseCoord.");
    static_assert((int)tla::tuple_size<tla::remove_cvref_t<BaseCoord>>::value == (int)Layout::rank,
                  "slice_and_offset requires BaseCoord rank == Layout::rank.");

    constexpr int k = detail::underscore_count<CoordArg>::value;
    static_assert(k > 0, "slice_and_offset requires at least one underscore.");
    static_assert(k <= Layout::rank, "Invalid underscore count.");

    // Compute base offset using zeros for underscores
    auto coord0 = detail::replace_underscore_with_zero(coord_arg);
    auto full0 = Add(base_coord, coord0);
    auto offset = (int64_t)layout(full0);

    // Determine output dims (underscored dims) and build projected layout.
    using Us = detail::underscore_indices<CoordArg>;
    auto layout_proj = detail::select_layout(layout, Us{});

    return tla::MakeTuple(layout_proj, offset);
}

// Convenience overload (no base_coord): assume base_coord == 0.
template <class CoordArg, class Layout>
CATLASS_HOST_DEVICE constexpr auto slice_and_offset(CoordArg const& coord_arg, Layout const& layout)
{
    using Z = detail::MakeZeroTuple<Layout::rank>;
    return slice_and_offset(coord_arg, layout, Z{});
}


//
// Tensor
//

namespace detail {

template <class A, class B, int... Is>
CATLASS_DEVICE constexpr
auto HadamardU32(A const& a, B const& b, seq<Is...>)
{
    return MakeCoord((static_cast<uint32_t>(get<Is>(a)) * static_cast<uint32_t>(get<Is>(b)))...);
}

template <class TensorT, class CoordT, class ShapeT, int R>
CATLASS_DEVICE constexpr
auto GetTileImpl(TensorT const& tensor, CoordT const& coord, ShapeT const& shape, Int<R>)
{
    static_assert(is_tuple<CoordT>::value && depth_v<CoordT> == 1 && rank_v<CoordT> == R, "Coord rank mismatch.");
    static_assert(is_tuple<ShapeT>::value && depth_v<ShapeT> == 1 && rank_v<ShapeT> == R, "Shape rank mismatch.");

    auto layoutNew = GetTileLayout(tensor.layout(), shape, coord);
    auto coordNew = Add(tensor.coord(), coord);
    return MakeTensor(tensor.data(), layoutNew, coordNew, Catlass::Arch::PositionType<TensorT::position>{});
}

template <class TensorT, class TileCoord, class TileShape, int R>
CATLASS_DEVICE constexpr
auto TileViewImpl(TensorT const& tensor, TileCoord const& tileCoord, TileShape const& tileShape, Int<R>)
{
    static_assert(is_tuple<TileCoord>::value && depth_v<TileCoord> == 1 && rank_v<TileCoord> == R, "TileCoord rank mismatch.");
    static_assert(is_tuple<TileShape>::value && depth_v<TileShape> == 1 && rank_v<TileShape> == R, "TileShape rank mismatch.");

    auto elementOffset = HadamardU32(tileCoord, tileShape, tuple_seq<TileCoord>{});
    auto layoutNew = GetTileLayout(tensor.layout(), tileShape, elementOffset);
    auto coordNew = Add(tensor.coord(), elementOffset);
    return MakeTensor(tensor.data(), layoutNew, coordNew, Catlass::Arch::PositionType<TensorT::position>{});
}

} // namespace detail

template <class BuiltinTensor, class Layout_, class Coord_, AscendC::TPosition Position>
struct Tensor {
    using Element = typename BuiltinTensor::PrimType;
    using Layout = Layout_;
    using Coord = Coord_;
    static constexpr AscendC::TPosition position = Position;

    CATLASS_HOST_DEVICE constexpr
    Tensor() {}

    CATLASS_HOST_DEVICE constexpr
    Tensor(BuiltinTensor const& builtinTensor, Layout const& layout, Coord const& coord = {})
        : rep_(builtinTensor, layout, coord) {}

    //
    // Accessors
    //

    static constexpr int rank  = Layout::rank;

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) tensor() const
    {
        return *this;
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) data() const
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) data()
    {
        return get<0>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) layout() const
    {
        return get<1>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) coord() const
    {
        return get<2>(rep_);
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) shape() const
    {
        return layout().shape();
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) stride() const
    {
        return layout().stride();
    }

    CATLASS_HOST_DEVICE constexpr
    decltype(auto) originShape() const
    {
        return layout().originShape();
    }

    //
    // Indexing / slicing
    //
    // - No underscore: returns `data()[layout()(coord()+coord_arg)]`
    // - Underscores (0..rank, one-level coord): returns a subtensor view over the underscored dimensions (kept in-order)
    //   Notes:
    //   - Coord must be one-level (no nested tuples in coord elements).
    //   - Fixed (non-underscore) indices are expected to be within originShape(). This implementation does not
    //     generally perform runtime bounds checks/cropping; out-of-bounds indices result in undefined behavior.
    template <class CoordArg>
    CATLASS_HOST_DEVICE constexpr
    decltype(auto) operator()(CoordArg const& coord_arg) const
    {
        if constexpr (tla::is_tuple<tla::remove_cvref_t<CoordArg>>::value) {
            static_assert(depth_v<CoordArg> == 1, "Underscore slicing only supports one-level Coord (no nested tuples).");
            static_assert(tla::tuple_size<tla::remove_cvref_t<CoordArg>>::value == Layout::rank,
                          "Tensor::operator()(coord): Coord rank must equal tensor rank (Layout::rank).");

            constexpr int k = detail::underscore_count<CoordArg>::value;
            if constexpr (k > 0) {
                static_assert(k <= Layout::rank, "Invalid underscore count.");

                auto sliced = tla::slice_and_offset(coord_arg, layout(), coord());
                auto layout_proj = tla::get<0>(sliced);
                auto offset = (int64_t)tla::get<1>(sliced);

                using CoordZ = detail::MakeZeroTuple<(size_t)k>;
                auto data_new = data()[static_cast<uint64_t>(offset)];
                return Tensor<decltype(data_new), decltype(layout_proj), CoordZ, position>(data_new, layout_proj, CoordZ{});
            } else {
                // No underscore: point view at coord() + coord_arg
                auto full = Add(coord(), coord_arg);
                return data()[layout()(full)];
            }
        } else {
            // Scalar coordinate convenience (rank-1): treat it as a 1D coord tuple.
            static_assert(Layout::rank == 1, "Tensor::operator()(scalar) is only supported for rank-1 tensors.");
            auto full = Add(coord(), MakeCoord(coord_arg));
            return data()[layout()(full)];
        }
    }

    template <class Coord0, class Coord1, class... Coords>
    CATLASS_HOST_DEVICE constexpr
    decltype(auto) operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const
    {
        return operator()(MakeCoord(c0, c1, cs...));
    }

    tla::tuple<BuiltinTensor, Layout, Coord> rep_;
};

template <class BuiltinTensor, class Layout, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(BuiltinTensor const& builtinTensor, Layout const& layout, PositionType)
{
    using Coord = detail::MakeZeroTuple<Layout::rank>;
    return Tensor<BuiltinTensor, Layout, Coord, PositionType::value>(builtinTensor, layout);
}

template <class BuiltinTensor, class Layout, class Coord, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensor(BuiltinTensor const& builtinTensor, Layout const& layout, Coord const& coord, PositionType)
{
    return Tensor<BuiltinTensor, Layout, Coord, PositionType::value>(builtinTensor, layout, coord);
}

// Get a tile from tensor, automatically handling boundary cases (tail tile).
// coord is element coordinate; shape is tile size for memory layout.
// The returned Tensor's layout.shape() is used for memory layout calculation,
// and layout.originShape() is the actual logical size (may be smaller than shape).
// Supports tensors of any rank (rank >= 1).
template <class Tensor, class Coord, class Shape>
CATLASS_DEVICE constexpr
auto GetTile(Tensor const& tensor, Coord const& coord, Shape const& shape)
{
    static_assert(Tensor::rank >= 1, "GetTile requires tensor rank >= 1.");
    static_assert(Tensor::rank == rank_v<Coord> && Tensor::rank == rank_v<Shape>, 
                  "GetTile: coord and shape must have the same rank as the tensor.");
    return detail::GetTileImpl(tensor, coord, shape, Int<Tensor::rank>{});
}


// 从 tensor 中获取一个 tile，自动处理边界情况（tail tile）。
// tileCoord 是 tile 单位坐标；tileShape 是用于内存布局的 tile 尺寸。
// 返回的 Tensor 的 layout.shape() 用于内存布局计算，layout.originShape() 是实际逻辑尺寸（可能小于 tileShape）。
// TileView(tensor, tileCoord, tileShape) = GetTile(tensor, tileCoord ⊙ tileShape, tileShape)
// Supports tensors of any rank (rank >= 1).
template <class TensorT, class TileCoord, class TileShape>
CATLASS_DEVICE constexpr
auto TileView(TensorT const& tensor, TileCoord const& tileCoord, TileShape const& tileShape)
{
    static_assert(TensorT::rank >= 1, "TileView requires tensor rank >= 1.");
    static_assert(TensorT::rank == rank_v<TileCoord> && TensorT::rank == rank_v<TileShape>, 
                  "TileView: tileCoord and tileShape must have the same rank as the tensor.");
    return detail::TileViewImpl(tensor, tileCoord, tileShape, Int<TensorT::rank>{});
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 目标 layout 根据 LayoutTagDst 构造，从 LikeTensor::Element 推断 ElementDst，从 likeTensor 的 originShape 提取尺寸。
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                      LikeTensor const& likeTensor,
                      PositionType)
{
    using ElementDst = typename LikeTensor::Element;
    static_assert(std::is_same_v<typename BuiltinTensor::PrimType, ElementDst>,
                  "BuiltinTensor element type must match LikeTensor element type");
    return MakeTensorLike<LayoutTagDst, ElementDst>(builtinTensor, likeTensor, PositionType{});
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 使用 layoutBase 的 shape/stride，但继承 likeTensor 的 originShape，从 LikeTensor::Element 推断 ElementDst。
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
CATLASS_HOST_DEVICE constexpr
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                      LikeTensor const& likeTensor,
                      PositionType,
                      LayoutBase const& layoutBase)
{
    using ElementDst = typename LikeTensor::Element;
    static_assert(std::is_same_v<typename BuiltinTensor::PrimType, ElementDst>,
                  "BuiltinTensor element type must match LikeTensor element type");
    return MakeTensorLike<LayoutTagDst, ElementDst>(builtinTensor, likeTensor, PositionType{}, layoutBase);
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 目标 layout 根据 LayoutTagDst 构造，从 LikeTensor::Element 推断 ElementDst，从 likeTensor 的 originShape 提取尺寸。(调用MakeLayout，可能会因分型布局合法要求对shape进行以分型为粒度的向上取整)
// 允许 BuiltinTensor 的元素类型与 LikeTensor 的元素类型不同（例如 L0C 使用 ElementAccumulator 而不是 ElementC）。
template <class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType>
CATLASS_HOST_DEVICE constexpr
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                      LikeTensor const& likeTensor,
                      PositionType)
{
    static_assert(LikeTensor::rank == 1 || LikeTensor::rank == 2,
                  "MakeTensorLike<LayoutTag, Element>(..., likeTensor) expects rank-1 or rank-2 likeTensor.");
    static_assert(std::is_same_v<typename BuiltinTensor::PrimType, ElementDst>,
                  "BuiltinTensor element type must match specified ElementDst type");
    // 根据目标布局格式（LayoutTagDst）和指定的元素类型构造 layout
    if constexpr (LikeTensor::rank == 1) {
        auto layoutNominal = MakeLayout<ElementDst, LayoutTagDst>(get<0>(likeTensor.layout().originShape()));
        using Coord0 = detail::MakeZeroTuple<decltype(layoutNominal)::rank>;
        return Tensor<BuiltinTensor, decltype(layoutNominal), Coord0, PositionType::value>(builtinTensor, layoutNominal);
    } else {
        static_assert(LikeTensor::rank == 2, "MakeTensorLike<LayoutTag, Element>(..., likeTensor) expects rank-1 or rank-2 likeTensor.");
        auto layoutNominal = MakeLayout<ElementDst, LayoutTagDst>(get<0>(likeTensor.layout().originShape()), get<1>(likeTensor.layout().originShape()));
        using Coord0 = detail::MakeZeroTuple<decltype(layoutNominal)::rank>;
        return Tensor<BuiltinTensor, decltype(layoutNominal), Coord0, PositionType::value>(builtinTensor, layoutNominal);
    }
}

// 创建一个与另一个 Tensor 类似的 Tensor：
// 使用 layoutBase 的 shape/stride，但继承 likeTensor 的 originShape。允许 BuiltinTensor 的元素类型与 LikeTensor 的元素类型不同。
template <class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
CATLASS_HOST_DEVICE constexpr
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                      LikeTensor const& likeTensor,
                      PositionType,
                      LayoutBase const& layoutBase)
{
    static_assert(LikeTensor::rank == 1 || LikeTensor::rank == 2, "MakeTensorLike<LayoutTag, Element>(..., likeTensor, layoutBase) expects rank-1 or rank-2 likeTensor.");
    static_assert(std::is_same_v<typename BuiltinTensor::PrimType, ElementDst>,
                  "BuiltinTensor element type must match specified ElementDst type");

    auto layoutFixedStride = MakeLayout(layoutBase.shape(), layoutBase.stride(), likeTensor.originShape());
    using Coord0 = detail::MakeZeroTuple<decltype(layoutFixedStride)::rank>;
    return Tensor<BuiltinTensor, decltype(layoutFixedStride), Coord0, PositionType::value>(builtinTensor, layoutFixedStride);
}

} // end namespace tla

#endif // TLA_TENSOR_HPP
