/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_COPY_HPP
#define CATLASS_GEMM_TILE_TILE_COPY_HPP

#include <type_traits>

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/detail/tag_to_layout.hpp"
#include "catlass_a5/gemm/helper.hpp"
#include "catlass_a5/gemm/tile/cast_fp8_to_fp16.hpp"
#include "catlass_a5/gemm/tile/cast_int4_to_int8.hpp"
#include "catlass_a5/gemm/tile/cast_int8_to_fp16.hpp"
#include "catlass_a5/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass_a5/gemm/tile/copy_gm_to_ub.hpp"
#include "catlass_a5/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass_a5/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass_a5/gemm/tile/copy_l1_to_bt.hpp"
#include "catlass_a5/gemm/tile/copy_l1_to_fp.hpp"
#include "catlass_a5/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass_a5/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass_a5/gemm/tile/copy_ub_to_gm.hpp"
#include "catlass_a5/gemm/tile/tile_copy_tla.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
template <class ArchTag, class AType, class BType, class CType, class PrologueA_, class PrologueB_, class BiasType = void>
struct TileCopyWithPrologueDeqPerTensor {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType, Tile::ScaleGranularity::PER_TENSOR>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

template <class ArchTag, class AType, class BType, class CType, class PrologueA_, class PrologueB_, class BiasType = void>
struct TileCopyWithPrologue {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

///////////////////////////////////
/// new add
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmTpe type for Bias operand
    class BiasType = void>
struct TileCopyGemm {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    // change structual
    using L1AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1AType;
    using L1BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1BType;
    using L0AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0AType;
    using L0BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0BType;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType, L1BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1AType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1BType, L0BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};
//////////////////////////////
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType>
struct ConvTileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<
        std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

// fixpipe开启relu开关
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct ReluTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType> {
    // 重写 CopyL0CToGm
    using ElementAccumulator = typename TileCopy<ArchTag, AType, BType, CType, BiasType>::ElementAccumulator;
    using CopyL0CToGm = Gemm::Tile::
        CopyL0CToGm<ArchTag, ElementAccumulator, CType, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, true>;
};

// fixpipe开启随路量化
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void,
    /// GemmType type for Bias operand
    ScaleGranularity SCALE_GRANU = ScaleGranularity::PER_TENSOR>
struct QuantTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType> {
    // 重写 CopyL0CToGm
    using ElementAccumulator = typename TileCopy<ArchTag, AType, BType, CType, BiasType>::ElementAccumulator;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>;

    using CopyGmToL1Scale = Gemm::Tile::CopyGmToL1<
        ArchTag,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::GM>,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>>;

    using CopyL1ToFP = Gemm::Tile::CopyL1ToFP<
        ArchTag,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>>;
};

// sparse Tile
template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementB_,
    class LayoutTagB,
    class ElementC_,
    class LayoutTagC>
struct SparseTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = layout::zZ;
    using LayoutTagL0B = layout::nZ;
    using LayoutTagL1BIdx = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;

    using LayoutA = detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL1BIdx = detail::TagToLayout_t<int32_t, LayoutTagL1BIdx>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1BIdx =
        tla::Tensor<AscendC::LocalTensor<int32_t>, LayoutL1BIdx, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::
        Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Gemm::Tile::TileCopySparseTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Gemm::Tile::TileCopySparseTla<ArchTag, TensorB, TensorL1B>;

    template <class TensorIdx>
    using CopyGmToL1BIdx = Gemm::Tile::TileCopySparseTla<ArchTag, TensorIdx, TensorL1BIdx>;

    using CopyL1ToL0A = Gemm::Tile::TileCopySparseTla<ArchTag, TensorL1A, TensorL0A>;

    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0BSparseTla<ArchTag, ElementA, TensorL1B, TensorL0B, TensorL1BIdx>;

    template <class TensorC>
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmSparseTla<ArchTag, TensorL0C, TensorC>;
};

#endif

template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA_,
    class ElementB_,
    class LayoutTagB_,
    class ElementC_,
    class LayoutTagC_,
    class ElementBias = void,
    bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY_ = ScaleGranularity::NO_QUANT,
    class L0CCopyMode = CopyToGM>
struct PackedTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    static constexpr bool ReluEnable = ReluEnable_;
    static constexpr ScaleGranularity DEQUANT_GRANULARITY = DEQUANT_GRANULARITY_;

    static constexpr bool HAS_BIAS = !std::is_void_v<ElementBias>;
    static constexpr bool HAS_QUANT_TENSOR = (DEQUANT_GRANULARITY == ScaleGranularity::PER_CHANNEL);

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = typename helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = layout::nZ;
    using LayoutTagL0C = layout::L0C;

    using LayoutA = detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1AVectorLayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0>, AscendC::TPosition::A1>;
    using TensorL1ALayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;

    using TensorL1A =
        std::conditional_t<tla::detail::isVector<LayoutTagA>::value, TensorL1AVectorLayout, TensorL1ALayout>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::
        Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;
    using TensorL1Bias = std::conditional_t<
        HAS_BIAS,
        tla::Tensor<
            AscendC::LocalTensor<ElementBias>,
            detail::TagToLayout_t<ElementBias, layout::VectorLayout>,
            tla::Coord<tla::_0>,
            AscendC::TPosition::A1>,
        EmptyClass>;
    using TensorL0Bias = tla::Tensor<
        AscendC::LocalTensor<ElementAccumulator>,
        detail::TagToLayout_t<ElementAccumulator, layout::VectorLayout>,
        tla::Coord<tla::_0>,
        AscendC::TPosition::C2>;
    using TensorL1Quant = std::conditional_t<
        HAS_QUANT_TENSOR,
        tla::Tensor<AscendC::LocalTensor<uint64_t>,
            detail::TagToLayout_t<uint64_t, layout::VectorLayout>,
            tla::Coord<tla::_0>,
            AscendC::TPosition::A1>,
        EmptyClass>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Gemm::Tile::TileCopyTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorL1B>;

    template <class TensorBias>
    using CopyGmToL1Bias =
        std::conditional_t<HAS_BIAS, Gemm::Tile::TileCopyTla<ArchTag, TensorBias, TensorL1Bias>, EmptyClass>;

    template <class TensorQuant>
    using CopyGmToL1Scale =
        std::conditional_t<HAS_QUANT_TENSOR, Gemm::Tile::TileCopyTla<ArchTag, TensorQuant, TensorL1Quant>, EmptyClass>;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL1ToBT =
        std::conditional_t<HAS_BIAS, Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>, EmptyClass>;

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    template <class TensorC>
    using CopyL0CToDst = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
    template <class TensorC>
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
#endif
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class TensorA,
    class LayoutTagA,
    class TensorB,
    class LayoutTagB,
    class TensorC,
    class LayoutTagC,
    class TensorBias = void,
    class LayoutTagBias = void,
    bool IS_PADDING_A = false,
    bool IS_PADDING_B = false>
struct PaddingPackedTileCopyTla {
    static_assert(
        std::is_same_v<LayoutTagA, layout::RowMajor> || std::is_same_v<LayoutTagA, layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor"
    );
    static_assert(
        std::is_same_v<LayoutTagB, layout::RowMajor> || std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor"
    );
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = layout::zZ;
    using LayoutTagL0B = layout::nZ;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::
        Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using LayoutPaddingTagA = std::
        conditional_t<std::is_same_v<LayoutTagA, layout::RowMajor>, layout::PaddingRowMajor, layout::PaddingColumnMajor>;
    using LayoutPaddingTagB = std::
        conditional_t<std::is_same_v<LayoutTagB, layout::RowMajor>, layout::PaddingRowMajor, layout::PaddingColumnMajor>;

    using CopyGmToL1A = std::conditional_t<
        IS_PADDING_A,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutPaddingTagA, LayoutTagL1A>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutTagA, LayoutTagL1A>>;
    using CopyGmToL1B = std::conditional_t<
        IS_PADDING_B,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutPaddingTagB, LayoutTagL1B>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutTagB, LayoutTagL1B>>;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    using CopyL0CToDst = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>;
#endif
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementB_,
    class LayoutTagB,
    class ElementC_,
    class LayoutTagC,
    class ElementBias = void,
    CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,
    bool ReluEnable = false,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct PackedTileCopyTlaToUB
    : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB, ElementC_, LayoutTagC, ElementBias> {
    static constexpr CopyL0CToUBMode CopyMode = CopyMode_;
    // 重写 CopyL0CToDst
    using TensorL0C = typename PackedTileCopyTla<
        ArchTag,
        ElementA_,
        LayoutTagA,
        ElementB_,
        LayoutTagB,
        ElementC_,
        LayoutTagC,
        ElementBias>::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode, DEQUANT_GRANULARITY, ReluEnable>;
};

template <
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementB_,
    class LayoutTagB,
    class ElementMxScaleA_,
    class LayoutMxScaleA_,
    class ElementMxScaleB_,
    class LayoutMxScaleB_,
    class ElementC_,
    class LayoutTagC,
    class ElementBias = void,
    bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    class L0CCopyMode = CopyToGM>
struct PackedMxTileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
    ElementC_, LayoutTagC, ElementBias, ReluEnable_, DEQUANT_GRANULARITY, L0CCopyMode> {
    using ElementMxScaleA = ElementMxScaleA_;
    using ElementMxScaleB = ElementMxScaleB_;

    using LayoutMxScaleA = LayoutMxScaleA_;
    using LayoutMxScaleB = LayoutMxScaleB_;

    using LayoutTagL1MxScaleA = layout::zZ;
    using LayoutTagL1MxScaleB = layout::nN;
    using LayoutL1MxScaleA = detail::TagToLayout_t<ElementMxScaleA, LayoutTagL1MxScaleA>;
    using LayoutL1MxScaleB = detail::TagToLayout_t<ElementMxScaleB, LayoutTagL1MxScaleB>;

    using TensorL1MxScaleA = tla::Tensor<AscendC::LocalTensor<ElementMxScaleA>, LayoutL1MxScaleA,
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1MxScaleB = tla::Tensor<AscendC::LocalTensor<ElementMxScaleB>, LayoutL1MxScaleB,
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;

    template <class TensorMxScaleA>
    using CopyGmToL1MxScaleA = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;

    template <class TensorMxScaleB>
    using CopyGmToL1MxScaleB = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;
};

template <
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementB_,
    class LayoutTagB,
    class ElementMxScaleA_,
    class LayoutMxScaleA_,
    class ElementMxScaleB_,
    class LayoutMxScaleB_,
    class ElementC_,
    class LayoutTagC,
    class ElementBias = void,
    CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct PackedMxTileCopyTlaToUB
    : public PackedMxTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
        ElementMxScaleA_, LayoutMxScaleA_, ElementMxScaleB_, LayoutMxScaleB_,
        ElementC_, LayoutTagC, ElementBias> {
    static constexpr CopyL0CToUBMode CopyMode = CopyMode_;

    using TensorL0C = typename PackedMxTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
        ElementMxScaleA_, LayoutMxScaleA_, ElementMxScaleB_, LayoutMxScaleB_,
        ElementC_, LayoutTagC, ElementBias>::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode, DEQUANT_GRANULARITY, false>;
};

template <
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementPrologueB_,
    class LayoutTagPrologueB,
    class ElementB_,
    class LayoutTagB,
    class ElementMxScaleA_,
    class LayoutMxScaleA_,
    class ElementMxScaleB_,
    class LayoutMxScaleB_,
    class ElementC_,
    class LayoutTagC,
    class ElementBias = void,
    bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    class L0CCopyMode = CopyToGM
>
struct PackedMxA8W4TileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagPrologueB,
    ElementC_, LayoutTagC, ElementBias, ReluEnable_, DEQUANT_GRANULARITY, L0CCopyMode>{
    using ElementMxScaleA = ElementMxScaleA_;
    using ElementMxScaleB = ElementMxScaleB_;

    using LayoutMxScaleA = LayoutMxScaleA_;
    using LayoutMxScaleB = LayoutMxScaleB_;

    using LayoutTagL1MxScaleA = layout::zZ;
    using LayoutTagL1MxScaleB = layout::nN;
    using LayoutL1MxScaleA = detail::TagToLayout_t<ElementMxScaleA, LayoutTagL1MxScaleA>;
    using LayoutL1MxScaleB = detail::TagToLayout_t<ElementMxScaleB, LayoutTagL1MxScaleB>;

    using LayoutB = detail::TagToLayout_t<ElementPrologueB_, LayoutTagPrologueB>;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB_, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0B = layout::nZ;

    using LayoutL1B = detail::TagToLayout_t<ElementB_, LayoutTagL1B>;
    using LayoutL0B = detail::TagToLayout_t<ElementB_, LayoutTagL0B>;

    using TensorL1B =
    	tla::Tensor<AscendC::LocalTensor<ElementB_>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0B =
    	tla::Tensor<AscendC::LocalTensor<ElementB_>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;

    using TensorL1MxScaleA = tla::Tensor<AscendC::LocalTensor<ElementMxScaleA>, LayoutL1MxScaleA,
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1MxScaleB = tla::Tensor<AscendC::LocalTensor<ElementMxScaleB>, LayoutL1MxScaleB,
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;

    template <class TensorMxScaleA>
    using CopyGmToL1MxScaleA = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;

    template <class TensorMxScaleB>
    using CopyGmToL1MxScaleB = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;

    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
};

#endif

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_COPY_HPP
