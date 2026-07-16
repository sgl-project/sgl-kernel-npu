/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_GM_HPP
#define CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_GM_HPP

#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/catlass.hpp"
#include "catlass_a5/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "tla_a5/tensor.hpp"

namespace Catlass::Gemm::Tile {

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor"
        );

        AscendC::DataCopyCO12DstParams intriParams;

        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = true;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void
    operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch, uint32_t dstNdStride)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor"
        );

        const uint32_t L0CM = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t L0CN = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());

        // AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // // Fixpipe layout information
        // intriParams.nSize = tla::get<1>(dstTensor.shape());
        // intriParams.mSize = tla::get<0>(dstTensor.shape());
        // intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        // intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // // Fixpipe auxiliary arguments
        // intriParams.quantPre = quantPre;
        // intriParams.reluEn = reluEn;
        // intriParams.unitFlag = 0;

        // intriParams.params.ndNum = l0Batch;
        // intriParams.params.srcNdStride = L0CM * L0CN / tla::get<1, 0>(srcTensor.shape());
        // intriParams.params.dstNdStride = dstNdStride;

        // // Call AscendC Fixpipe
        // AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
        //     dstTensor.data(), srcTensor.data(), intriParams);

        AscendC::DataCopyCO12DstParams intriParams;

        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = true;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = 0;

        AscendC::SetFixpipeNz2ndFlag(l0Batch, L0CM * L0CN / tla::get<1, 0>(srcTensor.shape()), dstNdStride);
        AscendC::DataCopy(dstTensor.data(), srcTensor.data(), intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
            && TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and zN"
        );

        AscendC::DataCopyCO12DstParams intriParams;

        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / (BYTE_PER_C0 / sizeof(ElementDst));
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = false;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.channelSplit = true;
        }

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_TENSOR,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(float scalar)
        {
            scale = scalar;
        }
    };
    Params params;

    CATLASS_DEVICE
    CopyL0CToGmTla() = default;

    CATLASS_DEVICE
    CopyL0CToGmTla(Params const &params_) : params(params_) {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE 
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor"
        );
        
        AscendC::FixpipeParamsC310 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToGmTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::GlobalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::GM>,
    ScaleGranularity::PER_CHANNEL,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc, class TensorQuant>
    CATLASS_DEVICE 
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorQuant const &quantTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::GM,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be GM and RowMajor"
        );

        AscendC::FixpipeParamsC310 intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        auto quantOffset = quantTensor.layout()(quantTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], quantTensor.data()[quantOffset], intriParams);
    }
};

////////////////////////////////////CopyL0CToGm(No-TLA, Ascend950)////////////////////////////////////////////////
template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::Ascend950,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::DataCopyCO12DstParams intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = true;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = unitFlag;

        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(dst, src, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::Ascend950,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::PER_TENSOR,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::PER_TENSOR>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(float scalar)
        {
            scale = scalar;
        }
    };
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsC310 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::Ascend950,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::PER_CHANNEL,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::PER_CHANNEL>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src, AscendC::LocalTensor<uint64_t> const &scale,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsC310 intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, scale, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::Ascend950,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::zN>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::zN;
    static constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::DataCopyCO12DstParams intriParams;

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
        intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst));
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = false;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.channelSplit = true;
        }

        AscendC::DataCopy(dst, src, intriParams);
    }
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::Ascend950,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::NDC1HWC0>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::NDC1HWC0;
    static constexpr auto quantPre = CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {};
    Params params;

    CATLASS_DEVICE
    CopyL0CToGm() = default;

    CATLASS_DEVICE
    CopyL0CToGm(Params const &params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;

        intriParams.nSize = srcLayout.orgShape(1);
        intriParams.mSize = srcLayout.orgShape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
        intriParams.dstStride = dstLayout.shape(1) * dstLayout.shape(2);

        if constexpr (AscendC::IsSameType<ElementSrc, float>::value &&
                      AscendC::IsSameType<ElementDst, float>::value) {
            intriParams.isChannelSplit = true;
        }

        intriParams.quantPre = quantPre;
        intriParams.reluEn = false;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_GM_HPP
