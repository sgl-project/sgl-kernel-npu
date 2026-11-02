#ifndef ACT_GEMM_TILE_COPY_GM_TO_UB_HPP
#define ACT_GEMM_TILE_COPY_GM_TO_UB_HPP

#include "../../../act/act.hpp"
#include "../../../tla/tensor.hpp"

namespace Act::Gemm::Tile {

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<
    Arch::AtlasA2, Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value && tla::detail::isRowMajor<LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::VECCALC>;
    using TensorSrc = Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    // Methods

    ACT_DEVICE
    TileCopyTla() {};

    ACT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            get<0>(srcTensor.shape()), get<1>(srcTensor.shape()) * sizeof(ElementSrc),
            (get<0>(srcTensor.stride()) - get<1>(srcTensor.shape())) * sizeof(ElementSrc),
            (get<0>(dstTensor.stride()) - get<1>(dstTensor.shape())) / ELE_NUM_PER_BLK, 0);
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams, padParams);
    };
};

}  // namespace Act::Gemm::Tile

#endif  // ACT_GEMM_TILE_COPY_GM_TO_UB_HPP
