#ifndef ACT_EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_ROW_HPP
#define ACT_EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_ROW_HPP

#include "../../../act/act.hpp"

namespace Act::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    class TileShape_>
struct TileBroadcastInplaceByRow {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    ACT_DEVICE
    TileBroadcastInplaceByRow() {}

    ACT_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const &ubInOut)
    {
        constexpr uint32_t eleNumPerVectorFractal = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);

        constexpr uint64_t mask = eleNumPerVectorFractal;
        constexpr uint8_t repeatTimes = TileShape::COLUMN / eleNumPerVectorFractal;

        AscendC::CopyRepeatParams repeatParams;
        repeatParams.dstStride = 1;
        repeatParams.srcStride = 1;
        repeatParams.dstRepeatSize = BLK_NUM_PER_VECTOR_FRACTAL;
        repeatParams.srcRepeatSize = BLK_NUM_PER_VECTOR_FRACTAL;

        for (uint32_t rowOffset = 1; rowOffset < TileShape::ROW; ++rowOffset) {
            AscendC::Copy(ubInOut[rowOffset * TileShape::COLUMN], ubInOut, mask, repeatTimes, repeatParams);
        }
    }
};

}  // namespace Act::Epilogue::Tile

#endif
