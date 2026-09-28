#ifndef ACT_EPILOGUE_TILE_TILE_CAST_HPP
#define ACT_EPILOGUE_TILE_TILE_CAST_HPP

#include "../../../act/act.hpp"

namespace Act::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class DstType_, class SrcType_,
    /// Length of the compute buffer
    class TileShape_>
struct TileCast {
    using ArchTag = ArchTag_;
    using ElementDst = typename DstType_::Element;
    using ElementSrc = typename SrcType_::Element;
    using TileShape = TileShape_;

    ACT_DEVICE
    TileCast() {}

    ACT_DEVICE
    void operator()(AscendC::LocalTensor<ElementDst> const &ubOut, AscendC::LocalTensor<ElementSrc> const &ubIn)
    {
        AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
    }
};

}  // namespace Act::Epilogue::Tile

#endif
