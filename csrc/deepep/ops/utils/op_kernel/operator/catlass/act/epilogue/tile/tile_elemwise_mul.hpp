#ifndef ACT_EPILOGUE_TILE_TILE_ELEMWISE_MUL_HPP
#define ACT_EPILOGUE_TILE_TILE_ELEMWISE_MUL_HPP

#include "../../../act/act.hpp"

namespace Act::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    class TileShape_>
struct TileElemwiseMul {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    ACT_DEVICE
    TileElemwiseMul() {}

    ACT_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const &ubOut,
                    AscendC::LocalTensor<ElementCompute> const &ubIn0,
                    AscendC::LocalTensor<ElementCompute> const &ubIn1)
    {
        // Do the calculation
        AscendC::Mul(ubOut, ubIn0, ubIn1, TileShape::COUNT);
    }
};

}  // namespace Act::Epilogue::Tile

#endif
