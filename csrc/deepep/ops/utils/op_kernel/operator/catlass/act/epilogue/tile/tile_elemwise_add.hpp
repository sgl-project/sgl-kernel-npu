#ifndef ACT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP
#define ACT_EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP

#include "../../../act/act.hpp"

namespace Act::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    uint32_t COMPUTE_LENGTH_>
struct TileElemWiseAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    ACT_DEVICE
    TileElemWiseAdd() {}

    ACT_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const &ubOut,
                    AscendC::LocalTensor<ElementCompute> const &ubIn0,
                    AscendC::LocalTensor<ElementCompute> const &ubIn1)
    {
        // Do the calculation
        AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH);
    }
};

}  // namespace Act::Epilogue::Tile

#endif
