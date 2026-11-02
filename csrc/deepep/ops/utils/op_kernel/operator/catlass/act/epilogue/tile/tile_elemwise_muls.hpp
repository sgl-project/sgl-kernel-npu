#ifndef ACT_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
#define ACT_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP

#include "../../../act/gemm/helper.hpp"

namespace Act::Epilogue::Tile {
template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileElemWiseMuls {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    ACT_DEVICE
    TileElemWiseMuls() {}

    ACT_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> dstLocal, AscendC::LocalTensor<ElementCompute> srcTensor,
                    ElementCompute scalar)
    {
        AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH);
    }
};
}  // namespace Act::Epilogue::Tile

#endif  // ACT_EPILOGUE_TILE_TILE_ELEMWISE_MULS_HPP
