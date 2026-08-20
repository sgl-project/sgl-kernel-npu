#ifndef ACT_EPILOGUE_TILE_TILE_BROADCAST_ONE_BLK_HPP
#define ACT_EPILOGUE_TILE_TILE_BROADCAST_ONE_BLK_HPP

#include "../../../act/act.hpp"

namespace Act::Epilogue::Tile {

template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileBroadcastOneBlk {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    ACT_DEVICE
    TileBroadcastOneBlk() {}

    ACT_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const &ubOut, AscendC::LocalTensor<ElementCompute> const &ubIn)
    {
        constexpr uint32_t maxRepeatNum = 255;
        constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementCompute);

        AscendC::BrcbRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = BLK_NUM_PER_VECTOR_FRACTAL;

        constexpr uint32_t eleNumPerCompute = RoundDown<eleNumPerBlk>(maxRepeatNum * BLK_NUM_PER_VECTOR_FRACTAL);
        for (uint32_t offset = 0; offset < COMPUTE_LENGTH; offset += eleNumPerCompute) {
            uint32_t residueM = COMPUTE_LENGTH - offset;
            uint32_t computeM = (residueM > eleNumPerCompute) ? eleNumPerCompute : residueM;
            uint8_t repeatTimes = static_cast<uint8_t>(CeilDiv<BLK_NUM_PER_VECTOR_FRACTAL>(computeM));
            AscendC::Brcb(ubOut[offset * eleNumPerBlk], ubIn[offset], repeatTimes, repeatParams);
        }
    }
};

}  // namespace Act::Epilogue::Tile

#endif
