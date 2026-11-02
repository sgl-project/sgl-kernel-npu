#ifndef ACT_EPILOGUE_TILE_TILE_SWIZZLE_HPP
#define ACT_EPILOGUE_TILE_TILE_SWIZZLE_HPP

#include "../../../act/act.hpp"
#include "../../../act/detail/alignment.hpp"
#include "../../../act/matrix_coord.hpp"

namespace Act::Epilogue::Tile {

struct EpilogueIdentityTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    ACT_DEVICE
    EpilogueIdentityTileSwizzle() = default;

    ACT_DEVICE
    EpilogueIdentityTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape)
        : blockShape(blockShape), tileShape(tileShape)
    {
        loopsMN = CeilDiv(blockShape, tileShape);
    }

    ACT_DEVICE
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    ACT_DEVICE
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{loopIdx / loopsMN.column(), loopIdx % loopsMN.column()};
    }

    ACT_DEVICE
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

struct EpilogueHorizontalTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    ACT_DEVICE
    EpilogueHorizontalTileSwizzle() = default;

    ACT_DEVICE
    EpilogueHorizontalTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape)
        : blockShape(blockShape), tileShape(tileShape)
    {
        loopsMN = CeilDiv(blockShape, tileShape);
    }

    ACT_DEVICE
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    ACT_DEVICE
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{loopIdx % loopsMN.row(), loopIdx / loopsMN.row()};
    }

    ACT_DEVICE
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

}  // namespace Act::Epilogue::Tile

#endif  // ACT_EPILOGUE_TILE_TILE_SWIZZLE_HPP
