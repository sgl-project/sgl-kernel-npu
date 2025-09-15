/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_BLOCK_BLOCK_SWIZZLE_HPP
#define CATLASS_CONV_BLOCK_BLOCK_SWIZZLE_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Conv::Block {
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Block swizzling function for conv3d
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct Conv3dIdentityBlockSwizzle {
    /// Data members
    Conv3d6HdCoord outShape;
    Conv3d6HdCoord coreTileShape;
    MatrixCoord loopsMN;
    Conv3d6HdCoord  loops;
    uint64_t nStart, doStart, co1Start, howoStart;

    // Methods
    CATLASS_DEVICE
    Conv3dIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    Conv3dIdentityBlockSwizzle(Conv3d6HdCoord const &outShape_, Conv3d6HdCoord const &loops_)
        : outShape(outShape_), loops(loops_)
    {
        loops = Conv3d6HdCoord{min(outShape.n(), loops.n()), min(outShape.d(), loops.d()),
                               min(outShape.c1(), loops.c1()), min(outShape.hw(), loops.hw())};
        coreTileShape = Conv3d6HdCoord{CeilDiv(outShape.n(), loops.n()), CeilDiv(outShape.d(), loops.d()),
                                       CeilDiv(outShape.c1(), loops.c1()), CeilDiv(outShape.hw(), loops.hw())};
        loopsMN = MatrixCoord{loops.hw(), loops.n() * loops.d() * loops.c1()};
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    CATLASS_DEVICE
    Conv3d6HdCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) {
            uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMN.column() - nIdx - 1;
            }

            uint32_t howoIdx = mIdx;

            uint32_t noIdx = nIdx / (loops[1]*loops[2]);
            uint32_t doIdx = nIdx / loops[2] % loops[1];
            uint32_t c1Idx = nIdx % loops[2];
            return Conv3d6HdCoord{noIdx, doIdx, c1Idx, howoIdx};
        } else if constexpr (SwizzleDirection ==1) {
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }

            uint32_t howoIdx = mIdx;

            uint32_t noIdx = nIdx / (loops[1]*loops[2]);
            uint32_t doIdx = nIdx / loops[2] % loops[1];
            uint32_t c1Idx = nIdx % loops[2];
            return Conv3d6HdCoord{noIdx, doIdx, c1Idx, howoIdx};
        }
    }

    CATLASS_DEVICE
    Conv3d6HdCoord GetDimStartIdx(Conv3d6HdCoord blockCoord)
    {
        uint32_t nStart = blockCoord.n() * coreTileShape.n();
        uint32_t doStart = blockCoord.d() * coreTileShape.d();
        uint32_t c1Start = blockCoord.c1() * coreTileShape.c1();
        uint32_t howoStart = blockCoord.hw() * coreTileShape.hw();
        return Conv3d6HdCoord{nStart, doStart, c1Start, howoStart};
    }

    CATLASS_DEVICE
    Conv3d6HdCoord GetActualBlockShape(Conv3d6HdCoord blockCoord, Conv3d6HdCoord dimStartIdx)
    {
        uint32_t nActual = (blockCoord.n() == loops.n() - 1) ?
            (outShape[0] - dimStartIdx.n()) : coreTileShape.n();
        
        uint32_t doActual = (blockCoord.d() == loops.d() - 1) ?
            (outShape[1] - dimStartIdx.d()) : coreTileShape.d();
        
        uint32_t c1Actual = (blockCoord.c1() == loops.c1() - 1) ?
            (outShape[2] - dimStartIdx.c1()) : coreTileShape.c1();
        
        uint32_t hwActual = (blockCoord.hw() == loops.hw() - 1) ?
            (outShape[3] - dimStartIdx.hw()) : coreTileShape.hw();
        return Conv3d6HdCoord{nActual, doActual, c1Actual, hwActual};
    }
};
}  // namespace Catlass::Conv::Block

#endif  // CATLASS_CONV_BLOCK_BLOCK_SWIZZLE_HPP
