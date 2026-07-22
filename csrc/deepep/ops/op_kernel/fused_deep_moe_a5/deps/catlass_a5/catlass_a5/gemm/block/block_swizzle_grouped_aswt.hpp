/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_GROUPED_ASWT_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_GROUPED_ASWT_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/detail/alignment.hpp"
#include "catlass_a5/gemm_coord.hpp"

namespace Catlass::Gemm::Block {
template <uint32_t WindowLen = 4, bool TransA = false, bool TransB = true>
struct GemmGroupedAswtTailSplitSwizzle {
    static constexpr uint32_t BLOCK_CUBE = 16;                 // cube alignment
    static constexpr uint32_t INNER_AXIS_MIN_SPLIT_VAL = 128;  // ND2NZ cache-line size

    /// Result of GetBlockShape: tile size (m, n) plus the sub-tile offset inside the base tile.
    /// For non tail-split tiles mOffset == nOffset == 0.
    struct AswtBlockShape {
        uint32_t m;
        uint32_t n;
        uint32_t mOffset;
        uint32_t nOffset;
    };

    /// Data members ---------------------------------------------------------------------------

    // Current group problem dims.
    uint32_t m_{0};
    uint32_t n_{0};
    uint32_t k_{0};
    // Base tile. baseM_ may be rebalanced per group through UpdateBaseM().
    uint32_t baseM_{1};
    uint32_t baseN_{1};
    // Derived per-problem quantities (cached while m/n unchanged across groups).
    uint32_t mCnt_{1};
    uint32_t nCnt_{1};
    uint32_t totalCnt_{1};
    uint32_t mBaseTail_{0};
    uint32_t nBaseTail_{0};
    uint32_t mainMWindow_{1};
    uint32_t mainRow_{0};
    uint32_t tailWindow_{1};
    // Per-core rolling round bookkeeping (carried across groups, no barrier).
    uint32_t blockNum_{1};
    uint32_t blockIdx_{0};
    uint32_t startBlockIdx_{0};
    uint32_t endBlockIdx_{0};
    uint32_t round_{0};
    uint32_t roundIdx_{0};
    // Tail-split state (only the last group ever sets tailCnt_ > 1).
    uint32_t mTailCnt_{1};
    uint32_t nTailCnt_{1};
    uint32_t tailCnt_{1};
    uint32_t tailBlockBase_{0};
    uint32_t nTailAlign_{BLOCK_CUBE};

    /// Methods --------------------------------------------------------------------------------

    CATLASS_DEVICE
    GemmGroupedAswtTailSplitSwizzle() {}

    CATLASS_DEVICE
    GemmGroupedAswtTailSplitSwizzle(uint32_t baseM, uint32_t baseN)
        : baseM_(baseM),
          baseN_(baseN),
          blockNum_(AscendC::GetBlockNum()),
          blockIdx_(AscendC::GetBlockIdx()),
          endBlockIdx_(AscendC::GetBlockNum() - 1)
    {
        // Tail-split N min alignment. For TransB (ColumnMajor weight) aclnn uses BLOCK_CUBE; for
        // !TransB the split is additionally forced to INNER_AXIS_MIN_SPLIT_VAL inside GetBlockShape
        // (a multiple of BLOCK_CUBE), so this default suffices for both layouts.
        nTailAlign_ = BLOCK_CUBE;
    }

    /// Optional base-M rebalance for the current group. Pass L1_TILE_M for plain ASWT.
    CATLASS_DEVICE
    void UpdateBaseM(uint32_t baseM)
    {
        baseM_ = baseM;
    }

    /// Start a new group. Recomputes window params (cached on m/n) and rolls the core window.
    CATLASS_DEVICE
    void UpdateNextProblem(GemmCoord const &problemShape)
    {
        k_ = problemShape.k();
        if (m_ != problemShape.m() || n_ != problemShape.n()) {
            m_ = problemShape.m();
            n_ = problemShape.n();
            mCnt_ = CeilDiv(m_, baseM_);
            nCnt_ = CeilDiv(n_, baseN_);
            mBaseTail_ = m_ - (mCnt_ - 1) * baseM_;
            nBaseTail_ = n_ - (nCnt_ - 1) * baseN_;
            totalCnt_ = mCnt_ * nCnt_;
            mainMWindow_ = (WindowLen < mCnt_) ? WindowLen : mCnt_;
            mainRow_ = mCnt_ / mainMWindow_ - 1;
            tailWindow_ = mCnt_ - mainMWindow_ * mainRow_;
        }
        // Reset tail-split state for every group (only the last group re-enables it).
        mTailCnt_ = 1;
        nTailCnt_ = 1;
        tailCnt_ = 1;
        tailBlockBase_ = 0;

        roundIdx_ = 0;
        round_ = CeilDiv(totalCnt_, blockNum_);
        // The first physical block of the new group continues right after the previous group's end.
        startBlockIdx_ = (endBlockIdx_ == blockNum_ - 1) ? 0 : (endBlockIdx_ + 1);
        // The last physical block of the new group.
        endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;
        // Cores outside [startBlockIdx_, endBlockIdx_] handle one fewer round.
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    /// Number of physical cores that own a tile in the current group's final wave.
    CATLASS_DEVICE
    uint32_t GetTailTileCnt() const
    {
        return Min(endBlockIdx_ + 1, totalCnt_);
    }

    CATLASS_DEVICE
    uint32_t GetEndBlockIdx() const
    {
        return endBlockIdx_;
    }

    /// Whether a tail-split is worthwhile: only when at least half the cores are idle in the final
    /// wave (matches aclnn IfNeedSplit).
    CATLASS_DEVICE
    bool NeedTailSplit() const
    {
        return (endBlockIdx_ + 1) <= (blockNum_ / 2);
    }

    /// Explicit tail-split factors.
    CATLASS_DEVICE
    void UpdateTailTile(uint32_t mTailCnt, uint32_t nTailCnt)
    {
        mTailCnt_ = mTailCnt;
        nTailCnt_ = nTailCnt;
        tailCnt_ = mTailCnt_ * nTailCnt_;
        uint32_t tailOriCnt = GetTailTileCnt();
        uint32_t newEndBlockIdx = endBlockIdx_ + tailOriCnt * (tailCnt_ - 1);
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailCnt_ = 1;
            nTailCnt_ = 1;
            tailCnt_ = 1;
            tailBlockBase_ = 0;
        } else if (tailCnt_ > 1) {
            // Base physical block of the original (unsplit) tail-tile window.
            tailBlockBase_ = endBlockIdx_ + 1 - tailOriCnt;
        } else {
            tailBlockBase_ = 0;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    /// Auto-pick tail-split factors to spread the tail tiles over the remaining idle cores.
    CATLASS_DEVICE
    void UpdateTailTile()
    {
        uint32_t tailOriCnt = GetTailTileCnt();
        uint32_t remainTile = (blockNum_ - endBlockIdx_ - 1) / tailOriCnt + 1;
        if (remainTile <= 1) {
            return;
        }
        uint32_t mMin = BLOCK_CUBE;
        uint32_t nMin = BLOCK_CUBE;
        if constexpr (TransA) {
            mMin = INNER_AXIS_MIN_SPLIT_VAL;
        }
        if constexpr (!TransB) {
            nMin = INNER_AXIS_MIN_SPLIT_VAL;
        }
        uint32_t mTile = Min(CeilDiv(mBaseTail_, mMin), remainTile);
        uint32_t nTile = Min(CeilDiv(nBaseTail_, nMin), remainTile);
        while (mTile * nTile > remainTile) {
            if (mTile >= nTile) {
                mTile -= 1;
            } else {
                nTile -= 1;
            }
        }
        UpdateTailTile(mTile, nTile);
    }

    /// Advance to this core's next tile. Returns false once the core has no more work.
    CATLASS_DEVICE
    bool GetTileIdx(GemmCoord &blockCoord)
    {
        if (round_ == 0 || roundIdx_ > round_ - 1) {
            return false;
        }
        int64_t bn = static_cast<int64_t>(blockNum_);
        int64_t tc = static_cast<int64_t>(tailCnt_);
        int64_t tbb = static_cast<int64_t>(tailBlockBase_);
        int64_t sbi = static_cast<int64_t>(startBlockIdx_);

        int64_t newBlockIdx = static_cast<int64_t>(blockIdx_);
        if (roundIdx_ == round_ - 1 && tailCnt_ > 1) {
            newBlockIdx = (tbb + ((newBlockIdx - tbb) / tc) * tc) / tc;
        }
        int64_t index = newBlockIdx + static_cast<int64_t>(roundIdx_) * bn;
        // Apply the startBlockIdx offset.
        if (blockIdx_ < startBlockIdx_) {
            index += bn - sbi;
        } else if (tailCnt_ > 1 && static_cast<int64_t>(endBlockIdx_) + 1 >= tc * static_cast<int64_t>(totalCnt_)) {
            index -= (tbb + ((sbi - tbb) / tc) * tc) / tc;
        } else {
            index -= sbi;
        }

        int64_t mcW = static_cast<int64_t>(nCnt_) * static_cast<int64_t>(mainMWindow_);
        int64_t rowIdx = index / mcW;
        int64_t mIdx;
        int64_t nIdx;
        if (rowIdx < static_cast<int64_t>(mainRow_)) {
            mIdx = rowIdx * static_cast<int64_t>(mainMWindow_) + index % static_cast<int64_t>(mainMWindow_);
            nIdx = (index / static_cast<int64_t>(mainMWindow_)) % static_cast<int64_t>(nCnt_);
        } else {
            rowIdx = static_cast<int64_t>(mainRow_);
            int64_t tailIndex = index - static_cast<int64_t>(mainRow_) * mcW;
            mIdx = static_cast<int64_t>(mainRow_) * static_cast<int64_t>(mainMWindow_) +
                   tailIndex % static_cast<int64_t>(tailWindow_);
            nIdx = (tailIndex / static_cast<int64_t>(tailWindow_)) % static_cast<int64_t>(nCnt_);
        }
        if (rowIdx & 1) {
            nIdx = static_cast<int64_t>(nCnt_) - 1 - nIdx;
        }
        roundIdx_++;
        blockCoord = GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), 0};
        return true;
    }

    /// Tile shape (and tail-split sub-tile offsets) for the coord returned by GetTileIdx().
    CATLASS_DEVICE
    AswtBlockShape GetBlockShape(GemmCoord const &blockCoord)
    {
        uint32_t singleCoreM = (blockCoord.m() != (mCnt_ - 1)) ? baseM_ : mBaseTail_;
        uint32_t singleCoreN = (blockCoord.n() != (nCnt_ - 1)) ? baseN_ : nBaseTail_;
        // Full tile for every round except the final (split) round (roundIdx_ already incremented).
        if (tailCnt_ == 1 || roundIdx_ < round_) {
            return AswtBlockShape{singleCoreM, singleCoreN, 0, 0};
        }

        uint32_t singleCoreMSplit = CeilDiv(singleCoreM, mTailCnt_);
        uint32_t singleCoreNSplit = CeilDiv(singleCoreN, nTailCnt_);
        if constexpr (TransA) {
            singleCoreMSplit = RoundUp(singleCoreMSplit, INNER_AXIS_MIN_SPLIT_VAL);
        }
        singleCoreNSplit = RoundUp(singleCoreNSplit, nTailAlign_);
        if constexpr (!TransB) {
            singleCoreNSplit = RoundUp(singleCoreNSplit, INNER_AXIS_MIN_SPLIT_VAL);
        }
        uint32_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        uint32_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        uint32_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        uint32_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
            return AswtBlockShape{0, 0, 0, 0};
        }
        singleCoreM = Min(singleCoreM - mSplitAddrOffset, singleCoreMSplit);
        singleCoreN = Min(singleCoreN - nSplitAddrOffset, singleCoreNSplit);
        return AswtBlockShape{singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset};
    }
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_GROUPED_ASWT_HPP
