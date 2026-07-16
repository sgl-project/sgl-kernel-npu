/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ASWT_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ASWT_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/detail/alignment.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"
#include "tla_a5/layout.hpp"

namespace Catlass::Gemm::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class L1TileShape_, class L0TileShape_, bool isGmm_ = false>
struct BlockSchedulerAswt {
    /// Data members
    uint32_t mTileNum_{0};
    uint32_t nTileNum_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};
    uint32_t m_{0};
    uint32_t n_{0};
    uint32_t k_{0};
    uint32_t tailL1M_{0};
    uint32_t tailL1N_{0};
    uint32_t tileNum_{1};
    uint32_t mainWindow_{1};
    uint32_t mainRow_{1};
    uint32_t tailWindow_{1};
    uint32_t mTileIdx_{1};
    uint32_t nTileIdx_{1};
    uint32_t nSplitOffset_{0};
    uint32_t mSplitOffset_{0};
    uint32_t mTailSplitCnt_{1};
    uint32_t nTailSplitCnt_{1};
    uint32_t tailSplitCnt_{1};
    uint32_t blockM_{0};
    uint32_t blockN_{0};
    uint32_t blockK_{0};
    uint32_t round_{0};
    uint32_t tileIdx_{0};
    uint32_t startBlockIdx_{0};
    uint32_t endBlockIdx_{0};

    static constexpr uint32_t ML1_ = tla::get<0>(L1TileShape_{});
    static constexpr uint32_t NL1_ = tla::get<1>(L1TileShape_{});
    static constexpr uint32_t KL1_ = tla::get<2>(L1TileShape_{});
    static constexpr uint32_t ML0_ = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t NL0_ = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t KL0_ = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t WINDOW_LEN = 4UL;

    CATLASS_DEVICE 
    BlockSchedulerAswt(uint32_t blockIdx, uint32_t blockNum, const GemmCoord& shape = {})
    {
        blockIdx_ = blockIdx;
        blockNum_ = blockNum;
        endBlockIdx_ = blockNum - 1;
        if constexpr (!isGmm_) {
            UpdateGroupParams(shape);
        }
    }

    CATLASS_DEVICE 
    void UpdateGroupParams(GemmCoord shape)
    {
        m_ = shape.m();
        n_ = shape.n();
        k_ = shape.k();

        if (m_ <= 0 || n_ <= 0 || k_ <= 0) {
            round_ = 0;
            return;
        }

        mTileNum_ = CeilDiv(m_, ML1_);
        nTileNum_ = CeilDiv(n_, NL1_);
        tileNum_ = mTileNum_ * nTileNum_;

        tailL1M_ = m_ - (mTileNum_ - 1) * ML1_;
        tailL1N_ = n_ - (nTileNum_ - 1) * NL1_;

        round_ = CeilDiv(tileNum_, blockNum_);

        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainRow_ * mainWindow_;

        startBlockIdx_ = endBlockIdx_ == blockNum_ - 1 ? 0 : (endBlockIdx_ + 1);
        endBlockIdx_= (tileNum_ + startBlockIdx_ - 1) % blockNum_;
        
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            --round_;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            --round_;
        }
    }

    CATLASS_DEVICE
    void UpdateTailTile()
    {
        uint32_t tailTileNum = endBlockIdx_ + 1;
        uint32_t remainTile = blockNum_ / tailTileNum;
        if (remainTile <= 1) {
            return;
        }

        uint32_t mTile = AscendC::Std::min(tailL1M_, remainTile);
        uint32_t nTile = AscendC::Std::min(tailL1N_, remainTile);
        while (mTile * nTile > remainTile) {
            if (mTile >= nTile) {
                --mTile;
            } else {
                --nTile;
            }
        }

        uint32_t splitTailM = CeilDiv(tailL1M_, mTile);
        uint32_t splitTailN = CeilDiv(tailL1N_, nTile);

        mTailSplitCnt_ = CeilDiv(tailL1M_, splitTailM);
        nTailSplitCnt_ = CeilDiv(tailL1N_, splitTailN);
        tailSplitCnt_ = mTailSplitCnt_ * nTailSplitCnt_;

        uint32_t newEndBlockIdx = tailSplitCnt_ * (endBlockIdx_ + 1) - 1;
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            ++round_;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    CATLASS_DEVICE 
    void UpdateMNTileIdx(uint32_t roundIdx, bool isLastGroupRound)
    {
        uint32_t newBlockIdx = isLastGroupRound ? (blockIdx_ / tailSplitCnt_) : blockIdx_;
        tileIdx_ = newBlockIdx + roundIdx * blockNum_;

        if constexpr (isGmm_) {
            if (blockIdx_ < startBlockIdx_) {
                tileIdx_ += blockNum_ - startBlockIdx_;
            } else {
                tileIdx_ -= startBlockIdx_;
            }
        }

        uint32_t rowIdx = tileIdx_ / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx_ = rowIdx * mainWindow_ + tileIdx_ % mainWindow_;
            nTileIdx_ = (tileIdx_ / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            uint32_t tailIndex = tileIdx_ - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx_ = (tailIndex / tailWindow_) % nTileNum_;
        }

        if (rowIdx % 2 != 0) {
            nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
        }
    }

    CATLASS_DEVICE 
    void UpdateBlockShape(uint32_t roundIdx, bool isLastGroupRound)
    {
        blockM_ = mTileIdx_ != (mTileNum_ - 1) ? ML1_ : tailL1M_;
        blockN_ = nTileIdx_ != (nTileNum_ - 1) ? NL1_ : tailL1N_;

        if (!isLastGroupRound || (mTailSplitCnt_ == 1 && nTailSplitCnt_ == 1)) {
            return;
        }

        if (roundIdx == round_ - 1) {
            uint32_t splitBlkM = CeilDiv(blockM_, mTailSplitCnt_);
            uint32_t splitBlkN = CeilDiv(blockN_, nTailSplitCnt_);

            uint32_t splitIdx = blockIdx_ % tailSplitCnt_;
            uint32_t mSplitIdx = splitIdx % mTailSplitCnt_;
            uint32_t nSplitIdx = splitIdx / mTailSplitCnt_;

            mSplitOffset_ = mSplitIdx * splitBlkM;
            nSplitOffset_ = nSplitIdx * splitBlkN;

            if (mSplitOffset_ >= blockM_ || nSplitOffset_ >= blockN_) {
                blockM_ = 0;
                blockN_ = 0;
                return;
            }

            blockM_ = AscendC::Std::min(blockM_ - mSplitOffset_, splitBlkM);
            blockN_ = AscendC::Std::min(blockN_ - nSplitOffset_, splitBlkN);
        }
    }

    CATLASS_DEVICE 
    GemmCoord GetBlockShape() const
    {
        return {blockM_, blockN_, k_};
    }

    CATLASS_DEVICE 
    GemmCoord GetBlockCoordByElement() const
    {
        return {mTileIdx_ * ML1_ + mSplitOffset_, nTileIdx_ * NL1_ + nSplitOffset_, 0};
    }

    CATLASS_DEVICE 
    uint32_t GetStartBlockIdx() const
    {
        return startBlockIdx_;
    }

    CATLASS_DEVICE 
    uint32_t GetEndBlockIdx() const
    {
        return endBlockIdx_;
    }
};
} // namespace Catlass::Gemm::Block 

#endif