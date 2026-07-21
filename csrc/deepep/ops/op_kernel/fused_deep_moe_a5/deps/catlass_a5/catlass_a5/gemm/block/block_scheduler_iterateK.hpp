/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ITERATEK_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ITERATEK_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/detail/alignment.hpp"
#include "catlass_a5/gemm_coord.hpp"
#include "catlass_a5/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class L1Shape_, class L0Shape_>
struct BlockSchedulerIterateK {
    CATLASS_DEVICE
    BlockSchedulerIterateK() {}

    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t kTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t b_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t totalTileNum_{0};
    int64_t aicCoreNum_{0};

    using BlockShape = tla::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = tla::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;
    using L1Shape = L1Shape_;
    using L0Shape = L0Shape_;

    static constexpr int64_t l1M = tla::get<0>(L1Shape{});
    static constexpr int64_t l1N = tla::get<1>(L1Shape{});
    static constexpr int64_t l1K = tla::get<2>(L1Shape{});
    static constexpr int64_t l0M = tla::get<0>(L0Shape{});
    static constexpr int64_t l0N = tla::get<1>(L0Shape{});
    static constexpr int64_t l0K = tla::get<2>(L0Shape{});

    CATLASS_DEVICE BlockSchedulerIterateK(ProblemShape shape, int64_t blockIdx, int64_t blockNum, int64_t aicCoreNum)
        : blockIdx_(blockIdx), blockNum_(blockNum), aicCoreNum_(aicCoreNum)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        b_ = shape.b ? shape.b : 1;
        mTileNum_ = CeilDiv(m_, l1M);
        nTileNum_ = CeilDiv(n_, l1N);
        kTileNum_ = CeilDiv(k_, l1K);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, b_);
        totalTileNum_ = mTileNum_ * nTileNum_ * b_;
    }

    CATLASS_DEVICE int64_t GetTileNum()
    {
        return totalTileNum_;
    }

    CATLASS_DEVICE BlockShape GetBlockShape(int64_t tileIdx)
    {
        int64_t tailL1M = (m_ % l1M == 0) ? l1M : m_ % l1M;
        int64_t tailL1N = (n_ % l1N == 0) ? l1N : n_ % l1N;
        int64_t tailL1K = (k_ % l1K == 0) ? l1K : k_ % l1K;
        int64_t mTileIdx = tileIdx / nTileNum_;
        int64_t batchTileIdx = tileIdx / (mTileNum_ * nTileNum_);
        mTileIdx = mTileIdx - batchTileIdx * mTileNum_;
        int64_t nTileIdx = tileIdx % nTileNum_;

        int64_t blockShapeM = IsMTail(mTileIdx, mTileNum_) ? tailL1M : l1M;
        int64_t blockShapeN = IsNTail(nTileIdx, nTileNum_) ? tailL1N : l1N;

        return {blockShapeM, blockShapeN, k_, b_};
    }

    CATLASS_DEVICE BlockCoord GetBlockCoord(int64_t tileIdx)
    {
        int64_t mTileIdx = tileIdx / nTileNum_;
        int64_t batchTileIdx = tileIdx / (mTileNum_ * nTileNum_);
        mTileIdx = mTileIdx - batchTileIdx * mTileNum_;
        int64_t nTileIdx = tileIdx % nTileNum_;

        return {mTileIdx * l1M, nTileIdx * l1N, 0, batchTileIdx};
    }

    CATLASS_DEVICE int64_t GetBlockNum(ProblemShape shape)
    {
        int maxCoreNum = aicCoreNum_;
        int64_t mTotalCnt = CeilDiv(shape.m, l1M);
        int64_t nTotalCnt = CeilDiv(shape.n, l1N);
        int64_t batch = shape.b ? shape.b : 1;
        int64_t blockNum = 0;
        int64_t totalCnt = mTotalCnt * nTotalCnt * batch;
        if (totalCnt < maxCoreNum) {
            blockNum = totalCnt;
        } else {
            int64_t perCoreBlockNum = CeilDiv(totalCnt, maxCoreNum);
            blockNum = CeilDiv(totalCnt, perCoreBlockNum);
        }
        return blockNum;
    }

    CATLASS_HOST_DEVICE static size_t GetWorkspaceSize(ProblemShape shape)
    {
        return 0;
    }

    CATLASS_DEVICE int64_t GetPerBlockNum(int64_t coreNum, int64_t mTileNum, int64_t nTileNum, int64_t b = 1)
    {
        int64_t perCoreBlockNum = CeilDiv(mTileNum * nTileNum * b, coreNum);
        return perCoreBlockNum;
    }

    CATLASS_DEVICE bool IsMTail(int64_t mTileIdx, int64_t mTileNum)
    {
        if ((mTileIdx - (mTileNum - 1)) % mTileNum == 0) {
            return true;
        }
        return false;
    }

    CATLASS_DEVICE bool IsNTail(int64_t nTileIdx, int64_t nTileNum)
    {
        if (nTileIdx == (nTileNum - 1)) {
            return true;
        }
        return false;
    }

    CATLASS_DEVICE bool IsKTail(int64_t kTileIdx, int64_t kTileNum)
    {
        if (kTileIdx == (kTileNum - 1)) {
            return true;
        }
        return false;
    }
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_ITERATEK_HPP
