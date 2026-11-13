/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SPARSE_ATTN_SHAREDKV_TREE_TILING_DATA_H
#define SPARSE_ATTN_SHAREDKV_TREE_TILING_DATA_H

#include <cstdint>

namespace optiling {

struct SparseAttnSharedkvTreeSwaParams {
    uint32_t batchSize = 0;
    uint32_t qSeqSize = 0;
    uint32_t kvSeqSize = 0;
    int64_t paBlockSize = 0;
    int64_t oriBlockSize = 0;
    int64_t cmpBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t actualLenDimsQ = 0;
    uint32_t actualLenDimsKV = 0;
    float softmaxScale = 1.0F;
    uint32_t outputLayout = 0;
    uint64_t oriMaskMode = 4;
    int64_t oriKvStride = 0;
    int64_t oriWinLeft = 127;
    int64_t oriWinRight = 0;
    int64_t sparseBlockSize = 1;
    bool hasOriSparseIndices = false;
    uint32_t oriSparseIndexWidth = 0;
    uint32_t usedCoreNum = 0;
    uint32_t mmResUbSize = 0;
    uint32_t bmm2ResUbSize = 0;
    uint32_t mBaseSize = 0;
    uint32_t s2BaseSize = 0;
    bool returnSoftmaxLse = false;

    void set_batchSize(uint32_t v) { batchSize = v; }
    void set_qSeqSize(uint32_t v) { qSeqSize = v; }
    void set_kvSeqSize(uint32_t v) { kvSeqSize = v; }
    void set_paBlockSize(int64_t v) { paBlockSize = v; }
    void set_oriBlockSize(int64_t v) { oriBlockSize = v; }
    void set_cmpBlockSize(int64_t v) { cmpBlockSize = v; }
    void set_oriMaxBlockNumPerBatch(uint32_t v) { oriMaxBlockNumPerBatch = v; }
    void set_nNumOfQInOneGroup(uint32_t v) { nNumOfQInOneGroup = v; }
    void set_actualLenDimsQ(uint32_t v) { actualLenDimsQ = v; }
    void set_actualLenDimsKV(uint32_t v) { actualLenDimsKV = v; }
    void set_softmaxScale(float v) { softmaxScale = v; }
    void set_outputLayout(uint32_t v) { outputLayout = v; }
    void set_oriMaskMode(uint64_t v) { oriMaskMode = v; }
    void set_oriKvStride(int64_t v) { oriKvStride = v; }
    void set_oriWinLeft(int64_t v) { oriWinLeft = v; }
    void set_oriWinRight(int64_t v) { oriWinRight = v; }
    void set_sparseBlockSize(int64_t v) { sparseBlockSize = v; }
    void set_hasOriSparseIndices(bool v) { hasOriSparseIndices = v; }
    void set_oriSparseIndexWidth(uint32_t v) { oriSparseIndexWidth = v; }
    void set_usedCoreNum(uint32_t v) { usedCoreNum = v; }
    void set_mmResUbSize(uint32_t v) { mmResUbSize = v; }
    void set_bmm2ResUbSize(uint32_t v) { bmm2ResUbSize = v; }
    void set_mBaseSize(uint32_t v) { mBaseSize = v; }
    void set_s2BaseSize(uint32_t v) { s2BaseSize = v; }
    void set_returnSoftmaxLse(bool v) { returnSoftmaxLse = v; }
};

struct SparseAttnSharedkvTreeCmpParams {
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t sparseBlockCount = 0;
    int64_t cmpRatio = 1;
    uint64_t cmpMaskMode = 3;
    int64_t cmpKvStride = 0;
    uint32_t hasTreeCmpKv = 0;
    uint32_t ancestorStride = 0;
    uint64_t treeNodeCount = 0;
    int64_t cmpCacheSpan = 0;

    void set_cmpMaxBlockNumPerBatch(uint32_t v) { cmpMaxBlockNumPerBatch = v; }
    void set_sparseBlockCount(uint32_t v) { sparseBlockCount = v; }
    void set_cmpRatio(int64_t v) { cmpRatio = v; }
    void set_cmpMaskMode(uint64_t v) { cmpMaskMode = v; }
    void set_cmpKvStride(int64_t v) { cmpKvStride = v; }
    void set_hasTreeCmpKv(uint32_t v) { hasTreeCmpKv = v; }
    void set_ancestorStride(uint32_t v) { ancestorStride = v; }
    void set_treeNodeCount(uint64_t v) { treeNodeCount = v; }
    void set_cmpCacheSpan(int64_t v) { cmpCacheSpan = v; }
};

struct SparseAttnSharedkvTreeTilingData {
    SparseAttnSharedkvTreeSwaParams baseParams{};
    SparseAttnSharedkvTreeCmpParams cmpParams{};
};

}  // namespace optiling
#endif
