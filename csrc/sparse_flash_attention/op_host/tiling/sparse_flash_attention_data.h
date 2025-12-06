#ifndef SPARSE_FLASH_ATTENTION_DATA_H
#define SPARSE_FLASH_ATTENTION_DATA_H

#include <cstdint>

namespace sglang::SFAHost {

#pragma pack(push, 1)
struct SparseFlashAttentionBaseParamsMla {
    uint32_t batchSize= 0U;
    uint32_t seqSize = 0U;
    uint32_t qSeqSize = 0U;
    int64_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0U;
    float scaleValue = 0.0f;
    uint32_t nNumOfQInOneGroup = 0U;
    uint32_t actualLenDimsQ = 0U;
    uint32_t actualLenDimsKV = 0U;
    uint32_t outputLayout = 0U;
    uint32_t sparseMode = 0U;
    int64_t sparseBlockSize = 0;
    uint32_t sparseBlockCount = 0U;
};

struct SparseFlashAttentionSplitKVParamsMla {
    uint32_t s2 = 0U;             // S2
    uint32_t accumOutSize = 0U;   // FD workspace
    uint32_t logSumExpSize = 0U;  // FD workspace
};

struct SparseFlashAttentionSingleCoreParamsMla {
    uint32_t usedCoreNum = 0U;
};

struct SparseFlashAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;
};

// InnerSplitParams 已在 sparse_flash_attention_tiling.h 中定义
struct SparseFlashAttentionInnerSplitParams {
    uint32_t mBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

// -----------算子TilingData定义---------------
struct SparseFlashAttentionTilingDataMla {
    SparseFlashAttentionBaseParamsMla baseParams;
    SparseFlashAttentionSplitKVParamsMla splitKVParams;
    SparseFlashAttentionSingleCoreParamsMla singleCoreParams;
    SparseFlashAttentionSingleCoreTensorSizeMla singleCoreTensorSize;
    SparseFlashAttentionInnerSplitParams innerSplitParams;
    uint32_t tilingKey = 0U;
};
#pragma pack(pop)
} // namespace sglang::SFAHost
#endif // SPARSE_FLASH_ATTENTION_DATA_H
