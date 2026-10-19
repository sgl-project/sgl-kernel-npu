// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.
#ifndef SGL_KERNEL_NPU_FUSED_COPY_SFA_TILING_DATA_H
#define SGL_KERNEL_NPU_FUSED_COPY_SFA_TILING_DATA_H

#include <cstddef>
#include <cstdint>

#include "sparse_flash_attention_tiling_data.h"

// Direct-launch ABI: the first 144 bytes are byte-for-byte the SGLang SFA
// tiling POD; the final 16 bytes describe source-aware gather capacities.
// Keep this header usable by both host and AscendC compilation.
struct alignas(8) FusedCopySfaTilingData {
    SparseFlashAttentionBaseParamsMla baseParams;
    SparseFlashAttentionSplitKVParamsMla splitKVParams;
    SparseFlashAttentionSingleCoreParamsMla singleCoreParams;
    SparseFlashAttentionSingleCoreTensorSizeMla singleCoreTensorSize;
    SparseFlashAttentionInnerSplitParams innerSplitParams;
    uint32_t copyCap;
    uint32_t missCap;
    uint32_t hbmMaxBlockNum;
    uint32_t dramMaxBlockNum;
};

static_assert(sizeof(SparseFlashAttentionTilingDataMla) == 144);
static_assert(offsetof(FusedCopySfaTilingData, splitKVParams) ==
              offsetof(SparseFlashAttentionTilingDataMla, splitKVParams));
static_assert(offsetof(FusedCopySfaTilingData, singleCoreParams) ==
              offsetof(SparseFlashAttentionTilingDataMla, singleCoreParams));
static_assert(offsetof(FusedCopySfaTilingData, singleCoreTensorSize) ==
              offsetof(SparseFlashAttentionTilingDataMla, singleCoreTensorSize));
static_assert(offsetof(FusedCopySfaTilingData, innerSplitParams) ==
              offsetof(SparseFlashAttentionTilingDataMla, innerSplitParams));
static_assert(offsetof(FusedCopySfaTilingData, copyCap) == 144);
static_assert(offsetof(FusedCopySfaTilingData, missCap) == 148);
static_assert(offsetof(FusedCopySfaTilingData, hbmMaxBlockNum) == 152);
static_assert(offsetof(FusedCopySfaTilingData, dramMaxBlockNum) == 156);
static_assert(sizeof(FusedCopySfaTilingData) == 160);

#endif  // SGL_KERNEL_NPU_FUSED_COPY_SFA_TILING_DATA_H
