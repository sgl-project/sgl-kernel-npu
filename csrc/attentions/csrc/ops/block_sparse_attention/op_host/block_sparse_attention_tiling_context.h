/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef BLOCK_SPARSE_ATTENTION_TILING_CONTEXT_H
#define BLOCK_SPARSE_ATTENTION_TILING_CONTEXT_H
#include <cstdint>
#include <vector>
#include <queue>
#include "exe_graph/runtime/tiling_context.h"
#include "data_copy_transpose_tiling_def.h"
#include "data_copy_transpose_tiling.h"


#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"


#include "register/op_def_registry.h"

namespace optiling {

/*
contextParams is a new structured defined for the use of FusedInferAttentionScore op.
It is meant to catch and organize all the necessary variables passed by FIAS tilling function.
It will be used as the input to the new 'runBigKernelWithParams' function in BSA tilling.
The old BSA tillingContext will also be transformed to this structure in the future.
*/
struct ContextParamsForBSATiling {
    const gert::Tensor *pseShift = nullptr;
    const gert::Tensor *attentionMask = nullptr;
    const gert::Tensor *actualSeqenceLengthQ = nullptr;
    const gert::Tensor *actualSeqenceLengthKV = nullptr;
    const gert::Tensor *antiquantScale = nullptr;
    const gert::Tensor *antiquantOffset = nullptr;
    const gert::Tensor *queryPaddingSize = nullptr;
    const gert::Tensor *kvPaddingSize = nullptr;
    const gert::Tensor *blockTable = nullptr;
    const gert::Tensor *keySharedPrefix = nullptr;
    const gert::Tensor *valueSharedPrefix = nullptr;
    const gert::Tensor *actualSharedPrefixLen = nullptr;

    const gert::Tensor *KeyAntiquantScale = nullptr;
    const gert::Tensor *valueAntiquantScale = nullptr;
    const gert::Tensor *KeyAntiquantOffset = nullptr;
    const gert::Tensor *valueAntiquantOffset = nullptr;

    const gert::Tensor *sparseMask = nullptr;
    const gert::Tensor *sparseCntTable = nullptr;

    ge::DataType inputDataType = ge::DataType::DT_FLOAT16;
    ge::DataType kDataType = ge::DataType::DT_FLOAT16;
    ge::DataType vDataType = ge::DataType::DT_FLOAT16;
    ge::DataType qRopeDataType = ge::DataType::DT_FLOAT16;
    ge::DataType kRopeDataType = ge::DataType::DT_FLOAT16;
    ge::DataType pseShiftDataType = ge::DataType::DT_FLOAT16;
    ge::DataType maskDataType = ge::DataType::DT_FLOAT16;
    ge::DataType blockTableType = ge::DataType::DT_FLOAT16;
    ge::DataType outputDataType = ge::DataType::DT_FLOAT16;
    const char *opName = nullptr;
    const gert::StorageShape *queryInputShape = nullptr;
    const gert::StorageShape *keyInputShape = nullptr;
    const gert::StorageShape *queryRopeInputShape = nullptr;
    const gert::StorageShape *keyRopeInputShape = nullptr;
    const gert::StorageShape *valueInputShape = nullptr;
    const gert::StorageShape *pseShiftShape = nullptr;
    const gert::StorageShape *attentionMaskShape = nullptr;
    const gert::StorageShape *deqScale1Shape = nullptr;
    const gert::StorageShape *scale1Shape = nullptr;
    const gert::StorageShape *deqScale2Shape = nullptr;
    const gert::StorageShape *scale2Shape = nullptr;
    const gert::StorageShape *offset2Shape = nullptr;
    const gert::StorageShape *antiquantScaleShape = nullptr;
    const gert::StorageShape *antiquantOffsetShape = nullptr;
    const gert::StorageShape *blockTableShape = nullptr;
    const gert::StorageShape *outputShape = nullptr;
    const gert::StorageShape *lseoutputShape = nullptr;

    const gert::StorageShape *KeyAntiquantScaleShape = nullptr;
    const gert::StorageShape *valueAntiquantScaleShape = nullptr;
    const gert::StorageShape *KeyAntiquantOffsetShape = nullptr;
    const gert::StorageShape *valueAntiquantOffsetShape = nullptr;
    const gert::StorageShape *sparseMaskShape = nullptr;
    const gert::StorageShape *queryRope = nullptr;
    const gert::StorageShape *keyRope = nullptr;
    ge::DataType KeyAntiquantScaleType = ge::DataType::DT_FLOAT16;
    ge::DataType valueAntiquantScaleType = ge::DataType::DT_FLOAT16;
    ge::DataType KeyAntiquantOffsetType = ge::DataType::DT_FLOAT16;
    ge::DataType valueAntiquantOffsetType = ge::DataType::DT_FLOAT16;
    ge::DataType sparseMaskType = ge::DataType::DT_FLOAT16;

    const int64_t *innerPrecisePtr = nullptr;
    const int32_t *headsNumber = nullptr;
    const uint8_t *causal = nullptr;
    const int32_t *sparseSize = nullptr;
    const int32_t *sparseMode = nullptr;
    const int64_t *preToken = nullptr;
    const int64_t *nextToken = nullptr;
    const float *scaleValue = nullptr;
    const int32_t *blockSize = nullptr;
    const char *layout = nullptr;
    const int32_t *numKeyValueHeads = nullptr;
    size_t *workspaceSize = nullptr;
    const BlockSparseAttentionCompileInfo *compileInfoPtr = nullptr;
    ge::DataType deqScaleType = ge::DataType::DT_FLOAT16;
    ge::DataType deqScale2Type = ge::DataType::DT_FLOAT16;
    ge::DataType quantScale2Type = ge::DataType::DT_FLOAT16;
    ge::DataType quantOffset2Type = ge::DataType::DT_FLOAT16;
    uint32_t isKvContinuous = 1;
    std::vector<const gert::StorageShape *> kTensorList = {nullptr};
    std::vector<const gert::StorageShape *> vTensorList = {nullptr};
    uint32_t maxKVs = 0;
    uint32_t fromFused = 0;
    uint32_t emptyTensor = 0;
    uint32_t isBSNDOut = 0;
    const bool *softmaxLseFlag = nullptr;
    bool isSoftMaxLseEnable = false;
    // Flag indicating whether it is the step to enter the workspace calculation from tiling sinking
    uint32_t fromTilingSink = 0;
    bool hasKeyAntiquantScale = 0;
    bool hasValueAntiquantScale = 0;
    uint32_t isMsd = 0;
    const int64_t *keyAntiquantMode = nullptr;
    const int64_t *valueAntiquantMode = nullptr;
    bool hasKeyAntiquantOffset = 0;
};

} // namespace optiling

#endif // BLOCK_SPARSE_ATTENTION_TILING_CONTEXT_H