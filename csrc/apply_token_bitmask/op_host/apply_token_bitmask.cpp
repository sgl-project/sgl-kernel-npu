// BSD 3-Clause License
// Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_apply_token_bitmask_fp32.h"
#include "aclrtlaunch_apply_token_bitmask_fp16.h"
#include "aclrtlaunch_apply_token_bitmask_bf16.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang {
namespace npu_kernel
{

HOST_API at::Tensor apply_token_bitmask(
    at::Tensor logits,
    at::Tensor bitmask,
    c10::optional<at::Tensor> indices) {

    // Input validation
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D, got ", logits.dim(), "D tensor");
    TORCH_CHECK(bitmask.dim() == 2, "bitmask must be 2D, got ", bitmask.dim(), "D tensor");
    TORCH_CHECK(logits.size(0) == bitmask.size(0),
                "logits and bitmask batch size mismatch: ", logits.size(0), " vs ", bitmask.size(0));
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(bitmask.is_contiguous(), "bitmask must be contiguous");
    TORCH_CHECK(bitmask.scalar_type() == at::kInt,
                "bitmask must be int32, got ", bitmask.scalar_type());

    at::ScalarType dtype = logits.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
                "logits must be float32, float16 or bfloat16, got ", dtype);

    int64_t batch = logits.size(0);
    int64_t vocabSize = logits.size(1);
    int64_t bitmaskWidth = bitmask.size(1);

    TORCH_CHECK(vocabSize > 0, "vocab_size must be > 0");

    // Handle indices
    bool hasIndices = indices.has_value() && indices->defined() && indices->numel() > 0;
    at::Tensor rowIndices;
    at::Tensor selectedBitmask;

    if (hasIndices) {
        rowIndices = indices.value().to(at::kLong).contiguous();
        TORCH_CHECK(rowIndices.dim() == 1, "indices must be 1D");
        selectedBitmask = bitmask.index({rowIndices}).contiguous();
    } else {
        selectedBitmask = bitmask;
    }

    int64_t numIndices = hasIndices ? rowIndices.size(0) : batch;
    if (numIndices == 0) return logits;

    // --- Alignment ---
    // Use 256 elements as alignment unit to satisfy both:
    //   logits DataCopy: 256 * sizeof(T) is a multiple of 32 bytes
    //   bitmask DataCopy: 256/32 = 8 int32 words = 32 bytes (DataCopy alignment for int32)
    // Most LLM vocab sizes (32000, 128256) are already multiples of 256.
    constexpr int64_t ALIGN_UNIT = 256;
    int64_t paddedVocabSize = ((vocabSize + ALIGN_UNIT - 1) / ALIGN_UNIT) * ALIGN_UNIT;
    bool needsPadding = (paddedVocabSize != vocabSize);

    // Prepare working logits [numIndices, paddedVocabSize]
    at::Tensor workingLogits;
    at::Tensor selectedLogits;
    if (hasIndices) {
        selectedLogits = logits.index({rowIndices});
        if (!selectedLogits.is_contiguous()) selectedLogits = selectedLogits.contiguous();
    } else {
        selectedLogits = logits;
    }

    if (needsPadding) {
        workingLogits = at::zeros({numIndices, paddedVocabSize}, selectedLogits.options());
        workingLogits.narrow(1, 0, vocabSize).copy_(selectedLogits);
    } else {
        workingLogits = selectedLogits;
    }

    // Prepare working bitmask [numIndices, paddedBitmaskWidth]
    int64_t paddedBitmaskWidth = (paddedVocabSize + 31) / 32;
    at::Tensor workingBitmask;
    if (static_cast<int64_t>(selectedBitmask.size(1)) >= paddedBitmaskWidth) {
        workingBitmask = selectedBitmask;
    } else {
        workingBitmask = at::zeros({numIndices, paddedBitmaskWidth},
                                   selectedBitmask.options());
        workingBitmask.narrow(1, 0, selectedBitmask.size(1)).copy_(selectedBitmask);
    }

    // Ensure working tensors are contiguous
    if (!workingLogits.is_contiguous()) workingLogits = workingLogits.contiguous();
    if (!workingBitmask.is_contiguous()) workingBitmask = workingBitmask.contiguous();

    // --- Tiling ---
    int64_t dtypeSize = static_cast<int64_t>(workingLogits.element_size());

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();

    // Block dimension: one block per row, capped at coreNum
    uint32_t blockDim = static_cast<uint32_t>(std::min(static_cast<int64_t>(numIndices), coreNum));
    if (blockDim == 0) blockDim = 1;

    uint32_t rowsPerCore = static_cast<uint32_t>(numIndices) / blockDim;
    uint32_t rowsLastCore = static_cast<uint32_t>(numIndices) - rowsPerCore * (blockDim - 1);

    // Compute tileLength from UB size
    // UB per tile (double buffered, BUFFER_NUM=2):
    //   logitsQueue:  2 * tileLength * sizeof(T)
    //   bitmaskQueue: 2 * (tileLength/32) * sizeof(int32_t)
    //   outQueue:     2 * tileLength * sizeof(T)
    // Calculate at ALIGN_UNIT granularity to avoid integer division truncation
    constexpr int32_t hostBufferNum = 2;
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // Subtract 16KB for pipe overhead
    uint64_t usableUb = (ubSize > 16384) ? (ubSize - 16384) : ubSize;

    int64_t bytesPerUnit = static_cast<int64_t>(hostBufferNum) *
        (2 * ALIGN_UNIT * dtypeSize + (ALIGN_UNIT / 32) * static_cast<int64_t>(sizeof(int32_t)));
    uint32_t tileLength = static_cast<uint32_t>(
        (usableUb / static_cast<uint64_t>(bytesPerUnit)) * static_cast<uint64_t>(ALIGN_UNIT));

    // Cap tileLength to paddedVocabSize
    if (tileLength > static_cast<uint32_t>(paddedVocabSize)) {
        tileLength = static_cast<uint32_t>(paddedVocabSize);
    }

    // Safety floor
    if (tileLength < static_cast<uint32_t>(ALIGN_UNIT)) {
        tileLength = static_cast<uint32_t>(ALIGN_UNIT);
    }

    // Prevent NPU storage from being reclaimed before async kernel completes
    auto npuStream = c10_npu::getCurrentNPUStream();
    workingLogits.record_stream(npuStream);
    workingBitmask.record_stream(npuStream);

    // Prepare kernel arguments (ConvertTypes requires lvalue references)
    uint32_t numRowsU32 = static_cast<uint32_t>(numIndices);
    uint32_t vocabSizeU32 = static_cast<uint32_t>(paddedVocabSize);
    uint32_t logitsStrideU32 = static_cast<uint32_t>(paddedVocabSize);
    uint32_t bitmaskStrideU32 = static_cast<uint32_t>(workingBitmask.size(1));
    uint32_t dtypeSizeU32 = static_cast<uint32_t>(dtypeSize);

    // Launch kernel
    if (dtype == at::kFloat) {
        EXEC_KERNEL_CMD(apply_token_bitmask_fp32, blockDim,
                        workingLogits, workingBitmask,
                        numRowsU32, vocabSizeU32, logitsStrideU32, bitmaskStrideU32,
                        rowsPerCore, rowsLastCore,
                        tileLength, blockDim, dtypeSizeU32);
    } else if (dtype == at::kHalf) {
        EXEC_KERNEL_CMD(apply_token_bitmask_fp16, blockDim,
                        workingLogits, workingBitmask,
                        numRowsU32, vocabSizeU32, logitsStrideU32, bitmaskStrideU32,
                        rowsPerCore, rowsLastCore,
                        tileLength, blockDim, dtypeSizeU32);
    } else {
        EXEC_KERNEL_CMD(apply_token_bitmask_bf16, blockDim,
                        workingLogits, workingBitmask,
                        numRowsU32, vocabSizeU32, logitsStrideU32, bitmaskStrideU32,
                        rowsPerCore, rowsLastCore,
                        tileLength, blockDim, dtypeSizeU32);
    }

    // Copy results back
    auto result = needsPadding ? workingLogits.narrow(1, 0, vocabSize) : workingLogits;
    if (hasIndices) {
        logits.index_put_({rowIndices}, result);
    } else if (needsPadding) {
        logits.copy_(result);
    }

    return logits;
}

}  // namespace npu_kernel
}  // namespace sglang
