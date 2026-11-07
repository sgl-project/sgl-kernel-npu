/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Provenance: csrc/attentions/csrc/ops/laser_attention/op_host/laser_attention_tiling.h
// The shared tiling layout lives in op_kernel/laser_attention_tiling_data.h.

#ifndef LASER_ATTENTION_H_
#define LASER_ATTENTION_H_

#include <cstdint>
#include <cstddef>

#include "../op_kernel/laser_attention_tiling_data.h"

namespace optiling {

struct SglLaserAttentionCompileInfo {
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t l1Size;
    uint64_t ubSize;
    uint32_t maxAicCoresNum;
    uint32_t maxAivCoresNum;
    size_t defaultSysWorkspaceSize;
};

}  // namespace optiling
#endif  // LASER_ATTENTION_H_
