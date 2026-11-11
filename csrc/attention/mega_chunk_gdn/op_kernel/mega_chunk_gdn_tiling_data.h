// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Framework adaptation of csrc/mega_chunk_gdn; PTO computation is unchanged.

#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace sgl_mega_gdn {
struct alignas(8) TilingData {
    int64_t has_initial_state;
    int64_t batch_size;
    int64_t seq_len;
    int64_t total_tokens;
    uint32_t num_matrices;
    uint32_t num_heads;
    uint32_t num_key_heads;
};
}  // namespace sgl_mega_gdn
