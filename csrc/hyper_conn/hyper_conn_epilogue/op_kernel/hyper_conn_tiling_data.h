// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_HYPER_CONN_TILING_DATA_H
#define SGL_HYPER_CONN_TILING_DATA_H

#include <cstdint>

namespace optiling {

// Shared host/kernel tiling layouts for the four hyper-connection epilogue
// operators. Field order and types must match byte-for-byte across Tiling and
// Kernel translation units.
struct SglHyperConnGroupedNormTilingData {
    uint32_t groups;
    uint32_t width;
    uint32_t weightLen;
    float eps;
    float invWidth;
    uint32_t reserved[3];
};

static_assert(sizeof(SglHyperConnGroupedNormTilingData) == 32, "SglHyperConnGroupedNormTilingData layout mismatch");

struct SglHyperConnScaledSiluTilingData {
    uint32_t rows;
    uint32_t cols;
    uint32_t srcStride;
    float invScale;
    uint32_t reserved[4];
};

static_assert(sizeof(SglHyperConnScaledSiluTilingData) == 32, "SglHyperConnScaledSiluTilingData layout mismatch");

struct SglHyperConnBranchMixTilingData {
    uint32_t rows;
    uint32_t hc;
    uint32_t width;
    uint32_t tile;
    float invHc;
    uint32_t reserved[3];
};

static_assert(sizeof(SglHyperConnBranchMixTilingData) == 32, "SglHyperConnBranchMixTilingData layout mismatch");

struct SglHyperConnBranchCombineTilingData {
    uint32_t rows;
    uint32_t hc;
    uint32_t width;
    uint32_t tile;
    uint32_t logitsStride;
    float invHc;
    uint32_t reserved[2];
};

static_assert(sizeof(SglHyperConnBranchCombineTilingData) == 32, "SglHyperConnBranchCombineTilingData layout mismatch");

}  // namespace optiling

#endif  // SGL_HYPER_CONN_TILING_DATA_H
