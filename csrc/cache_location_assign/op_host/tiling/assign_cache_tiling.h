// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CACHE_LOC_ASSIGN_TILING_H
#define CACHE_LOC_ASSIGN_TILING_H

#include <cstdint>

struct AssignCacheTillingData {
    uint64_t vcoreNum{0};
    uint64_t workspaceSize{0};
    uint64_t rowNumNoTail{0};
    uint64_t tailNum{0};
    uint64_t rowSize{0};
    uint64_t batchSize{0};
    uint64_t maxStep{0};

    uint64_t tokenColAlignInt32{0};
    uint64_t cacheLocSize{0};
    uint64_t cacheLocAlignIn32{0};
    uint64_t cacheLocIdxSize{0};
    uint64_t cacheLocIdxAlignIn32{0};

    uint64_t tokenCountAlignInt32{0};
    uint64_t cacheLocCountAlignIn32{0};
    uint64_t cacheLocIdxCountAlignIn32{0};
};

#endif  // CACHE_LOC_ASSIGN_TILING_H