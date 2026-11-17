// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>

namespace optiling {

/*
 * Tiling data shared by the Host tiling function and the Kernel entry.
 *
 * Everything the Kernel needs is resolved on the Host and passed through this
 * structure: the layout is fixed at compile time and the Kernel never queries
 * the platform itself.
 */
struct SglEluTilingData {
    uint32_t totalLength;  // elements in the flattened input
    uint32_t tileLength;   // elements per tile, a multiple of the 32B vector
                           // unit for the selected dtype
    uint32_t blockDim;     // AIV cores that participate in this launch
    uint32_t baseLength;   // elements owned by every participating core
    uint32_t extraBlocks;  // the first extraBlocks cores own one extra element
    float alpha;           // coefficient of the negative branch
};

static_assert(sizeof(SglEluTilingData) == 6 * sizeof(uint32_t),
              "Host and Kernel must share the six-field tiling layout");

}  // namespace optiling
