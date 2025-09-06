// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BUILD_TREE_TILING_H
#define BUILD_TREE_TILING_H

#include <cstdint>
namespace sglang {
namespace npu_kernel {

typedef enum { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 } TreeMaskMode;

struct BuildTreeTilingData {
    int64_t topk;
    int64_t depth;
    int64_t draft_token_num;
    int64_t tree_mask_mode;

    int32_t batch_size;
    int32_t mask_size;

    int32_t big_core_num;
    int32_t big_core_tile_num;
    int32_t small_core_tile_num;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // BUILD_TREE_TILING_H
