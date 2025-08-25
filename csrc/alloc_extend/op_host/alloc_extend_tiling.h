// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ALLOC_EXTENT_TILING_H
#define ALLOC_EXTENT_TILING_H

#include <cstdint>
namespace sglang {
namespace npu_kernel {

struct AllocExtendTilingData {
    int32_t batch_size;
    int32_t page_size;
    int32_t used_core_num;
    int64_t total_extend_tokens;
};

}
}

#endif  // ALLOC_EXTENT_TILING_H