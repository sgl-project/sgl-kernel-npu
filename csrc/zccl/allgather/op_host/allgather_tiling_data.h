// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ALL_GATHER_TILING_H
#define ALL_GATHER_TILING_H

#include <cstdint>
namespace sglang {
namespace npu_kernel {

struct AllGatherTilingData {
    uint32_t input_num_per_core;
    uint32_t output_num_per_core;
    uint32_t input_last_num_core;
    uint32_t output_last_num_core;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // ALL_GATHER_TILING_H