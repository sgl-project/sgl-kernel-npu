// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ALL_REDUCE_TILING_H
#define ALL_REDUCE_TILING_H

#include <cstdint>
namespace sglang {
namespace zccl {

struct AllReduceTilingData {
    // 构造函数（带默认参数）
    AllReduceTilingData(
        uint32_t inputNumPerCore = 0, uint32_t inputLastNumCore = 0, uint32_t outputCorePerRank = 0,
        uint32_t outputNumPerCore = 0, uint32_t outputLastNumCore = 0
    ) : inputNumPerCore_(inputNumPerCore), inputLastNumCore_(inputLastNumCore),
        outputNumPerCore_(outputNumPerCore), outputCorePerRank_(outputCorePerRank),
        outputLastNumCore_(outputLastNumCore)

    {}

    uint32_t inputNumPerCore_;
    uint32_t inputLastNumCore_;
    uint32_t outputNumPerCore_;
    uint32_t outputCorePerRank_;
    uint32_t outputLastNumCore_;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // ALL_REDUCE_TILING_H