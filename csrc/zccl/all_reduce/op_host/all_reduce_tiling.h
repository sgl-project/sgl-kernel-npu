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
		uint32_t formerNum = 0, uint32_t formerLength = 0, uint32_t tailNum = 0, uint32_t tailLength = 0
    ): formerNum_(formerNum), formerLength_(formerLength), tailNum_(tailNum), tailLength_(tailLength) {}
	uint32_t formerNum_;
    uint32_t formerLength_;
    uint64_t tailLength_;
    uint32_t tailNum_;
	uint32_t coreNumPerRank_;
	uint32_t eleNumPerRank_;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // ALL_REDUCE_TILING_H