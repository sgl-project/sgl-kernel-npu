// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef REDUCE_SCATTER_TILING_H
#define REDUCE_SCATTER_TILING_H

#include <cstdint>

namespace sglang {
namespace zccl {

struct ReduceScatterTilingData {
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t formerLength;
    uint64_t tailLength;
};

}  // namespace zccl
}  // namespace sglang

#endif  // REDUCE_SCATTER_TILING_H
