// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KERNEL_CATLASS_MATMUL_TILING_H
#define KERNEL_CATLASS_MATMUL_TILING_H

#include <cstdint>

namespace sglang {
namespace npu_kernel {

typedef enum { WEIGHT_ND = 0, WEIGHT_NZ = 1 } WeightFormatMode;

typedef enum { BF16 = 0, FP16 = 1, FP32 = 2} DataFormatMode;

struct KernelCatlassMatmulTilingData {
    int32_t m;
    int32_t n;
    int32_t k;

    int64_t weight_format_mode = WEIGHT_ND;
    int64_t data_format_mode = BF16;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // KERNEL_CATLASS_MATMUL_TILING_H
