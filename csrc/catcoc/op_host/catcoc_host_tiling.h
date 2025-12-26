// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KERNEL_CATCOC_HOST_TILING_H
#define KERNEL_CATCOC_HOST_TILING_H

#include <cstdint>

namespace sglang {
namespace npu_kernel {

typedef enum { WEIGHT_ND = 0, WEIGHT_NZ = 1 } WeightFormatMode;

typedef enum { BF16 = 0, FP16 = 1, FP32 = 2 } DataFormatMode;

struct KernelCATCOCHostTilingData {
    uint32_t m;  // get from matmul M
    uint32_t n;  // get from matmul N
    uint32_t k;  // get from matmul K

    uint32_t m0 = 128;
    uint32_t k0 = 256;
    uint32_t n0 = 256;
    uint32_t swizzleDirect = 1;
    uint32_t swizzleOffset = 7;
    uint32_t ubMoveNum = 16 * 1024;
    uint32_t pValue = 3;
    uint32_t commNpuSplit = 2;
    uint32_t commDataSplit = 1;
    uint32_t lenPerLoop = 128 * 256 / 2;

    int64_t weight_format_mode = WEIGHT_ND;
    int64_t data_format_mode = BF16;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // KERNEL_CATCOC_HOST_TILING_H
