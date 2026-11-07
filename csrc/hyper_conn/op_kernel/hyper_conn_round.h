// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_HYPER_CONN_ROUND_H
#define SGL_KERNEL_NPU_HYPER_CONN_ROUND_H

#include "kernel_operator.h"
#include "hyper_conn_limits.h"

namespace sglang {
namespace npu_kernel {
namespace hyper_conn {

// One fp32 UB block holds 8 lanes; the per-row gate pipeline uses one block.
constexpr uint32_t GATE_LANES = 8;
constexpr uint32_t BF16_SIZE = sizeof(bfloat16_t);
constexpr uint32_t F32_SIZE = sizeof(float);
// UB copies are issued in 32-byte units.
constexpr uint32_t UB_BLOCK = 32;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

/* k * sigmoid(x) = k / (1 + exp(-x)).
 *
 * `tmp` and `numerator` must be distinct from `dst`/`src` and at least `count`
 * elements long; `numerator` must already hold `k` (1.0f for sigmoid(x), 2.0f
 * for 2 * sigmoid(x), etc.). Both tails saturate correctly: exp(-x)
 * overflowing to inf gives 0, underflowing to 0 gives k.
 */
__aicore__ inline void SigmoidF32(const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &src,
                                  const AscendC::LocalTensor<float> &tmp, const AscendC::LocalTensor<float> &numerator,
                                  uint32_t count)
{
    AscendC::Muls(tmp, src, -1.0f, count);
    AscendC::Exp(tmp, tmp, count);
    AscendC::Adds(tmp, tmp, 1.0f, count);
    AscendC::Div(dst, numerator, tmp, count);
}

}  // namespace hyper_conn
}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_HYPER_CONN_ROUND_H
