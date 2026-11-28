// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_UTILS_DTYPE_H
#define SGL_KERNEL_NPU_UTILS_DTYPE_H

#include <cstdint>

namespace sgl {
namespace kernel_npu {
namespace utils {

// Data types understood by the native kernels. The numeric values are a stable
// ABI between the Torch adapter, the host launcher and the device kernel, so
// keep them dense: append new entries, bump kDTypeCount, and do not renumber
// values that have shipped.
enum class DTYPE : uint8_t {
    FP32 = 0,  // 32-bit floating point
    FP16 = 1,  // IEEE 16-bit floating point
    BF16 = 2,  // bfloat16
};

// Support measured on the target device (dav-2201, CANN 9.2.1, bisheng 15.0.5):
//   FP32: native storage and native vector arithmetic (vexp/vrec/vmul accept
//         f32).
//   FP16: native storage, and `half` is accepted by the vector arithmetic
//         (vexp/vrec/vmul); kernels may still widen to f32 for accuracy.
//   BF16: native storage, but vector arithmetic is not available: the CCE unary
//         and binary intrinsics reject bfloat16_t ("the 1st parameter maybe
//         need a type '__ubuf__ half *'"), so BF16 data must be widened to f32
//         and narrowed back with RNE.
//
// Number of declared values; keep in sync with DTYPE when appending.
inline constexpr uint8_t kDTypeCount = 3;

// Stable integer code for boundaries that cannot carry a C++ enum (ABI
// tables, logging, kernel metadata).
constexpr uint8_t dtype_to_code(DTYPE dtype)
{
    return static_cast<uint8_t>(dtype);
}

constexpr bool is_valid_dtype_code(uint8_t code)
{
    return code < kDTypeCount;
}

// 16-bit storage types share the packed two-elements-per-32-bit-word layout.
constexpr bool is_16bit_dtype(DTYPE dtype)
{
    return dtype == DTYPE::FP16 || dtype == DTYPE::BF16;
}

}  // namespace utils
}  // namespace kernel_npu
}  // namespace sgl

#endif  // SGL_KERNEL_NPU_UTILS_DTYPE_H
