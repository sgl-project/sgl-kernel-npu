// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "acl/acl.h"
#include "defines.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

enum TransferDirection : int64_t {
    H2D = 1,
    D2H = 2,
};

// Copy raw bytes between host and device using aclrtMemcpyAsync.
//
// This kernel performs a flat 1D byte copy — it does NOT interpret tensor
// layout (ND vs NZ). The caller must ensure that src and dst have the same
// byte size and that the data layout is compatible.
//
// Key use case: MoE weight DRAM offload.
//   - D2H: Copy NZ-format weight bytes from HBM to Host DRAM.
//   - H2D: Copy NZ-format weight bytes from Host DRAM to pre-allocated
//          NZ-format HBM buffer.
//   Because the copy is layout-agnostic, NZ bytes are preserved as-is,
//   eliminating the need for npu_format_cast at forward time.
//
// @dst:        destination tensor (device for H2D, host for D2H)
// @src:        source tensor (host for H2D, device for D2H)
// @direction:  1=H2D (host→device), 2=D2H (device→host)
HOST_API void transfer_weight(at::Tensor &dst, at::Tensor &src, int64_t direction)
{
    TORCH_CHECK(dst.nbytes() == src.nbytes(),
                "transfer_weight: size mismatch: dst.nbytes()=", dst.nbytes(),
                " vs src.nbytes()=", src.nbytes());
    TORCH_CHECK(direction == static_cast<int64_t>(TransferDirection::H2D) ||
                direction == static_cast<int64_t>(TransferDirection::D2H),
                "transfer_weight: direction must be 1(H2D) or 2(D2H), got ", direction);

    void *dst_ptr = dst.data_ptr();
    void *src_ptr = src.data_ptr();
    const size_t count = static_cast<size_t>(dst.nbytes());

    c10_npu::NPUStream current_stream = c10_npu::getCurrentNPUStream();
    aclrtStream acl_stream = current_stream.stream();

    aclrtMemcpyKind kind;
    if (direction == static_cast<int64_t>(TransferDirection::D2H)) {
        kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
    } else {
        kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
    }

    aclError ret = aclrtMemcpyAsync(dst_ptr, count, src_ptr, count, kind, acl_stream);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "transfer_weight: aclrtMemcpyAsync failed, error code: ", ret);
}

}  // namespace npu_kernel
}  // namespace sglang
