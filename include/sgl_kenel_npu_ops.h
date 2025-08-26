// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_OPS_H
#define SGL_KERNEL_NPU_OPS_H

namespace sglang {
namespace npu_kernel {
at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y);

at::Tensor cache_loc_assign(const at::Tensor &token_pool, const at::Tensor &start_offset, const at::Tensor &end_offset,
    const at::Tensor &out_cache_loc, const at::Tensor &out_cache_loc_idx);

bool assign_cache_op(at::Tensor &dst_tensor, const at::Tensor &src_tensor, const at::Tensor &dst_start_idx,
    const at::Tensor &dst_end_idx, const at::Tensor &src_start_idx, const at::Tensor &src_end_idx);
}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_OPS_H
