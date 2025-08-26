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

at::Tensor cache_loc_assign(const at::Tensor &req_indices, const at::Tensor &token_pool,
    const at::Tensor &start_offset, const at::Tensor &end_offset, const at::Tensor &out_cache_loc);

bool RunCustomAssign(at::Tensor &dstTensor, const at::Tensor &srcTensor, const at::Tensor &dstStartIdx,
    const at::Tensor &dstEndIdx, const at::Tensor &srcStartIdx, const at::Tensor &srcEndIdx);

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> mla_preprocess(
    const at::Tensor &hiddenState, const at::Tensor &gamma0, const at::Tensor &beta0, const at::Tensor &wdqkv,
    const at::Tensor &descale0, const at::Tensor &gamma1, const at::Tensor &beta1,
    const at::Tensor &wuq, const at::Tensor &descale1, const at::Tensor &gamma2,
    const at::Tensor &cos, const at::Tensor &sin, const at::Tensor &wuk,
    const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const at::Tensor &quant_scale0, const at::Tensor &quant_offset0, const at::Tensor &bias0,
    const at::Tensor &quant_scale1, const at::Tensor &quant_offset1, const at::Tensor &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode, c10::optional<c10::string_view> quant_mode,
    at::Tensor &q_out0,
    at::Tensor &kv_cache_out0,
    at::Tensor &q_out1,
    at::Tensor &kv_cache_out1);

}  // namespace npu_kernel

}  // namespace sglang

#endif  // SGL_KERNEL_NPU_OPS_H
