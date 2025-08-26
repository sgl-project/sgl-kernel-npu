// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "version.h"

#include "torch_helper.h"
#include "sgl_kenel_npu_ops.h"

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sgl_kernel_npu_print_version() -> ()", []() { printf("%s\n", LIB_VERSION_FULL); });
    m.def("sgl_kernel_npu_version() -> str", []() { return std::string("") + LIB_VERSION; });

    m.def("helloworld(Tensor x, Tensor y) -> Tensor");

    m.def("cache_loc_assign(Tensor req_indices, Tensor token_pool, Tensor start_offset, Tensor end_offset, Tensor "
          "out_cache_loc) -> Tensor");

    m.def("assign_cache_op(Tensor! out, Tensor src, Tensor dst_start_idx, Tensor dst_end_idx, Tensor src_start_idx, "
            "Tensor src_end_idx) -> bool");

    m.def("mla_preprocess(Tensor hiddenState, Tensor gamma0, Tensor beta0, Tensor wdqkv, "
                          "Tensor descale0, Tensor gamma1, Tensor beta1, Tensor wuq, "
                          "Tensor descale1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk,"
                          "Tensor kv_cache, Tensor kv_cache_rope, Tensor slotmapping, "
                          "Tensor quant_scale0, Tensor quant_offset0, Tensor bias0, "
                          "Tensor quant_scale1, Tensor quant_offset1, Tensor bias1, *, "
                          "Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, "
                          "str? cache_mode=None, str? quant_mode=None, "
                          "Tensor(a!) q_out0, Tensor(b!) kv_cache_out0, Tensor(c!) q_out1, Tensor(d!) kv_cache_out1) "
                          "-> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");

}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("helloworld", TORCH_FN(sglang::npu_kernel::helloworld));

    m.impl("cache_loc_assign", TORCH_FN(sglang::npu_kernel::cache_loc_assign));

    m.impl("assign_cache_op", TORCH_FN(sglang::npu_kernel::RunCustomAssign));
    m.impl("mla_preprocess", TORCH_FN(sglang::npu_kernel::mla_preprocess));
}
}  // namespace