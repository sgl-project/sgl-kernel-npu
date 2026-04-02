/*!
 * \file causal_conv1d.h
 * \brief causal_conv1d host-side function declaration
 * Adapted from: https://github.com/vllm-project/vllm-ascend/tree/main/csrc/causal_conv1d
                 https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_causal_conv1d
 */

#ifndef CAUSAL_CONV1D_HOST_H_
#define CAUSAL_CONV1D_HOST_H_

#include <ATen/ATen.h>

#include "defines.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor causal_conv1d_impl(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &conv_states,
                                       const at::Tensor &query_start_loc, const at::Tensor &cache_indices,
                                       const at::Tensor &has_initial_state, const at::Tensor &bias,
                                       bool activation_mode, int64_t pad_slot_id);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_HOST_H_
