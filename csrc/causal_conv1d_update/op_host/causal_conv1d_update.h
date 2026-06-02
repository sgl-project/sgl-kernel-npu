/*!
 * \file causal_conv1d_update.h
 * \brief causal_conv1d_update host-side function declaration
 * Adapted from: https://github.com/vllm-project/vllm-ascend/tree/main/csrc
                 https://gitcode.com/cann/ops-transformer/tree/master/attention
 */

#ifndef CAUSAL_CONV1D_UPDATE_HOST_H_
#define CAUSAL_CONV1D_UPDATE_HOST_H_

#include <ATen/ATen.h>
#include "defines.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor causal_conv1d_update_impl(const at::Tensor &x, const at::Tensor &weight,
                                              const at::Tensor &conv_state, const at::Tensor &conv_state_indices,
                                              const at::Tensor &bias, const at::Tensor &num_accepted_tokens,
                                              const at::Tensor &query_start_loc, bool activation_mode,
                                              int64_t pad_slot_id);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_UPDATE_HOST_H_
