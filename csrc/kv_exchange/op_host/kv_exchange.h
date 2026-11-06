/*!
 * \file kv_exchange.h
 * \brief MF+AIV kv_split transfer host-side declarations
 *
 * MLA HiCache L2<->L1 IO between a host page_first_kv_split pool
 * (pages, layers, page_size, 1, width) and a device layer_first_kv_split
 * pool (layers, pages, page_size, 1, width). The Python layer hands over
 * paired component views (layer offsets already folded by slicing) with
 * their MF DVAs; this adapter packs the same component/pitch/width/layer
 * description into the kernel launch parameter block and launches the AIV
 * device kernel, so no meta staging, upload or device-resident buffer
 * remains on the submission path.
 */

#ifndef SGL_KERNEL_NPU_KV_EXCHANGE_HOST_H_
#define SGL_KERNEL_NPU_KV_EXCHANGE_HOST_H_

#include <ATen/ATen.h>
#include <vector>

#include "defines.h"

namespace sglang {
namespace npu_kernel {

// Low-level launch: `meta` is a device-resident int64 tensor laid out per
// op_kernel/kv_exchange_kernel.cpp. Exposed for tests and debugging; the
// production entry below packs the same description into launch parameters.
HOST_API void kv_exchange(const at::Tensor &meta, int64_t block_dim);

// Submit one transfer described by the split-KV component list.
// device_components[i] / host_components[i] are paired component views
// (device: [L,P,T,H,D]; host: PF-shaped metadata view [P,L,T,H,D] whose
// strides preserve the physical page/layer layout); host_dvas[i] is the MF
// device-visible address of host_components[i]'s base. direction is 1 (H2D)
// or 2 (D2H); host_layout_mode is 0 (page_first) or 1 (layer_first).
//
// The transfer description travels inside the kernel launch parameter
// block, so nothing outlives the call on the host side.
HOST_API void kv_exchange_submit(const at::Tensor &device_indices, const at::Tensor &host_indices,
                                 at::TensorList device_components, at::TensorList host_components,
                                 const std::vector<int64_t> &host_dvas, int64_t page_size, int64_t direction,
                                 int64_t host_layout_mode);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_KV_EXCHANGE_HOST_H_
