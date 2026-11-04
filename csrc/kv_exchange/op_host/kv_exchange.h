/*!
 * \file kv_exchange.h
 * \brief MF+AIV kv_split transfer host-side declarations
 *
 * MLA HiCache L2<->L1 IO between a host page_first_kv_split pool
 * (pages, layers, page_size, 1, width) and a device layer_first_kv_split
 * pool (layers, pages, page_size, 1, width). The Python layer hands over
 * paired component views (layer offsets already folded by slicing) with
 * their MF DVAs; this adapter builds the 44-slot kv_exchange meta with a
 * zero-based layer range -- the same component/pitch/width contract the
 * AICPU build_kv_sparse_metadata path produces -- and launches the AIV
 * device kernel, so no meta handling remains on the framework side.
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
// production entry below builds the meta from plain tensors.
HOST_API void kv_exchange(const at::Tensor &meta, int64_t block_dim);

// Submit one transfer described by the AICPU-style component list.
// device_components[i] / host_components[i] are paired component views
// (device: [L,P,T,H,D]; host: PF-shaped metadata view [P,L,T,H,D] whose
// strides preserve the physical page/layer layout); host_dvas[i] is the MF
// device-visible address of host_components[i]'s base. direction is 1 (H2D)
// or 2 (D2H); host_layout_mode is 0 (page_first) or 1 (layer_first).
//
// Returns the pinned host staging tensor that carries the meta upload; the
// caller must keep it alive until the submission's completion event (the
// non_blocking H2D copy may still be in flight after this call returns).
HOST_API at::Tensor kv_exchange_submit(const at::Tensor &device_indices, const at::Tensor &host_indices,
                                       at::TensorList device_components, at::TensorList host_components,
                                       const std::vector<int64_t> &host_dvas, int64_t page_size, int64_t direction,
                                       int64_t host_layout_mode);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_KV_EXCHANGE_HOST_H_
