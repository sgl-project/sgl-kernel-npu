#include <torch_npu/csrc/aten/common/from_blob.h>
#include <vector>

namespace sglang {
namespace npu_kernel {

at::Tensor weak_ref_tensor(const at::Tensor &tensor)
{
    if (!tensor.is_privateuseone()) {
        throw std::runtime_error("Tensor must be on NPU device");
    }
    // Get the raw data pointer
    void *data_ptr = tensor.data_ptr();
    // Get tensor sizes and strides
    std::vector<int64_t> sizes = tensor.sizes().vec();
    std::vector<int64_t> strides = tensor.strides().vec();
    // Get tensor options (dtype, device)
    auto options = tensor.options();
    // Create a new tensor from the raw data pointer
    auto new_tensor = at_npu::native::from_blob(data_ptr, sizes, strides, options);
    return new_tensor;
}

}  // namespace npu_kernel

}  // namespace sglang
