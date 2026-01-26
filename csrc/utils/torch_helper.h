// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_TORCH_NPU_HELPER_H
#define SGL_KERNEL_NPU_TORCH_NPU_HELPER_H

#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "common_tiling.h"

namespace sglang {
namespace npu_kernel {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

class TorchNpuHelper
{
public:
    inline static at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
    {
        at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
        int deviceIndex = 0;
        c10_npu::GetDevice(&deviceIndex);
        return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
    }

    inline static at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
    {
        return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
    }

    inline static void *ConvertType(const at::Tensor &at_tensor)
    {
        return const_cast<void *>(at_tensor.data_ptr());
    }

    inline static host_utils::DataType ConvertDataType(const at::ScalarType type)
    {
        switch (type) {
            case at::ScalarType::Float:
                return host_utils::DataType::DT_FLOAT;

            case at::ScalarType::Half:
                return host_utils::DataType::DT_FLOAT16;

            case at::ScalarType::BFloat16:
                return host_utils::DataType::DT_BFLOAT16;

            case at::ScalarType::Double:
                return host_utils::DataType::DT_DOUBLE;

            case at::ScalarType::Bool:
                return host_utils::DataType::DT_BOOL;

            case at::ScalarType::Char:
                return host_utils::DataType::DT_INT8;

            case at::ScalarType::Short:
                return host_utils::DataType::DT_INT16;

            case at::ScalarType::Int:
                return host_utils::DataType::DT_INT32;

            case at::ScalarType::Long:
                return host_utils::DataType::DT_INT64;

            case at::ScalarType::Byte:
                return host_utils::DataType::DT_UINT8;

            case at::ScalarType::ComplexFloat:
                return host_utils::DataType::DT_COMPLEX64;

            case at::ScalarType::ComplexDouble:
                return host_utils::DataType::DT_COMPLEX128;
        }

        return host_utils::DataType::DT_MAX;
    }

    template <typename T>
    inline static T ConvertType(T value)
    {
        return value;
    }

    template <typename... Ts>
    inline static constexpr auto ConvertTypes(Ts &...args)
    {
        return std::make_tuple(ConvertType(args)...);
    }
};
}  // namespace npu_kernel
}  // namespace sglang

/**
 * @brief Launch real kernel function on NPU
 *
 * @param kernel_name      [in] name of kernel
 * @param blockdim         [in] dim size of block
 */
#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                                            \
    do {                                                                                       \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                        \
        auto converted_params = sglang::npu_kernel::TorchNpuHelper::ConvertTypes(__VA_ARGS__); \
        auto acl_call = [acl_stream, blockdim, converted_params]() -> int {                    \
            std::apply(                                                                        \
                [&](auto &&...params) {                                                        \
                    ACLRT_LAUNCH_KERNEL(kernel_name)                                           \
                    (blockdim, acl_stream, params...);                                         \
                },                                                                             \
                converted_params);                                                             \
            return 0;                                                                          \
        };                                                                                     \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);                           \
    } while (false)

#endif  // SGL_KERNEL_NPU_TORCH_NPU_HELPER_H
