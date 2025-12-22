// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <map>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <ATen/ATen.h>
#include <torch/library.h>

#include "tiling/platform/platform_ascendc.h"
#include "defines.h"
#include "torch_helper.h"
#include "catlass_op_utils.h"
#include "aclrtlaunch_catlass_fp8w8a16_gmm_bfloat16_t.h"

namespace sglang {
namespace npu_kernel {
HOST_API at::Tensor fp8_w8a16_grouped_matmul(const at::Tensor &mat1, const at::Tensor &mat2, const at::Tensor &scale,
                                             const at::Tensor &groupList, const std::string &outDType)
{
    at::ScalarType scalar_type = mat1.scalar_type();

    TORCH_CHECK(scalar_type == at::kBFloat16, "only support bf16");
    TORCH_CHECK(mat1.dim() == 2, "x should be [M, K]");
    TORCH_CHECK(mat2.dim() == 3, "weight should be [g, k, n]");

    uint32_t m = mat1.size(0);
    uint32_t k = mat1.size(1);
    uint32_t n = mat2.size(2);
    uint32_t g = mat2.size(0);

    void *x_ptr = mat1.data_ptr();
    void *w_ptr = mat2.data_ptr();
    void *scale_ptr = scale.data_ptr();
    void *group_list_ptr = groupList.data_ptr();

    auto outputDataType = TypeStrToAclDtype(outDType);
    at::Tensor output = GetOutputTensor({m, n}, AclDtypeToTorchDtype(outputDataType));
    void *y_ptr = output.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint32_t workspace_size = 4 * 256 * 256 * sizeof(outputDataType) * aicCoreNum;
    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(mat1.options().device()));
    void *workspace_ptr = workspace_tensor.data_ptr();

    at_npu::native::OpCommand cmd;
    cmd.Name("catlass_fp8w8a16_gmm_bfloat16_t");
    cmd.SetCustomHandler(
        [aicCoreNum, stream, x_ptr, w_ptr, scale_ptr, group_list_ptr, y_ptr, workspace_ptr, g, m, n, k]() -> int {
            int device_id = 0;
            int64_t aiv_num = 0;
            TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);

            ACLRT_LAUNCH_KERNEL(catlass_fp8w8a16_gmm_bfloat16_t)
            (aicCoreNum, stream, x_ptr, w_ptr, scale_ptr, group_list_ptr, y_ptr, workspace_ptr, g, m, n, k);
            return 0;
        });
    cmd.Run();

    return output;
}

} // namespace npu_kernel
} // namespace sglang
