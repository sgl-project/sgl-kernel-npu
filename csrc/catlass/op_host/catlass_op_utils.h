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

namespace sglang {
namespace npu_kernel {

    torch::Dtype TypeStrToTorchDtype(const std::string &typeStr);

    aclDataType TorchDtypeToAclDtype(torch::Dtype torchDtype);

    torch::Dtype AclDtypeToTorchDtype(aclDataType aclDtype);

    aclDataType TypeStrToAclDtype(const std::string &typeStr);

    torch::Tensor GetOutputTensor(const std::vector<int64_t> &shape, torch::Dtype dtype);

}
}