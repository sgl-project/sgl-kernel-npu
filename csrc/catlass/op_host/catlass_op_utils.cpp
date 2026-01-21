// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "catlass_op_utils.h"

namespace sglang {
namespace npu_kernel {

torch::Dtype TypeStrToTorchDtype(const std::string &typeStr)
{
    static const std::unordered_map<std::string, torch::Dtype> mapper = {{"float32", torch::kFloat32},
                                                                         {"float16", torch::kFloat16},
                                                                         {"int8", torch::kInt8},
                                                                         {"int32", torch::kInt32},
                                                                         {"bf16", torch::kBFloat16}};
    auto iter = mapper.find(typeStr);
    return iter != mapper.end() ? iter->second : torch::kFloat16;
}

aclDataType TorchDtypeToAclDtype(const torch::Dtype torchDtype)
{
    static const std::unordered_map<torch::Dtype, aclDataType> mapper = {{torch::kFloat32, ACL_FLOAT},
                                                                         {torch::kFloat16, ACL_FLOAT16},
                                                                         {torch::kInt8, ACL_INT8},
                                                                         {torch::kInt32, ACL_INT32},
                                                                         {torch::kBFloat16, ACL_BF16}};
    auto iter = mapper.find(torchDtype);
    return iter != mapper.end() ? iter->second : ACL_FLOAT16;
};

torch::Dtype AclDtypeToTorchDtype(const aclDataType aclDtype)
{
    static const std::map<aclDataType, torch::Dtype> mapper = {{ACL_FLOAT16, torch::kFloat16},
                                                               {ACL_FLOAT, torch::kFloat32},
                                                               {ACL_INT32, torch::kInt32},
                                                               {ACL_INT8, torch::kInt8},
                                                               {ACL_BF16, torch::kBFloat16}};
    auto iter = mapper.find(aclDtype);
    return iter != mapper.end() ? iter->second : torch::kFloat16;
};

aclDataType TypeStrToAclDtype(const std::string &typeStr)
{
    return TorchDtypeToAclDtype(TypeStrToTorchDtype(typeStr));
};

torch::Tensor GetOutputTensor(const std::vector<int64_t> &shape, const torch::Dtype dtype)
{
    at::TensorOptions options = at::TensorOptions();
    options =
        options.dtype(dtype).layout(at::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());
    return at_npu::native::empty_with_format(shape, options, ACL_FORMAT_ND);
};
}  // namespace npu_kernel
}  // namespace sglang
