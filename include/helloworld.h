// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_HELLOWORLD_H
#define SGL_KERNEL_NPU_HELLOWORLD_H

#include "aclrtlaunch_helloworld.h"

namespace sglang {
namespace npu_kernel {

at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y);
}
}

#endif  //SGL_KERNEL_NPU_HELLOWORLD_H
