// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_helloworld.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y)
{
    /* create a result tensor */
    at::Tensor z = at::empty_like(x);

    /* define the block dim */
    uint32_t blockDim = 8;

    /* memory size */
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }

    /* lauch the kernal function via torch */
    EXEC_KERNEL_CMD(helloworld, blockDim, x, y, z, totalLength);
    return z;
}

HOST_API void printVersion()
{
    /*
     * dont remove this, this is used to put LIB_VERSION into symbol of the library,
     * then we can get the version and commit id via strings command, i.e.
     * strings libsgl_kernel_npu.so | grep commit
     */
    printf("%s\n", LIB_VERSION_FULL);
}

}  // namespace npu_kernel
}  // namespace sglang
