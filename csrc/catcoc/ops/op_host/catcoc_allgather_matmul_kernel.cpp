// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "catcoc_host_tiling.h"
#include "catcoc_kernel.h"
#include "../op_kernel/catcoc_allgather_matmul_kernel.hpp"

using namespace AscendC;

void catcoc_allgather_matmul_bf16_wnd_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling)
{
    catcoc_allgather_matmul_bf16<<<blockNum, nullptr, stream>>>(fftsAddr, teamIdx, gmA, gmB, gmC, gmSymmetric,
                                                                gmWorkspace, gmTiling);
}

void catcoc_allgather_matmul_bf16_wnz_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling)
{
    catcoc_allgather_matmul_bf16_wnz<<<blockNum, nullptr, stream>>>(fftsAddr, teamIdx, gmA, gmB, gmC, gmSymmetric,
                                                                    gmWorkspace, gmTiling);
}

void catcoc_allgather_matmul_fp16_wnd_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling)
{
    catcoc_allgather_matmul_fp16<<<blockNum, nullptr, stream>>>(fftsAddr, teamIdx, gmA, gmB, gmC, gmSymmetric,
                                                                gmWorkspace, gmTiling);
}

void catcoc_allgather_matmul_fp16_wnz_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling)
{
    catcoc_allgather_matmul_fp16_wnz<<<blockNum, nullptr, stream>>>(fftsAddr, teamIdx, gmA, gmB, gmC, gmSymmetric,
                                                                    gmWorkspace, gmTiling);
}
