// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KERNEL_CATCOC_KERNEL_H
#define KERNEL_CATCOC_KERNEL_H

#include <acl/acl.h>
#include "catcoc_host_tiling.h"

void catcoc_allgather_matmul_bf16_wnd_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);

void catcoc_allgather_matmul_fp16_wnd_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);

void catcoc_allgather_matmul_bf16_wnz_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);

void catcoc_allgather_matmul_fp16_wnz_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);

void catcoc_matmul_allreduce_bf16_wnd_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);

void catcoc_matmul_allreduce_fp16_wnd_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);

void catcoc_matmul_allreduce_bf16_wnz_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);

void catcoc_matmul_allreduce_fp16_wnz_kernel(uint32_t blockNum, aclrtStream stream, uint64_t fftsAddr, uint64_t teamIdx,
                                             uint8_t *gmA, uint8_t *gmB, uint8_t *gmC, uint8_t *gmSymmetric,
                                             uint8_t *gmWorkspace, uint8_t *gmTiling);
#endif  // KERNEL_CATCOC_KERNEL_H
