/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * ZBCCL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef __ZBCCL_MEM_ALLOCATOR_H__
#define __ZBCCL_MEM_ALLOCATOR_H__

#include "zbccl_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the allocator with options, this is allocator is secondary memory allocator,
 * the original memory is already allocated from Device, for example from mem fabric or shmem on Ascend.
 * The allocator can be plugged into torch, any address allocated from this allocator can be accessed by
 * other device directly, then we don't need to have CCL buffer to store the data temporarily.
 *
 * @param options          [in] the options of allocator
 * @param flags            [in] optional flags
 * @return 0 if successful
 */
int32_t zbccl_torch_allocator_init(zbccl_allocator_options *options, int32_t flags);

/**
 * @brief Un-initialize allocator
 *
 * @param flags            [in] optional flags
 */
void zbccl_torch_allocator_uninit(int32_t flags);

/**
 * @brief Allocate memory from secondary memory allocator
 *
 * @param size             [in] size of memory to be allocated
 * @param device           [in] device id
 * @param stream           [in] current stream
 * @return memory ptr is successful, nullptr is failed
 */
void *zbccl_torch_malloc(ssize_t size, int32_t device, aclrtStream stream);

/**
 * @brief Free the memory allocated back to the secondary memory allocator
 *
 * @param ptr              [in] pointer allocated by zbccl_torch_malloc
 * @param size             [in] size of memory
 * @param device           [in] device id
 * @param stream           [in] stream
 */
void zbccl_torch_free(void *ptr, ssize_t size, int32_t device, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // __ZBCCL_MEM_ALLOCATOR_H__
