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
#ifndef ZBCCL_SMA_H
#define ZBCCL_SMA_H

#include "zbccl_common_includes.h"

namespace zbccl {
namespace sma {
class SecondaryMemoryAllocator : public ZReferable
{
public:
    virtual ~SecondaryMemoryAllocator() = default;

    /**
     * @brief Initialize the allocator
     *
     * @param options      [in] options for the allocator
     * @param flags        [in] extra flags
     * @return 0 is successful
     */
    virtual ZResult Initialize(zbccl_allocator_options *options, int32_t flags) noexcept = 0;

    /**
     * @brief Un-initialize the allocator
     *
     * @param flags        [in] extra flags
     */
    virtual void UnInitialize(int32_t flags) noexcept = 0;

    /**
     * @brief Allocate memory
     *
     * @param size         [in] size to be allocated
     * @param device       [in] device id
     * @param stream       [in] stream
     * @param flags        [in] optional flags
     * @param out          [out] pointer that allocated
     * @return 0 if successful
     */
    virtual ZResult Allocate(ssize_t size, int32_t device, aclrtStream stream, int32_t flags, void *&out) noexcept = 0;

    /**
     * @brief Free memory
     *
     * @param ptr          [in] memory pointer allocated by <i>Allocate</i>
     * @param size         [in] size of the memory
     * @param device       [in] deivce id
     * @param stream       [in] stream
     * @param flags        [in] optional flags
     * @return 0 if successful
     */
    virtual ZResult Free(void *ptr, ssize_t size, int32_t device, aclrtStream stream, int32_t flags) noexcept = 0;
};
using SMAPtr = ZRef<SecondaryMemoryAllocator>;
}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_H
