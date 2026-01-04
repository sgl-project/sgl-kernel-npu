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
#ifndef ZBCCL_H_
#define ZBCCL_H_

#include "zbccl_mem_allocator.h"
#include "zbccl_operations.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get version of zbccl
 *
 * @return string of version
 */
const char *zbccl_version();

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_H_
