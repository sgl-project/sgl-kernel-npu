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
#ifndef ZBCCL_OPERATIONS_H_
#define ZBCCL_OPERATIONS_H_

#include "zbccl_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize zero buffer collective communication library
 *
 * @return 0 if successful
 */
int32_t zbccl_init();

/**
 * @brief Do all reduce operation
 *
 * @param send_buff         [in] pointer of send buffer
 * @param recv_buff         [in] pointer of receive buffer
 * @param count             [in] size of buffer
 * @param data_type         [in] data type
 * @param op                [in] operation type of reduce
 * @param comm              [in] zbccl communication handle
 * @param stream            [in] stream
 * @return 0 if successful
 */
int32_t zbccl_all_reduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                         zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream);

/**
 * @brief Do reduce scatter operation
 *
 * @param send_buff        [in] pointer of send buffer
 * @param recv_buff        [in] pointer of receive buffer
 * @param recv_count       [in] size of buffer
 * @param data_type        [in] data type
 * @param op               [in] operation type of reduce
 * @param comm             [in] zbccl communication handle
 * @param stream           [in] stream
 * @return 0 if successful
 */
int32_t zbccl_reduce_scatter(const void *send_buff, void *recv_buff, size_t recv_count, zbccl_datatype_t data_type,
                             zbccl_reduce_op_t op, zbccl_comm_t comm, aclrtStream stream);

/**
 * @brief Do all gather operation
 *
 * @param send_buff        [in] pointer of send buffer
 * @param recv_buff        [in] pointer of receive buffer
 * @param send_count       [in] size of buffer
 * @param data_type        [in] data type
 * @param comm             [in] zbccl communication handle
 * @param stream           [in] stream
 * @return
 */
int32_t zbccl_all_gather(const void *send_buff, void *recv_buff, size_t send_count, zbccl_datatype_t data_type,
                         zbccl_comm_t comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ZBCCL_OPERATIONS_H_
