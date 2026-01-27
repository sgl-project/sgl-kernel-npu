/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: FusedDeepMoe tiling function implementation file
 * Author: Wang Yibo
 * Create: 2026-01-15
 * Note:
 * History: 2026-01-15 create log implementation file
 */

#pragma once

#include "ops_log.h"

/* base error */
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E89999", OPS_DESC, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E69999", OPS_DESC, __VA_ARGS__)

/* conditional error */
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR) OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)
