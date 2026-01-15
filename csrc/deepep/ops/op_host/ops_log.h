/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: FusedDeepMoe tiling function implementation file
 * Author: Wang Yibo
 * Create: 2026-01-15
 * Note:
 * History: 2026-01-15 create log implementation file
 */

#pragma once

#include "dfx_base.h"

/* base log */
#define OPS_LOG_D(OPS_DESC, ...) OPS_LOG_STUB_D(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_I(OPS_DESC, ...) OPS_LOG_STUB_I(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_W(OPS_DESC, ...) OPS_LOG_STUB_W(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_E(OPS_DESC, ...) OPS_INNER_ERR_STUB("EZ9999", OPS_DESC, __VA_ARGS__)
#define OPS_LOG_E_WITHOUT_REPORT(OPS_DESC, ...) OPS_LOG_STUB_E(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_EVENT(OPS_DESC, ...) OPS_LOG_STUB_EVENT(OPS_DESC, __VA_ARGS__)

/* entire log
 * output long log, log will be divided by line if too long */
#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...) OPS_LOG_STUB_FULL(LEVEL, OPS_DESC, __VA_ARGS__)
#define OPS_LOG_D_FULL(OPS_DESC, ...) OPS_LOG_STUB_FULL(DLOG_DEBUG, OPS_DESC, __VA_ARGS__)
#define OPS_LOG_I_FULL(OPS_DESC, ...) OPS_LOG_STUB_FULL(DLOG_INFO, OPS_DESC, __VA_ARGS__)
#define OPS_LOG_W_FULL(OPS_DESC, ...) OPS_LOG_STUB_FULL(DLOG_WARN, OPS_DESC, __VA_ARGS__)

/* conditional log */
#define OPS_LOG_D_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_D(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_I_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_I(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_W_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_W(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_E_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_E(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_EVENT_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_EVENT(OP_DESC, __VA_ARGS__), EXPR)

#define OPS_LOG_E_IF_NULL(OPS_DESC, PTR, EXPR)                         \
    if (__builtin_expect((PTR) == nullptr, 0)) {                       \
        OPS_LOG_STUB_E(OPS_DESC, "%s is nullptr!", #PTR);              \
        OPS_CALL_ERR_STUB("EZ9999", OPS_DESC, "%s is nullptr!", #PTR); \
        EXPR;                                                          \
    }

#define OPS_CHECK(COND, LOG_FUNC, EXPR) \
    if (COND) {                         \
        LOG_FUNC;                       \
        EXPR;                           \
    }

#define OP_CHECK(COND, LOG_FUNC, EXPR) \
    if (COND) {                        \
        LOG_FUNC;                      \
        EXPR;                          \
    }
