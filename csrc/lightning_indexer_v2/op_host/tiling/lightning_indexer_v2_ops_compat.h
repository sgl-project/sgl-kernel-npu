/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_v2_ops_compat.h
 * \brief CANN 算子工程日志/校验宏在 sgl-kernel-npu 中的替身。
 *
 * 上游 tiling 代码依赖 ops 仓的 err/ops_err.h (OP_CHECK_IF / OP_LOGE / ...)，
 * 该头文件只在 CANN 算子工程内可见。这里提供等价实现，使得 tiling 主体逻辑
 * 可以逐字复用、便于后续与上游对比同步：
 *   - OP_LOGE  记录一条格式化后的错误信息（不抛出），供随后的检查/返回使用；
 *   - OP_CHECK_IF 在条件成立时记录信息并通过 TORCH_CHECK 抛出；
 *   - OP_LOGI / OP_LOGW 为空实现。
 */
#ifndef SGL_LI_V2_OPS_COMPAT_H
#define SGL_LI_V2_OPS_COMPAT_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

#include "torch_helper.h"

namespace sglang {
namespace LIV2Host {

inline std::string &LastOpErrorRef()
{
    thread_local std::string lastError;
    return lastError;
}

inline const std::string &LastOpError()
{
    return LastOpErrorRef();
}

template <typename... Args>
inline void RecordOpError(const char *opName, const char *fmt, Args... args)
{
    char buf[1024];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-security"
    const int written = std::snprintf(buf, sizeof(buf), fmt, args...);
#pragma GCC diagnostic pop
    const std::string message =
        (written > 0) ? std::string(buf, std::min(static_cast<size_t>(written), sizeof(buf) - 1)) : std::string(fmt);
    LastOpErrorRef() = std::string(opName != nullptr ? opName : "LightningIndexerV2") + ": " + message;
}

}  // namespace LIV2Host
}  // namespace sglang

#define OP_LOGE(opName, ...) ::sglang::LIV2Host::RecordOpError((opName), __VA_ARGS__)
#define OPS_REPORT_VECTOR_INNER_ERR(opName, ...) ::sglang::LIV2Host::RecordOpError((opName), __VA_ARGS__)
#define OP_LOGI(...) ((void)0)
#define OP_LOGW(...) ((void)0)

// 上游用法: OP_CHECK_IF(cond, OP_LOGE(...), return ge::GRAPH_FAILED);
// 这里改为直接抛出，retExpr 不再需要（保留形参以维持调用点原样）。
#define OP_CHECK_IF(cond, logExpr, retExpr)                          \
    do {                                                             \
        if (cond) {                                                  \
            (logExpr);                                               \
            TORCH_CHECK(false, ::sglang::LIV2Host::LastOpError());   \
        }                                                            \
    } while (0)

#endif  // SGL_LI_V2_OPS_COMPAT_H
