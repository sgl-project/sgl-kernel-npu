/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_ops_compat.h
 * \brief Stand-ins for the CANN op-project logging/validation macros.
 *
 * The vendored tiling includes "err/ops_err.h" for OP_CHECK_IF / OP_LOGE / OPS_REPORT_VECTOR_INNER_ERR,
 * a header that only exists inside a CANN custom-op package. Equivalents live here so the tiling body
 * can be reused verbatim and stay diffable against upstream.
 *
 * Unlike csrc/sparse_attn_sharedkv -- which stubs the log macros out to `((void)0)` -- these keep the
 * formatted message. The tiling is one long chain of shape/dtype/layout checks, and when one of them
 * rejects a call from Python the message is the only diagnostic available; discarding it would turn
 * every rejection into a bare "tiling failed". OP_CHECK_IF records the message and returns via the
 * caller's own retExpr, so upstream's control flow is preserved exactly and the message surfaces in
 * the TORCH_CHECK raised by the launcher.
 */
#ifndef SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_OPS_COMPAT_H
#define SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_OPS_COMPAT_H

#include <algorithm>
#include <cstdio>
#include <string>

namespace optiling {

inline std::string &SFALastErrorRef()
{
    thread_local std::string lastError;
    return lastError;
}

inline const std::string &SFALastError()
{
    return SFALastErrorRef();
}

template <typename... Args>
inline void SFARecordError(const char *opName, const char *fmt, Args... args)
{
    char buf[1024];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-security"
    const int written = std::snprintf(buf, sizeof(buf), fmt, args...);
#pragma GCC diagnostic pop
    const std::string message =
        (written > 0) ? std::string(buf, std::min(static_cast<size_t>(written), sizeof(buf) - 1)) : std::string(fmt);
    SFALastErrorRef() = std::string(opName != nullptr ? opName : "SparseFlashAttention") + ": " + message;
}

}  // namespace optiling

/*
 * Undefined first: unlike csrc/sparse_attn_sharedkv, which defines these at the top of its own .cpp,
 * these arrive through sparse_flash_attention_tiling.h *after* the acl/CANN headers the launcher
 * includes, and those headers are entitled to define log macros of their own. Redefining one is at
 * best a warning and at worst an error under -Werror, and it would be a confusing one to read.
 */
#undef OP_LOGE
#undef OP_LOGI
#undef OP_LOGW
#undef OP_LOGD
#undef OP_CHECK_IF
#undef OPS_REPORT_VECTOR_INNER_ERR

#define OP_LOGE(opName, ...) ::optiling::SFARecordError((opName), __VA_ARGS__)
#define OPS_REPORT_VECTOR_INNER_ERR(opName, ...) ::optiling::SFARecordError((opName), __VA_ARGS__)
#define OP_LOGI(...) ((void)0)
#define OP_LOGW(...) ((void)0)
#define OP_LOGD(...) ((void)0)

// Upstream usage: OP_CHECK_IF(cond, OP_LOGE(...), return ge::GRAPH_FAILED);
// The retExpr is kept so the vendored control flow is unchanged.
#define OP_CHECK_IF(cond, logExpr, retExpr) \
    do {                                    \
        if (cond) {                         \
            (logExpr);                      \
            retExpr;                        \
        }                                   \
    } while (0)

#endif  // SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_OPS_COMPAT_H
