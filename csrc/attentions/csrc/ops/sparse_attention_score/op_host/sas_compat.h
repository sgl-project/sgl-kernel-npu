#pragma once
/*
 * Minimal self-contained shims for the vllm-ascend log/err macros used by the
 * sparse_attention_score host files, so they compile under sgl-kernel-npu
 * (which lacks CANN op_common/log/log.h and the vllm-ascend err/ops_err.h +
 * tiling_base shims). Logging is a no-op; the control-flow checks keep their
 * return semantics so tiling/infershape still fail fast on null inputs.
 */
#ifndef SAS_COMPAT_H
#define SAS_COMPAT_H

// vllm-ascend tiling_base.h defines this empty in normal builds (extern "C"
// only under ASCENDC_OP_TEST). The tiling entry functions use it for linkage.
#ifndef ASCENDC_EXTERN_C
#define ASCENDC_EXTERN_C
#endif

#define OP_LOGE(...) ((void)0)
#define OP_LOGD(...) ((void)0)
#define OP_LOGW(...) ((void)0)
#define OP_LOGI(...) ((void)0)

#define OPS_REPORT_VECTOR_INNER_ERR(...) ((void)0)
#define OPS_REPORT_CUBE_INNER_ERR(...) ((void)0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    do {                                         \
        if ((ptr) == nullptr) {                  \
            return ge::GRAPH_FAILED;             \
        }                                        \
    } while (0)

#define OP_CHECK_IF(cond, log_action, ret_expr) \
    do {                                        \
        if (cond) {                             \
            log_action;                         \
            ret_expr;                           \
        }                                       \
    } while (0)

#endif  // SAS_COMPAT_H
