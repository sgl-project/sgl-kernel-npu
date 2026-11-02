#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_

#include <string>

#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)      \
    printf("[WARN]" __VA_ARGS__); \
    printf("\n")
#define OP_LOGE_WITHOUT_REPORT(opname, ...) \
    printf("[ERRORx]" __VA_ARGS__);         \
    printf("\n")
#define OP_LOGE(opname, ...)       \
    printf("[ERROR]" __VA_ARGS__); \
    printf("\n")
#define OP_LOGD(opname, ...)

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)   \
    do {                                                         \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
    } while (0)

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
