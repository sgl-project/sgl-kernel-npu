#ifndef SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_OPS_ERR_H
#define SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_OPS_ERR_H

#include <cstdarg>
#include <cstdio>

namespace sgl_kernel_npu::sparse_flash_attention {

inline void Log(const char *level, const char *format, ...)
{
    std::printf("[%s] ", level);
    va_list arguments;
    va_start(arguments, format);
    std::vprintf(format, arguments);
    va_end(arguments);
    std::printf("\n");
}

}  // namespace sgl_kernel_npu::sparse_flash_attention

#define OP_LOGI(op_name, ...)
#define OP_LOGD(op_name, ...)

#define OP_LOGW(op_name, ...)                                             \
    do {                                                                  \
        (void)(op_name);                                                  \
        sgl_kernel_npu::sparse_flash_attention::Log("WARN", __VA_ARGS__); \
    } while (0)

#define OP_LOGE(op_name, ...)                                              \
    do {                                                                   \
        (void)(op_name);                                                   \
        sgl_kernel_npu::sparse_flash_attention::Log("ERROR", __VA_ARGS__); \
    } while (0)

#define OP_CHECK_IF(condition, log_func, expression) \
    do {                                             \
        if (condition) {                             \
            log_func;                                \
            expression;                              \
        }                                            \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer)                      \
    do {                                                                  \
        if ((pointer) == nullptr) {                                       \
            OP_LOGE((context)->GetNodeName(), "%s is nullptr", #pointer); \
            return ge::GRAPH_FAILED;                                      \
        }                                                                 \
    } while (0)

#define OPS_REPORT_VECTOR_INNER_ERR(op_name, fmt, ...) OP_LOGE(op_name, fmt, ##__VA_ARGS__)

#endif
