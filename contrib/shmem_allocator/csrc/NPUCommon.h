#pragma once

#include <mutex>
#include <atomic>

struct OptionsManager {
    static bool IsHcclZeroCopyEnable;
    static bool CheckForceUncached;
};

std::string formatErrorCode(int32_t errorCode);

#define PTA_ERROR_MOCK(err_code) formatErrorCode((int32_t)err_code)
#define OPS_ERROR_MOCK(err_code) formatErrorCode((int32_t)err_code)

#define NPU_CHECK_ERROR_MOCK(err_code, ...)                                  \
    do {                                                                     \
        int error_code = err_code;                                           \
        if ((error_code) != ACL_ERROR_NONE) {                                \
            std::ostringstream oss;                                          \
            oss << " NPU function error: [ShmemAllocator Currently do not support detail error log]" << std::endl; \
            std::string err_msg = oss.str();                                 \
            ASCEND_LOGE("%s", err_msg.c_str());                              \
        }                                                                    \
    } while (0)

#define NPU_CHECK_WARN_MOCK(err_code, ...)                                   \
    do {                                                                     \
        int error_code = err_code;                                           \
        if ((error_code) != ACL_ERROR_NONE) {                                \
            std::ostringstream oss;                                          \
            oss << " NPU function warning: [ShmemAllocator Currently do not support detail warning log]" << std::endl; \
            std::string err_msg = oss.str();                                 \
            ASCEND_LOGW("%s", err_msg.c_str());                              \
        }                                                                    \
    } while (0)

const int32_t ACL_SYNC_TIMEOUT = 3600 * 1000;  // ms
