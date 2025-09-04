#pragma once
#include <iostream>
#include <acl/acl.h>

//#define TMS_DEBUG_LOG

// Cannot use pytorch (libc10.so) since LD_PRELOAD happens earlier than `import torch`
// Thus copy from torch Macros.h
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

#define SIMPLE_CHECK(COND, MSG) \
  do { \
    if (!(COND)) { \
        std::cerr << "[torch_memory_saver.cpp] " << MSG \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define CURESULT_CHECK(EXPR) \
  do { \
    CUresult __result = (EXPR); \
    if (__result != CUDA_SUCCESS) { \
        const char* err_str = nullptr; \
        cuGetErrorString(__result, &err_str); \
        std::cerr << "[torch_memory_saver.cpp] CUresult error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

#define CUDA_ERROR_CHECK(EXPR) \
  do { \
    cudaError_t __result = (EXPR); \
    if (__result != cudaSuccess) { \
        const char* err_str = cudaGetErrorString(__result); \
        std::cerr << "[torch_memory_saver.cpp] cudaError error: " \
                  << __result << " (" << (err_str ? err_str : "Unknown error") << ") " \
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__ \
                  << std::endl; \
        exit(1); \
    } \
  } while (false)

namespace CANNUtils {
    static void cann_mem_create(aclrtDrvMemHandle *alloc_handle, size_t size, int device) {
        aclrtPhysicalMemProp prop = {};
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.reserve = 0;
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE ;
        int ret = aclrtMallocPhysical(alloc_handle, size, &prop, 0);
    }

    static int cann_ctx_get_device() {
        int ans;
        aclrtGetDevice(&ans);
        return ans;
    }

    static int cann_device_get(int device_ordinal) {
        return device_ordinal;
    }

}
