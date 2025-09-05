#include "api_forwarder.h"
#include "utils.h"
#include <iostream>

namespace APIForwarder {

using AclrtMallocFuncAlign32 = aclError (*)(void **, size_t,
                                            aclrtMemMallocPolicy);
using AclrtFreeFunc = aclError (*)(void *);

static void *check_dlsym(void *value) {
  if (nullptr == value) {
    std::cerr << "[torch_memory_saver.cpp] dlsym failed dlerror=" << dlerror()
              << std::endl;
    exit(1);
  }
  return value;
}

static AclrtMallocFuncAlign32 real_aclrt_malloc_align32_ = NULL;
static AclrtFreeFunc real_aclrt_free_ = NULL;

aclError call_real_aclrt_malloc_align32(void **ptr, size_t size,
                                        aclrtMemMallocPolicy policy) {
  if (C10_UNLIKELY(nullptr == real_aclrt_malloc_align32_)) {
    real_aclrt_malloc_align32_ = (AclrtMallocFuncAlign32)check_dlsym(
        dlsym(RTLD_NEXT, "aclrtMallocAlign32"));
  }

  aclError ret = real_aclrt_malloc_align32_(ptr, size, policy);

#ifdef TMS_DEBUG_LOG
  std::cout << "[torch_memory_saver.cpp] cudaMalloc [MODE NORMAL]"
            << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
            << " ret=" << ret << std::endl;
#endif

  return ret;
}

aclError call_real_aclrt_free(void *ptr) {
  if (C10_UNLIKELY(nullptr == real_aclrt_free_)) {
    real_aclrt_free_ =
        (AclrtFreeFunc)check_dlsym(dlsym(RTLD_NEXT, "aclrtFree"));
  }

  aclError ret = real_aclrt_free_(ptr);

#ifdef TMS_DEBUG_LOG
  std::cout << "[torch_memory_saver.cpp] cudaFree [MODE NORMAL]"
            << " ptr=" << ptr << " ret=" << ret << std::endl;
#endif

  return ret;
}
} // namespace APIForwarder
