#include "api_forwarder.h"
#include "core.h"
#include "utils.h"

// ----------------------------------------------- threadlocal configs
// --------------------------------------------------

struct ThreadLocalConfig {
  bool is_interesting_region_ = false;
  std::string current_tag_ = "default";
  bool enable_cpu_backup_ = false;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook
// ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
aclError aclrtMallocAlign32(void **ptr, size_t size,
                            aclrtMemMallocPolicy policy) {
  if (thread_local_config.is_interesting_region_) {
    return TorchMemorySaver::instance().malloc(
        ptr, CANNUtils::cann_ctx_get_device(), size,
        thread_local_config.current_tag_,
        thread_local_config.enable_cpu_backup_);
  } else {
    return APIForwarder::call_real_aclrt_malloc_align32(ptr, size, policy);
  }
}

aclError aclrtFree(void *ptr) {
  if (thread_local_config.is_interesting_region_) {
    return TorchMemorySaver::instance().free(ptr);
  } else {
    return APIForwarder::call_real_aclrt_free(ptr);
  }
}
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, aclrtStream stream) {
#ifdef TMS_DEBUG_LOG
  std::cout << "[torch_memory_saver.cpp] tms_torch_malloc "
            << " size=" << size << " device=" << device << " stream=" << stream
            << std::endl;
#endif
  SIMPLE_CHECK(thread_local_config.is_interesting_region_,
               "only support interesting region");
  void *ptr;
  TorchMemorySaver::instance().malloc(&ptr, CANNUtils::cann_device_get(device),
                                      size, thread_local_config.current_tag_,
                                      thread_local_config.enable_cpu_backup_);
  return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, aclrtStream stream) {
#ifdef TMS_DEBUG_LOG
  std::cout << "[torch_memory_saver.cpp] tms_torch_free "
            << " ptr=" << ptr << " ssize=" << ssize << " device=" << device
            << " stream=" << stream << std::endl;
#endif
  SIMPLE_CHECK(thread_local_config.is_interesting_region_,
               "only support interesting region");
  TorchMemorySaver::instance().free(ptr);
}
}
#endif

// ------------------------------------------------- entrypoints :: others
// ------------------------------------------------

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
  thread_local_config.is_interesting_region_ = is_interesting_region;
}

bool tms_get_interesting_region() {
  return thread_local_config.is_interesting_region_;
}

void tms_set_current_tag(const char *tag) {
  SIMPLE_CHECK(tag != nullptr, "tag should not be null");
  thread_local_config.current_tag_ = tag;
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
  thread_local_config.enable_cpu_backup_ = enable_cpu_backup;
}

void tms_pause(const char *tag) {
  std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
  TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char *tag) {
  std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
  TorchMemorySaver::instance().resume(tag_str);
}
}
