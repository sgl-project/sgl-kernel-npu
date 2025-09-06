#pragma once
#include "utils.h"
#include <mutex>
#include <stdio.h>
#include <string>
#include <sys/types.h>
#include <unordered_map>

struct AllocationMetadata {
  size_t size;
  int device;
  aclrtDrvMemHandle allocHandle;
  std::string tag;
  bool enable_cpu_backup;
  void *cpu_backup;
};

class TorchMemorySaver {
public:
  static TorchMemorySaver &instance();

  aclError malloc(void **ptr, int device, size_t size, const std::string &tag,
                  bool enable_cpu_backup);
  aclError free(void *ptr);

  void pause(const std::string &tag);
  void resume(const std::string &tag);

private:
  TorchMemorySaver();
  ~TorchMemorySaver() = default;
  TorchMemorySaver(const TorchMemorySaver &) = delete;
  TorchMemorySaver &operator=(const TorchMemorySaver &) = delete;

  std::mutex allocator_metadata_mutex_;
  std::unordered_map<void *, AllocationMetadata> allocation_metadata_;
};
