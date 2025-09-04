#include "core.h"
#include "utils.h"

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

aclError TorchMemorySaver::malloc(void **ptr, int device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
    aclrtDrvMemHandle allocHandle;
    CANNUtils::cann_mem_create(&allocHandle, size, device);
    int ret = aclrtReserveMemAddress(ptr, size, 0, nullptr, 0);
    SIMPLE_CHECK(ret == ACL_SUCCESS, "aclrtReserveMemAddress failed");
    ret = aclrtMapMem(*ptr, size, 0, allocHandle, 0);
    SIMPLE_CHECK(ret == ACL_SUCCESS, "aclrtMapMem failed");
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(*ptr, AllocationMetadata{size, device, allocHandle, tag, enable_cpu_backup, nullptr});
    }
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif
    return ACL_SUCCESS;
}

aclError TorchMemorySaver::free(void *ptr)
{
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        SIMPLE_CHECK(allocation_metadata_.count(ptr), "Trying to free a pointer not allocated here");
        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }
    int ret = aclrtUnmapMem(ptr);
    ret = aclrtFreePhysical(metadata.allocHandle);
    ret = aclrtReleaseMemAddress(ptr);
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif
    return ACL_SUCCESS;
}

void TorchMemorySaver::pause(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;
        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                aclrtMallocHost(&metadata.cpu_backup, metadata.size);
            }
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            aclrtMemcpy(metadata.cpu_backup, metadata.size, ptr, metadata.size, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        int ret = aclrtUnmapMem(ptr);
        ret = aclrtFreePhysical(metadata.allocHandle);

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
        
#endif

    }
}

void TorchMemorySaver::resume(const std::string& tag) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        aclrtDrvMemHandle newAllocHandle;
        CANNUtils::cann_mem_create(&newAllocHandle, metadata.size, metadata.device);

        aclrtMapMem(ptr, metadata.size, 0, newAllocHandle, 0);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            // TODO may use cudaMemcpyAsync if needed
            aclrtMemcpy(ptr, metadata.size, metadata.cpu_backup, metadata.size, ACL_MEMCPY_HOST_TO_DEVICE);
            // maybe we can free host memory if needed (currently keep it there to reduce re-alloc time)
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " (old)metadata.allocHandle="
                  << metadata.allocHandle
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif

        metadata.allocHandle = newAllocHandle;
    }
}
