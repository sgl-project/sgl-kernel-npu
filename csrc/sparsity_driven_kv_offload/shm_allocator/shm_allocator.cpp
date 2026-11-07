// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "shm_allocator.h"

#include <acl/acl.h>
#include <driver/ascend_hal_define.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include <sys/shm.h>

extern "C" {
drvError_t halHostRegister(void *srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void **dstPtr);
drvError_t halHostUnregister(void *srcPtr, UINT32 devid);
}

namespace {

constexpr int kShmWriteOwn = 0600;

enum class ShmRegisterBackend {
    ACL,
    HAL,
};

struct ShmEntry {
    void *host_ptr = nullptr;
    void *dev_ptr = nullptr;
    int shm_id = -1;
    uint64_t size = 0;
    int capacity = 0;
    int user_device_id = -1;
    int driver_device_id = -1;
    uint32_t register_flag = 0;
    ShmAllocatorMode mode = SHM_ALLOCATOR_MODE_PCIE_TH_DEV;
    ShmRegisterBackend register_backend = ShmRegisterBackend::HAL;
    bool owns_host_memory = false;
    bool is_registered = false;
};

std::mutex g_mu;
std::unordered_map<std::string, ShmEntry> g_entries;
ShmAllocatorMode g_mode = SHM_ALLOCATOR_MODE_AUTO;

ShmAllocatorMode ResolveMode()
{
    return g_mode;
}

uint32_t GetRegisterFlag(ShmAllocatorMode mode)
{
    if (mode == SHM_ALLOCATOR_MODE_SVM_MAP_DEV) {
        return HOST_SVM_MAP_DEV;
    }
    return HOST_MEM_MAP_DEV_PCIE_TH;
}

int EnsureAclDevice(int device_id)
{
    int32_t current_device = -1;
    aclError ret = aclrtGetDevice(&current_device);
    if (ret == ACL_ERROR_NONE) {
        if (current_device == device_id) {
            return 0;
        }
        ret = aclrtSetDevice(device_id);
        if (ret != ACL_ERROR_NONE) {
            return -2;
        }
        return 0;
    }

    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return -1;
    }
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_ERROR_NONE) {
        return -2;
    }
    return 0;
}

int AllocHostMemory(uint64_t size, void **host_ptr, int *shm_id)
{
    int id = shmget(IPC_PRIVATE, size, IPC_CREAT | kShmWriteOwn);
    if (id == -1) {
        return -1;
    }

    void *ptr = shmat(id, nullptr, 0);
    if (ptr == reinterpret_cast<void *>(-1)) {
        (void)shmctl(id, IPC_RMID, nullptr);
        return -2;
    }

    // Keep the current mapping alive, but reclaim the segment automatically
    // after the last attachment is detached, including on abnormal process exit.
    if (shmctl(id, IPC_RMID, nullptr) == -1) {
        (void)shmdt(ptr);
        (void)shmctl(id, IPC_RMID, nullptr);
        return -3;
    }

    std::memset(ptr, 0, size);
    *host_ptr = ptr;
    *shm_id = -1;
    return 0;
}

void FreeHostMemory(const ShmEntry &entry)
{
    if (!entry.owns_host_memory) {
        return;
    }
    if (entry.host_ptr != nullptr) {
        (void)shmdt(entry.host_ptr);
    }
    if (entry.shm_id >= 0) {
        (void)shmctl(entry.shm_id, IPC_RMID, nullptr);
    }
}

int RegisterHostPtrLocked(const char *name, int user_device_id, void *host_ptr, uint64_t size, int capacity,
                          bool owns_host_memory, uint64_t *out_dev_ptr)
{
    auto it = g_entries.find(name);
    if (it != g_entries.end()) {
        if (it->second.is_registered) {
            *out_dev_ptr = reinterpret_cast<uint64_t>(it->second.dev_ptr);
            return 0;
        }
        if (it->second.host_ptr != host_ptr) {
            return -5;
        }
    }

    int ret = EnsureAclDevice(user_device_id);
    if (ret != 0) {
        return ret;
    }

    int32_t driver_device_id = -1;
    ShmAllocatorMode mode = ResolveMode();
    // AUTO uses the public ACL registration API; HAL registration remains opt-in through shm_set_mode.
    ShmRegisterBackend backend = mode == SHM_ALLOCATOR_MODE_AUTO ? ShmRegisterBackend::ACL : ShmRegisterBackend::HAL;
    void *dev_ptr = nullptr;
    uint32_t flag = 0;
    if (backend == ShmRegisterBackend::ACL) {
        aclError register_ret = aclrtHostRegister(host_ptr, size, ACL_HOST_REGISTER_MAPPED, &dev_ptr);
        if (register_ret != ACL_SUCCESS) {
            std::fprintf(stderr,
                         "[shm_allocator] aclrtHostRegister failed: ret=%d user_device_id=%d type=%u bytes=%llu\n",
                         static_cast<int>(register_ret), user_device_id,
                         static_cast<unsigned>(ACL_HOST_REGISTER_MAPPED), static_cast<unsigned long long>(size));
            return -4;
        }
        flag = static_cast<uint32_t>(ACL_HOST_REGISTER_MAPPED);
    } else {
        aclError mapping_ret = aclrtGetLogicDevIdByUserDevId(user_device_id, &driver_device_id);
        if (mapping_ret != ACL_ERROR_NONE || driver_device_id < 0) {
            std::fprintf(stderr,
                         "[shm_allocator] aclrtGetLogicDevIdByUserDevId failed: ret=%d user_device_id=%d "
                         "driver_device_id=%d\n",
                         static_cast<int>(mapping_ret), user_device_id, driver_device_id);
            return -7;
        }

        flag = GetRegisterFlag(mode);
        drvError_t register_ret =
            halHostRegister(host_ptr, size, flag, static_cast<UINT32>(driver_device_id), &dev_ptr);
        if (register_ret != DRV_ERROR_NONE) {
            std::fprintf(stderr,
                         "[shm_allocator] halHostRegister failed: ret=%d user_device_id=%d driver_device_id=%d "
                         "flag=%u bytes=%llu\n",
                         static_cast<int>(register_ret), user_device_id, driver_device_id, flag,
                         static_cast<unsigned long long>(size));
            return -4;
        }
    }

    if (it != g_entries.end()) {
        it->second.dev_ptr = dev_ptr;
        it->second.size = size;
        it->second.capacity = capacity > 0 ? capacity : 1;
        it->second.user_device_id = user_device_id;
        it->second.driver_device_id = driver_device_id;
        it->second.register_flag = flag;
        it->second.mode = mode;
        it->second.register_backend = backend;
        it->second.owns_host_memory = owns_host_memory || it->second.owns_host_memory;
        it->second.is_registered = true;
        *out_dev_ptr = reinterpret_cast<uint64_t>(dev_ptr);
        return 0;
    }

    ShmEntry entry;
    entry.host_ptr = host_ptr;
    entry.dev_ptr = dev_ptr;
    entry.shm_id = -1;
    entry.size = size;
    entry.capacity = capacity > 0 ? capacity : 1;
    entry.user_device_id = user_device_id;
    entry.driver_device_id = driver_device_id;
    entry.register_flag = flag;
    entry.mode = mode;
    entry.register_backend = backend;
    entry.owns_host_memory = owns_host_memory;
    entry.is_registered = true;
    g_entries.emplace(name, entry);

    *out_dev_ptr = reinterpret_cast<uint64_t>(dev_ptr);
    return 0;
}

}  // namespace

extern "C" {

void shm_set_mode(ShmAllocatorMode mode)
{
    std::lock_guard<std::mutex> lock(g_mu);
    g_mode = mode;
}

int shm_create_and_register(const char *name, int device_id, uint64_t size, int capacity, uint64_t *out_dev_ptr)
{
    if (name == nullptr || out_dev_ptr == nullptr || size == 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mu);

    auto existing = g_entries.find(name);
    if (existing != g_entries.end()) {
        if (existing->second.size != size) {
            return -6;
        }
        if (existing->second.is_registered) {
            *out_dev_ptr = reinterpret_cast<uint64_t>(existing->second.dev_ptr);
            return 0;
        }
        return RegisterHostPtrLocked(name, device_id, existing->second.host_ptr, existing->second.size, capacity,
                                     existing->second.owns_host_memory, out_dev_ptr);
    }

    void *host_ptr = nullptr;
    int shm_id = -1;
    int ret = AllocHostMemory(size, &host_ptr, &shm_id);
    if (ret != 0) {
        return ret;
    }

    ShmEntry entry;
    entry.host_ptr = host_ptr;
    entry.shm_id = shm_id;
    entry.size = size;
    entry.capacity = capacity > 0 ? capacity : 1;
    entry.mode = SHM_ALLOCATOR_MODE_PCIE_TH_DEV;
    entry.owns_host_memory = true;
    entry.is_registered = false;
    g_entries.emplace(name, entry);

    ret = RegisterHostPtrLocked(name, device_id, host_ptr, size, capacity, true, out_dev_ptr);
    if (ret != 0) {
        auto it = g_entries.find(name);
        if (it != g_entries.end()) {
            FreeHostMemory(it->second);
            g_entries.erase(it);
        }
        return ret;
    }
    return 0;
}

int shm_create(const char *name, uint64_t size, void **out_host_ptr)
{
    if (name == nullptr || out_host_ptr == nullptr || size == 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mu);

    auto existing = g_entries.find(name);
    if (existing != g_entries.end()) {
        *out_host_ptr = existing->second.host_ptr;
        return 0;
    }

    void *host_ptr = nullptr;
    int shm_id = -1;
    int ret = AllocHostMemory(size, &host_ptr, &shm_id);
    if (ret != 0) {
        return ret;
    }

    ShmEntry entry;
    entry.host_ptr = host_ptr;
    entry.shm_id = shm_id;
    entry.size = size;
    entry.capacity = 1;
    entry.mode = SHM_ALLOCATOR_MODE_PCIE_TH_DEV;
    entry.owns_host_memory = true;
    entry.is_registered = false;
    g_entries.emplace(name, entry);

    *out_host_ptr = host_ptr;
    return 0;
}

int shm_register(const char *name, int device_id, void *host_ptr, uint64_t size, uint64_t *out_dev_ptr)
{
    if (name == nullptr || host_ptr == nullptr || out_dev_ptr == nullptr || size == 0) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    return RegisterHostPtrLocked(name, device_id, host_ptr, size, 1, false, out_dev_ptr);
}

void *shm_get_host_ptr(const char *name)
{
    if (name == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_entries.find(name);
    if (it == g_entries.end()) {
        return nullptr;
    }
    return it->second.host_ptr;
}

int shm_free_by_name(const char *name, int /*device_id*/)
{
    if (name == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_entries.find(name);
    if (it == g_entries.end()) {
        return -2;
    }
    if (it->second.is_registered) {
        int unregister_status = 0;
        int unregister_ret = 0;
        if (it->second.register_backend == ShmRegisterBackend::ACL) {
            unregister_status = EnsureAclDevice(it->second.user_device_id);
            if (unregister_status == 0) {
                unregister_ret = static_cast<int>(aclrtHostUnregister(it->second.host_ptr));
            }
        } else {
            unregister_ret = static_cast<int>(
                halHostUnregister(it->second.host_ptr, static_cast<UINT32>(it->second.driver_device_id)));
            unregister_status = unregister_ret == DRV_ERROR_NONE ? 0 : -1;
        }
        if (unregister_status != 0 || unregister_ret != 0) {
            std::fprintf(stderr,
                         "[shm_allocator] unregister failed: backend=%s ret=%d status=%d user_device_id=%d "
                         "driver_device_id=%d flag=%u bytes=%llu\n",
                         it->second.register_backend == ShmRegisterBackend::ACL ? "acl" : "hal", unregister_ret,
                         unregister_status, it->second.user_device_id, it->second.driver_device_id,
                         it->second.register_flag, static_cast<unsigned long long>(it->second.size));
            return -3;
        }
    }
    FreeHostMemory(it->second);
    g_entries.erase(it);
    return 0;
}

int shm_free_all(int /*device_id*/)
{
    std::lock_guard<std::mutex> lock(g_mu);
    int ret = 0;
    for (auto it = g_entries.begin(); it != g_entries.end();) {
        if (it->second.is_registered) {
            int unregister_status = 0;
            int unregister_ret = 0;
            if (it->second.register_backend == ShmRegisterBackend::ACL) {
                unregister_status = EnsureAclDevice(it->second.user_device_id);
                if (unregister_status == 0) {
                    unregister_ret = static_cast<int>(aclrtHostUnregister(it->second.host_ptr));
                }
            } else {
                unregister_ret = static_cast<int>(
                    halHostUnregister(it->second.host_ptr, static_cast<UINT32>(it->second.driver_device_id)));
                unregister_status = unregister_ret == DRV_ERROR_NONE ? 0 : -1;
            }
            if (unregister_status != 0 || unregister_ret != 0) {
                std::fprintf(stderr,
                             "[shm_allocator] unregister failed: backend=%s ret=%d status=%d user_device_id=%d "
                             "driver_device_id=%d flag=%u bytes=%llu\n",
                             it->second.register_backend == ShmRegisterBackend::ACL ? "acl" : "hal", unregister_ret,
                             unregister_status, it->second.user_device_id, it->second.driver_device_id,
                             it->second.register_flag, static_cast<unsigned long long>(it->second.size));
                ret = -3;
                ++it;
                continue;
            }
        }
        FreeHostMemory(it->second);
        it = g_entries.erase(it);
    }
    return ret;
}

}  // extern "C"
