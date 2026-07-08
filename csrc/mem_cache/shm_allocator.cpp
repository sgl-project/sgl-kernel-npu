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

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <sys/shm.h>

extern "C" {
drvError_t halHostRegister(void *srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void **dstPtr);
drvError_t halHostUnregister(void *srcPtr, UINT32 devid);
}

namespace {

constexpr int kShmWriteOwn = 0600;

struct ShmEntry {
    void *host_ptr = nullptr;
    void *dev_ptr = nullptr;
    int shm_id = -1;
    uint64_t size = 0;
    int capacity = 0;
    int device_id = -1;
    ShmAllocatorMode mode = SHM_ALLOCATOR_MODE_PCIE_TH_DEV;
    bool owns_host_memory = false;
    bool is_registered = false;
};

std::mutex g_mu;
std::unordered_map<std::string, ShmEntry> g_entries;
std::unordered_set<int> g_acl_initialized_devices;
ShmAllocatorMode g_mode = SHM_ALLOCATOR_MODE_AUTO;

bool HasPrefix(const std::string &s, const std::string &prefix)
{
    return s.rfind(prefix, 0) == 0;
}

ShmAllocatorMode DetectModeByEnv()
{
    const char *soc = std::getenv("SOC_VERSION");
    if (soc == nullptr) {
        soc = std::getenv("ASCEND_SOC_VERSION");
    }
    if (soc == nullptr) {
        return SHM_ALLOCATOR_MODE_PCIE_TH_DEV;
    }

    std::string soc_name(soc);
    if (HasPrefix(soc_name, "Ascend910_93") || HasPrefix(soc_name, "910_93")) {
        return SHM_ALLOCATOR_MODE_SVM_MAP_DEV;
    }
    return SHM_ALLOCATOR_MODE_PCIE_TH_DEV;
}

ShmAllocatorMode ResolveMode()
{
    if (g_mode != SHM_ALLOCATOR_MODE_AUTO) {
        return g_mode;
    }
    return DetectModeByEnv();
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
    if (g_acl_initialized_devices.find(device_id) != g_acl_initialized_devices.end()) {
        return 0;
    }

    int32_t current_device = -1;
    aclError ret = aclrtGetDevice(&current_device);
    if (ret == ACL_ERROR_NONE) {
        if (current_device == device_id) {
            g_acl_initialized_devices.insert(device_id);
            return 0;
        }
        ret = aclrtSetDevice(device_id);
        if (ret != ACL_ERROR_NONE) {
            return -2;
        }
        g_acl_initialized_devices.insert(device_id);
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
    g_acl_initialized_devices.insert(device_id);
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

    std::memset(ptr, 0, size);
    *host_ptr = ptr;
    *shm_id = id;
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

int RegisterHostPtrLocked(const char *name, int device_id, void *host_ptr, uint64_t size, int capacity,
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

    int ret = EnsureAclDevice(device_id);
    if (ret != 0) {
        return ret;
    }

    ShmAllocatorMode mode = ResolveMode();
    void *dev_ptr = nullptr;
    uint32_t flag = GetRegisterFlag(mode);
    if (halHostRegister(host_ptr, size, flag, static_cast<UINT32>(device_id), &dev_ptr) != DRV_ERROR_NONE) {
        return -4;
    }

    if (it != g_entries.end()) {
        it->second.dev_ptr = dev_ptr;
        it->second.size = size;
        it->second.capacity = capacity > 0 ? capacity : 1;
        it->second.device_id = device_id;
        it->second.mode = mode;
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
    entry.device_id = device_id;
    entry.mode = mode;
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

int shm_free_by_name(const char *name, int device_id)
{
    if (name == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_entries.find(name);
    if (it == g_entries.end()) {
        return -2;
    }
    if (it->second.is_registered &&
        halHostUnregister(it->second.host_ptr, static_cast<UINT32>(device_id)) != DRV_ERROR_NONE) {
        return -3;
    }
    FreeHostMemory(it->second);
    g_entries.erase(it);
    return 0;
}

int shm_free_all(int device_id)
{
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &kv : g_entries) {
        if (kv.second.is_registered) {
            (void)halHostUnregister(kv.second.host_ptr, static_cast<UINT32>(device_id));
        }
        FreeHostMemory(kv.second);
    }
    g_entries.clear();
    g_acl_initialized_devices.erase(device_id);
    return 0;
}

}  // extern "C"
