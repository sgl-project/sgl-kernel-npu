// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_SHM_ALLOCATOR_H
#define SGL_KERNEL_NPU_SHM_ALLOCATOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ShmAllocatorMode {
    SHM_ALLOCATOR_MODE_AUTO = 0,
    SHM_ALLOCATOR_MODE_SVM_MAP_DEV = 1,
    SHM_ALLOCATOR_MODE_PCIE_TH_DEV = 2
} ShmAllocatorMode;

int shm_create(const char *name, uint64_t size, void **out_host_ptr);
int shm_create_and_register(const char *name, int device_id, uint64_t size, int capacity, uint64_t *out_dev_ptr);
int shm_register(const char *name, int device_id, void *host_ptr, uint64_t size, uint64_t *out_dev_ptr);
void *shm_get_host_ptr(const char *name);
int shm_free_all(int device_id);
int shm_free_by_name(const char *name, int device_id);
void shm_set_mode(ShmAllocatorMode mode);

#ifdef __cplusplus
}
#endif

#endif  // SGL_KERNEL_NPU_SHM_ALLOCATOR_H
