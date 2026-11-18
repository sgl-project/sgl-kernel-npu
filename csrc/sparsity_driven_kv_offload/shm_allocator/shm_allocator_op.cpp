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

#include "torch_helper.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <tuple>

namespace sglang {
namespace npu_kernel {
namespace {

std::atomic<uint64_t> g_shm_name_counter{0};

std::string ResolveName(c10::string_view name)
{
    if (!name.empty()) {
        return std::string(name);
    }
    uint64_t index = g_shm_name_counter.fetch_add(1, std::memory_order_relaxed);
    return "sgl_kernel_npu_shm_" + std::to_string(index);
}

}  // namespace

std::tuple<int64_t, int64_t> shm_allocator_create_and_register(int64_t size, int64_t device_id, c10::string_view name)
{
    TORCH_CHECK(size > 0, "shm_allocator_create_and_register: size must be positive, got ", size);
    TORCH_CHECK(device_id >= 0, "shm_allocator_create_and_register: device_id must be non-negative, got ", device_id);

    std::string shm_name = ResolveName(name);
    uint64_t dev_ptr = 0;
    int ret = shm_create_and_register(shm_name.c_str(), static_cast<int>(device_id), static_cast<uint64_t>(size), 1,
                                      &dev_ptr);
    TORCH_CHECK(ret == 0, "shm_allocator_create_and_register failed, ret=", ret);

    void *host_ptr = shm_get_host_ptr(shm_name.c_str());
    TORCH_CHECK(host_ptr != nullptr, "shm_allocator_create_and_register: host pointer is null");
    return {reinterpret_cast<int64_t>(host_ptr), static_cast<int64_t>(dev_ptr)};
}

void shm_allocator_free_all(int64_t device_id)
{
    TORCH_CHECK(device_id >= 0, "shm_allocator_free_all: device_id must be non-negative, got ", device_id);
    int ret = shm_free_all(static_cast<int>(device_id));
    TORCH_CHECK(ret == 0, "shm_allocator_free_all failed, ret=", ret);
}

}  // namespace npu_kernel
}  // namespace sglang
