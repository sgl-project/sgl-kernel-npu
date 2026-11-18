#include <c10/util/string_view.h>
#include <cstdint>
#include <torch/library.h>
#include <tuple>

namespace sglang::npu_kernel {
std::tuple<int64_t, int64_t> shm_allocator_create_and_register(int64_t size, int64_t device_id, c10::string_view name);
void shm_allocator_free_all(int64_t device_id);
}  // namespace sglang::npu_kernel

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("shm_allocator_create_and_register(int size, int device_id, str name) -> (int, int)",
          &sglang::npu_kernel::shm_allocator_create_and_register);
    m.def("shm_allocator_free_all(int device_id) -> ()", &sglang::npu_kernel::shm_allocator_free_all);
}
