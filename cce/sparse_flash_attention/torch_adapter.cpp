// Torch adapter for the CCE sparse flash attention operator, exposing
// torch.ops.cce.sparse_flash_attention. The kernel launches through the host
// dispatch archived in libsgl_host_operators.a.
#ifdef SGL_CPP_CCE_SPARSE_FLASH_ATTENTION

#include <acl/acl.h>
#include <cmath>
#include <tuple>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

extern "C" aclError cce_launch_sfa(void* q, void* qr, void* kv, void* kr, void* ids,
    void* out, void* maximum, void* sum, int heads, int queries, int groups, aclrtStream stream);

namespace {
void check_input(const at::Tensor& tensor, c10::Device device, at::ScalarType dtype,
                 at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(tensor.device() == device, name, " must be on the query NPU");
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.sizes() == shape,
                name, " has an unsupported dtype or shape");
    auto format = at_npu::native::get_npu_format(tensor);
    TORCH_CHECK(tensor.is_contiguous() && (format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW),
                name, " must use contiguous base-format storage");
    TORCH_CHECK(!tensor.requires_grad(), "cce::sparse_flash_attention is forward-only");
    const auto alignment = dtype == at::kBFloat16 ? 32 : alignof(int32_t);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
                name, " has an unaligned base address");
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_flash_attention_npu(
    const at::Tensor& query, const at::Tensor& queryRope, const at::Tensor& latent,
    const at::Tensor& keyRope, const at::Tensor& indices, double scale) {
    const auto device = query.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "expected NPU tensors");
    TORCH_CHECK(query.dim() == 4 && query.size(0) == 1 && query.size(1) > 0 && query.size(1) <= 8192,
                "query must have shape [1,SQ,H,512], 1<=SQ<=8192");
    const int queries = query.size(1), heads = query.size(2);
    TORCH_CHECK(heads == 16 || heads == 32 || heads == 64 || heads == 128,
                "heads must be 16/32/64/128");
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale - 1.0 / 24.0) < 1e-9,
                "this kernel requires scale=1/24");
    check_input(query, device, at::kBFloat16, {1, queries, heads, 512}, "query");
    check_input(queryRope, device, at::kBFloat16, {1, queries, heads, 64}, "query_rope");
    check_input(latent, device, at::kBFloat16, {1, 8192, 1, 512}, "kv_latent");
    check_input(keyRope, device, at::kBFloat16, {1, 8192, 1, 64}, "key_rope");
    check_input(indices, device, at::kInt, {1, queries, 1, 2048}, "sparse_indices");
    const c10::DeviceGuard guard(device);
    int64_t groups = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(device.index(), ACL_DEV_ATTR_CUBE_CORE_NUM, &groups) == ACL_SUCCESS
                    && groups > 0,
                "could not query Cube core count");
    auto output = at::empty_like(query);
    auto maximum = at::empty({1, 1, queries, heads}, query.options().dtype(at::kFloat));
    auto sum = at::empty_like(maximum);
    const auto npuStream = c10_npu::getCurrentNPUStream(device.index());
    const auto stream = npuStream.stream();
    TORCH_CHECK(stream != nullptr, "could not acquire the current NPU stream");
    for (const auto* input : {&query, &queryRope, &latent, &keyRope, &indices})
        c10_npu::NPUCachingAllocator::recordStream(input->storage().data_ptr(), npuStream);
    const auto status = cce_launch_sfa(query.data_ptr(), queryRope.data_ptr(),
        latent.data_ptr(), keyRope.data_ptr(), indices.data_ptr(), output.data_ptr(),
        maximum.data_ptr(), sum.data_ptr(), heads, queries, groups, stream);
    TORCH_CHECK(status == ACL_SUCCESS, "sparse flash attention kernel launch failed: ", status);
    return {output, maximum, sum};
}
}  // namespace

TORCH_LIBRARY(cce, m) {
    m.def("sparse_flash_attention(Tensor query, Tensor query_rope, Tensor kv_latent, "
          "Tensor key_rope, Tensor sparse_indices, float scale=0.041666666666666664) "
          "-> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(cce, PrivateUse1, m) {
    m.impl("sparse_flash_attention", &sparse_flash_attention_npu);
}

#endif  // SGL_CPP_CCE_SPARSE_FLASH_ATTENTION
