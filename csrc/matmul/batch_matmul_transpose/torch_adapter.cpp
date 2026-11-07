#include "aclnn_runtime.h"
#include "aclnn_sgl_batch_matmul_transpose.h"
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <array>
#include <limits>
#include <string>

namespace {
void batch_matmul_transpose(const at::Tensor &a, const at::Tensor &b, at::Tensor &c,
                            c10::optional<c10::string_view> format_mode, c10::optional<c10::string_view> quant_mode)
{
    std::string format = format_mode ? std::string(format_mode->data(), format_mode->size()) : "ND";
    std::string quant = quant_mode ? std::string(quant_mode->data(), quant_mode->size()) : "per_channel_symm";
    TORCH_CHECK(format == "ND" || format == "NZ", "format_mode must be ND or NZ, got ", format);
    TORCH_CHECK(quant == "per_channel_symm" || quant == "per_channel_asymm" || quant == "per_token_symm",
                "Unsupported quant_mode: ", quant);
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "tensor_a must be on an NPU");
    TORCH_CHECK(b.device() == a.device() && c.device() == a.device(), "all tensors must be on the same NPU");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
                "tensor type only supports float16 or bfloat16");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(),
                "all tensors must have the same dtype");
    TORCH_CHECK(a.dim() == 3 && c.dim() == 3, "tensor_a and tensor_c must be 3D");
    TORCH_CHECK(b.dim() == (format == "NZ" ? 4 : 3), "tensor_b has the wrong rank for ", format);
    for (const auto &tensor : {a, b, c}) {
        TORCH_CHECK(tensor.is_contiguous(), "all tensors must be contiguous");
        TORCH_CHECK(tensor.numel() > 0 && tensor.numel() <= std::numeric_limits<uint32_t>::max(),
                    "tensor element counts must be in [1, UINT32_MAX]");
    }
    const int64_t m = a.size(0), batch = a.size(1), k = a.size(2);
    TORCH_CHECK(m <= 1024, "M must be <= 1024, got ", m);
    const bool nz = format == "NZ";
    TORCH_CHECK(b.size(0) == batch && b.size(nz ? 2 : 1) == k, "tensor_a/tensor_b shapes do not match");
    TORCH_CHECK(!nz || (b.size(3) == 16 && k % 16 == 0), "NZ requires [B, N/16, K, 16] with K divisible by 16");
    const int64_t n = nz ? b.size(1) * 16 : b.size(2);
    TORCH_CHECK(c.size(0) == m && c.size(1) == batch && c.size(2) == n, "tensor_c must have shape [M, B, N]");
    at::assert_no_overlap(c, a);
    at::assert_no_overlap(c, b);
    const c10::DeviceGuard deviceGuard(a.device());

    static auto query = reinterpret_cast<decltype(&aclnnSglBatchMatmulTransposeGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglBatchMatmulTransposeGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglBatchMatmulTranspose)>(
        sgl_cann::op_api_symbol("aclnnSglBatchMatmulTranspose"));
    const auto dtype = a.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16;
    std::array<std::shared_ptr<aclTensor>, 3> descriptors = {sgl_cann::tensor_descriptor(a, dtype),
                                                             sgl_cann::tensor_descriptor(b, dtype),
                                                             sgl_cann::tensor_descriptor(c, dtype)};
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = query(descriptors[0].get(), descriptors[1].get(), format.data(), quant.data(),
                              descriptors[2].get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "BatchMatmulTranspose workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, a.options().dtype(at::kByte));
    const std::array<at::Tensor, 3> tensors = {a, b, c};
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    // Captures retain host-queue resources; record_stream covers asynchronous device use.
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "BatchMatmulTranspose execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglBatchMatmulTranspose", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "batch_matmul_transpose(Tensor tensor_a, Tensor tensor_b, Tensor(a!) tensor_c, "
        "str? format_mode=None, str? quant_mode=None) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("batch_matmul_transpose", &batch_matmul_transpose);
}
