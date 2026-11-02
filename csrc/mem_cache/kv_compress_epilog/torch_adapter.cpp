#include "aclnn_runtime.h"
#include "aclnn_sgl_kv_compress_epilog.h"
#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>

namespace {
void kv_compress_epilog(at::Tensor cache, at::Tensor x, at::Tensor slot_mapping,
                        int64_t quant_group_size, int64_t quant_mode, bool round_scale_flag, int64_t layout)
{
    const c10::DeviceGuard deviceGuard(cache.device());
    if (cache.dim() == 4) cache = cache.squeeze(2);
    static auto query = reinterpret_cast<decltype(&aclnnSglKvCompressEpilogGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglKvCompressEpilogGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglKvCompressEpilog)>(
        sgl_cann::op_api_symbol("aclnnSglKvCompressEpilog"));
    std::array<at::Tensor, 3> tensors = {cache, x, slot_mapping};
    std::array<std::shared_ptr<aclTensor>, 3> descriptors = {
        sgl_cann::tensor_descriptor(cache, cache.scalar_type() == at::kFloat8_e4m3fn ? ACL_FLOAT8_E4M3FN : ACL_FLOAT8_E5M2),
        sgl_cann::tensor_descriptor(x, ACL_BF16),
        sgl_cann::tensor_descriptor(slot_mapping, slot_mapping.scalar_type() == at::kLong ? ACL_INT64 : ACL_INT32),
    };
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(),
                        quant_group_size, quant_mode, round_scale_flag, layout, cache.stride(0),
                        &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "KvCompressEpilog workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, cache.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors) tensor.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "KvCompressEpilog execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglKvCompressEpilog", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("kv_compress_epilog(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, "
          "int quant_group_size, int quant_mode, bool round_scale_flag, int layout) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("kv_compress_epilog", &kv_compress_epilog);
}
