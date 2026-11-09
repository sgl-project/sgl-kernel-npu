#include "aclnn_runtime.h"
#include "aclnn_sgl_kv_compress_epilog.h"
#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>

namespace {
void kv_compress_epilog(at::Tensor kv_compress_cache, at::Tensor x, at::Tensor slot_mapping, int64_t quant_group_size,
                        int64_t quant_mode, bool round_scale_flag, int64_t layout)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D tensor, but got dimensions: ", x.dim());
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x dimensions must be positive, but got: [", x.size(0), ", ", x.size(1),
                "]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16, but got ", x.scalar_type());
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D tensor, but got dimensions: ", slot_mapping.dim());
    TORCH_CHECK(slot_mapping.size(0) == x.size(0),
                "slot_mapping size must equal x's first dimension, but got slot_mapping_size=", slot_mapping.size(0),
                ", x.dim(0)=", x.size(0));
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt || slot_mapping.scalar_type() == at::kLong,
                "slot_mapping must be INT32 or INT64, but got ", slot_mapping.scalar_type());
    TORCH_CHECK(
        kv_compress_cache.scalar_type() == at::kFloat8_e5m2 || kv_compress_cache.scalar_type() == at::kFloat8_e4m3fn,
        "kv_compress_cache must be FP8_E5M2 or FP8_E4M3FN, but got ", kv_compress_cache.scalar_type());

    at::Tensor cache = kv_compress_cache;
    if (cache.dim() == 4) {
        TORCH_CHECK(cache.size(2) == 1, "kv_compress_cache 4D tensor requires headnum (dim 2) == 1, but got ",
                    cache.size(2));
        cache = cache.squeeze(2);
    }

    TORCH_CHECK(quant_mode == 1 || quant_mode == 2, "quant_mode must be 1 (FP8) or 2 (MXFP8), got ", quant_mode);
    TORCH_CHECK(layout == 1 || layout == 2, "layout must be 1 or 2, got ", layout);
    if (quant_mode == 2) {
        TORCH_CHECK(quant_group_size == 64 || quant_group_size == 128, "MXFP8 quant_group_size must be 64 or 128, got ",
                    quant_group_size);
    }
    const int64_t quantCol = x.size(1) - 64;
    const int64_t group = quant_mode == 2 ? quant_group_size : 128;
    const int64_t scaleCol = (quantCol + group - 1) / group;
    if (layout == 2) {
        TORCH_CHECK(quant_mode == 2, "layout=2 only supports MXFP8 (quant_mode=2), but got quant_mode=", quant_mode);
        TORCH_CHECK(cache.dim() >= 2, "layout=2 requires kv_compress_cache to be at least 2D, got dim ", cache.dim());
        const int64_t blockSize = cache.size(1);
        TORCH_CHECK(blockSize > 0, "blockSize must be positive, got ", blockSize);
        const int64_t autoBlockStride = blockSize * (quantCol + 128 + (scaleCol + 7) / 8 * 8);
        if (cache.stride(0) > 0) {
            TORCH_CHECK(cache.stride(0) >= autoBlockStride, "block_stride (", cache.stride(0),
                        ") must be >= auto-computed stride (", autoBlockStride, ")");
        }
    } else {
        const int64_t concatCol = quantCol + 128 + scaleCol * (quant_mode == 2 ? 1 : 4);
        const int64_t autoRowStride = (concatCol + 127) / 128 * 128;
        if (cache.stride(0) > 0) {
            TORCH_CHECK(cache.stride(0) >= autoRowStride, "block_stride (", cache.stride(0),
                        ") must be >= layout=1 row width (", autoRowStride, ")");
        }
    }
    const c10::DeviceGuard deviceGuard(cache.device());
    static auto query = reinterpret_cast<decltype(&aclnnSglKvCompressEpilogGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglKvCompressEpilogGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglKvCompressEpilog)>(sgl_cann::op_api_symbol("aclnnSglKvCompressEpilog"));
    std::array<at::Tensor, 3> tensors = {cache, x, slot_mapping};
    std::array<std::shared_ptr<aclTensor>, 3> descriptors = {
        sgl_cann::tensor_descriptor(cache,
                                    cache.scalar_type() == at::kFloat8_e4m3fn ? ACL_FLOAT8_E4M3FN : ACL_FLOAT8_E5M2),
        sgl_cann::tensor_descriptor(x, ACL_BF16),
        sgl_cann::tensor_descriptor(slot_mapping, slot_mapping.scalar_type() == at::kLong ? ACL_INT64 : ACL_INT32),
    };
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), quant_group_size, quant_mode,
                        round_scale_flag, layout, cache.stride(0), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "KvCompressEpilog workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, cache.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
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
    m.def(
        "kv_compress_epilog(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, "
        "int quant_group_size, int quant_mode, bool round_scale_flag, int layout) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("kv_compress_epilog", &kv_compress_epilog);
}
