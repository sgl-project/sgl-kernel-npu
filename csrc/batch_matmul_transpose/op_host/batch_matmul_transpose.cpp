#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_data.h"
#include "defines.h"
#include "torch_helper.h"
#include "common_tiling.h"
#include "aclrtlaunch_batch_matmul_transpose.h"

namespace sglang {
namespace npu_kernel {
using namespace pp_matmul;

std::unordered_map<c10::string_view, uint16_t> quantModeMap = {
    {"per_channel_symm", 0},
    {"per_channel_asymm", 1},
    {"per_token_symm", 2},
};

std::unordered_map<c10::string_view, uint16_t> formatModeMap = {
    {"ND", 0},
    {"NZ", 1},
};

std::unordered_map<c10::ScalarType, TensorDType> atType2tensorDType = {
    {at::ScalarType::BFloat16, TensorDType::TENSOR_DTYPE_BF16},
    {at::ScalarType::Half, TensorDType::TENSOR_DTYPE_FLOAT16}};

constexpr uint32_t MAX_M = 1024;

template <typename MapType>
inline int GetModeVal(const MapType &mode_map, c10::optional<c10::string_view> mode_opt, c10::string_view default_mode,
                      const char *mode_name)
{
    std::string modeStr(mode_name);
    c10::string_view mode_str = mode_opt.value_or(default_mode);
    auto it = mode_map.find(mode_str);
    // if input mode is unsupported, use default value
    TORCH_CHECK(it != mode_map.end(), modeStr, c10::str(": Unsupported mode value ", mode_str));
    return it->second;
}

HOST_API void batch_matmul_transpose(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
                                     c10::optional<c10::string_view> format_mode,
                                     c10::optional<c10::string_view> quant_mode)
{
    auto tensorAShape = tensor_a.sizes();
    auto tensorBShape = tensor_b.sizes();
    auto tensorCShape = tensor_c.sizes();
    uint32_t n;
    uint32_t block_dim;
    HardwareInfo hwInfo;
    std::map<c10::ScalarType, float> dTypeMap = {{at::ScalarType::Half, 2.0}, {at::ScalarType::BFloat16, 2.0}};

    at::ScalarType aType = tensor_a.scalar_type();
    at::ScalarType bType = tensor_b.scalar_type();
    at::ScalarType cType = tensor_c.scalar_type();
    TORCH_CHECK(aType == bType && bType == cType, "tensor type is not the same");
    TORCH_CHECK((aType == at::ScalarType::BFloat16) || (aType == at::ScalarType::Half),
                "tensor type only support half or bf16");

    TensorFormat formatMode = static_cast<TensorFormat>(GetModeVal(formatModeMap, format_mode, "ND", "format_mode"));
    MatMul::QuantMode quantMode =
        static_cast<MatMul::QuantMode>(GetModeVal(quantModeMap, quant_mode, "per_channel_symm", "quant_mode"));

    TORCH_CHECK(tensorAShape.size() == 3, "batch size is not same between srcTensor and dstTensor");
    if (formatMode == TensorFormat::TENSOR_FORMAT_ND) {
        TORCH_CHECK(tensorBShape.size() == 3, "tensor shape should be dim3 in ND format");
        TORCH_CHECK(tensorAShape[2] == tensorBShape[1], "tensor shape is wrong");
        n = tensorBShape[2];
    } else {
        TORCH_CHECK(tensorBShape.size() == 4, "tensor shape should be dim4 in nz format");
        TORCH_CHECK(tensorAShape[2] == tensorBShape[2], "tensor shape is wrong");
        n = tensorBShape[1] * tensorBShape[3];
    }
    TORCH_CHECK(tensorAShape[1] == tensorBShape[0], "tensor shape is wrong");

    OpShape opShape = {.batchSize = static_cast<uint32_t>(tensorAShape[1]),
                       .m = static_cast<uint32_t>(tensorAShape[0]),
                       .k = static_cast<uint32_t>(tensorAShape[2]),
                       .n = n};
    PpMatmulTilingData matmulTilingData = {
        .opShape = opShape,
    };
    auto dType = atType2tensorDType[aType];
    MatMulInfo mmInfo = {.batchSize = opShape.batchSize,
                         .m = opShape.m,
                         .k = opShape.k,
                         .n = opShape.n,
                         .dtypeA = dType,
                         .dtypeB = dType,
                         .dtypeC = dType,
                         .formatB = formatMode,
                         .mmType = MatMul::MatMulType::MATMUL_EIN_SUM,
                         .inDtype = dTypeMap[aType],
                         .outDtype = dTypeMap[cType],
                         .quantMode = quantMode};
    GetPpMatmulTiling(mmInfo, hwInfo, block_dim, matmulTilingData);
    host_utils::PpMatmulTilingCheck(matmulTilingData);

    TORCH_CHECK(opShape.m > 0 && opShape.m <= MAX_M, "m must be in [1, ", MAX_M, "], got ", opShape.m);

    // Captured graphs keep this address. Never overwrite a live configuration.
    // PpMatmulTilingData is packed and value-initialized, so its complete bytes
    // distinguish every kernel setting without hash collisions.
    constexpr size_t tilingSize = sizeof(PpMatmulTilingData);
    std::array<uint8_t, tilingSize> tilingBytes;
    std::memcpy(tilingBytes.data(), &matmulTilingData, tilingSize);
    auto key = std::make_pair(tensor_a.get_device(), tilingBytes);
    // Retain graph-pool tensors for the process lifetime. Destroying them from
    // a C++ static destructor can call the NPU allocator after Python shuts down.
    static auto &tilingCache = *new std::map<decltype(key), std::pair<at::Tensor, at::Tensor>>;
    static std::mutex tilingMutex;
    at::Tensor tiling_tensor;
    at::Tensor host_tiling;
    {
        std::lock_guard<std::mutex> lock(tilingMutex);
        auto entry = tilingCache.find(key);
        if (entry == tilingCache.end()) {
            auto hostTiling = at::empty({static_cast<int64_t>(tilingSize)}, at::kByte).pin_memory();
            std::memcpy(hostTiling.data_ptr(), &matmulTilingData, tilingSize);
            auto deviceTiling = at::empty({static_cast<int64_t>(tilingSize)}, tensor_a.options().dtype(at::kByte));
            entry = tilingCache.emplace(key, std::make_pair(hostTiling, deviceTiling)).first;
        }
        host_tiling = entry->second.first;
        tiling_tensor = entry->second.second;
    }
    // Dispatch the immutable data on this call's stream, including first-use
    // capture and calls made before another graph has replayed its initial copy.
    tiling_tensor.copy_(host_tiling, true);

    EXEC_KERNEL_CMD(batch_matmul_transpose, block_dim, tensor_a, tensor_b, tensor_c, tiling_tensor);
}

}  // namespace npu_kernel

}  // namespace sglang
