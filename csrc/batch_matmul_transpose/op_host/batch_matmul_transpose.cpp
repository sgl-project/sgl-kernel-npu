#include <iostream>
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
    {"nd", 0},
    {"nz", 1},
};

std::unordered_map<c10::ScalarType, TensorDType> atType2tensorDType = {
    {at::ScalarType::BFloat16, TensorDType::TENSOR_DTYPE_BF16},
    {at::ScalarType::Half, TensorDType::TENSOR_DTYPE_FLOAT16}
};

// batch size -> memory index
constexpr uint32_t MAX_CAPTURE_NUM = 1024;

template<typename MapType>
inline int GetModeVal(const MapType& mode_map,
                       c10::optional<c10::string_view> mode_opt,
                       c10::string_view default_mode,
                       const char* mode_name)
{
    std::string modeStr(mode_name);
    c10::string_view mode_str = mode_opt.value_or(default_mode);
    auto it = mode_map.find(mode_str);
    // if input mode is unsupported, use default value
    TORCH_CHECK(it != mode_map.end(), modeStr, c10::str(": Unsupported mode value ", mode_str));
    return it->second;
}

HOST_API void batch_matmul_transpose(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
    c10::optional<c10::string_view> format_mode, c10::optional<c10::string_view> quant_mode)
{
    auto tensorAShape = tensor_a.sizes();
    auto tensorBShape = tensor_b.sizes();
    auto tensorCShape = tensor_c.sizes();
    uint32_t n;
    uint32_t block_dim;
    HardwareInfo hwInfo;
    std::map<c10::ScalarType, float> dTypeMap = {
        {at::ScalarType::Half, 2.0},
        {at::ScalarType::BFloat16, 2.0}};

    at::ScalarType aType = tensor_a.scalar_type();
    at::ScalarType bType = tensor_b.scalar_type();
    at::ScalarType cType = tensor_c.scalar_type();
    TORCH_CHECK(aType == bType && bType == cType, "tensor type is not the same");
    TORCH_CHECK((aType == at::ScalarType::BFloat16) || (aType == at::ScalarType::Half), "tensor type only support half or bf16");

    TensorFormat formatMode = static_cast<TensorFormat>(GetModeVal(formatModeMap, format_mode, "nd", "format_mode"));
    MatMul::QuantMode quantMode = static_cast<MatMul::QuantMode>(GetModeVal(quantModeMap, quant_mode, "per_channel_symm", "quant_mode"));

    TORCH_CHECK(tensorAShape.size() == 3, "batch size is not same between srcTensor and dstTensor");
    if (formatMode == TensorFormat::TENSOR_FORMAT_ND) {
        TORCH_CHECK(tensorBShape.size() == 3, "tensor shape should be dim3 in nd format");
        TORCH_CHECK(tensorAShape[2] == tensorBShape[1], "tensor shape is wrong");
        n = tensorBShape[2];
    } else {
        TORCH_CHECK(tensorBShape.size() == 4, "tensor shape should be dim4 in nz format");
        TORCH_CHECK(tensorAShape[2] == tensorBShape[2], "tensor shape is wrong");
        n = tensorBShape[1] * tensorBShape[3];
    }
    TORCH_CHECK(tensorAShape[1] == tensorBShape[0], "tensor shape is wrong");

    OpShape opShape = {
        .batchSize = static_cast<uint32_t>(tensorAShape[1]),
        .m = static_cast<uint32_t>(tensorAShape[0]),
        .k = static_cast<uint32_t>(tensorAShape[2]),
        .n = n
    };
    PpMatmulTilingData matmulTilingData = {
        .opShape = opShape,
    };
    auto dType = atType2tensorDType[aType];
    MatMulInfo mmInfo = {
        .batchSize = opShape.batchSize,
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
        .quantMode = quantMode
    };
    GetPpMatmulTiling(mmInfo, hwInfo, block_dim, matmulTilingData);
    host_utils::PpMatmulTilingCheck(matmulTilingData);

    // tiling
    int32_t batchIdx = opShape.m - 1;
    uint32_t tilingSize = sizeof(PpMatmulTilingData);
    static auto global_tiling_data = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                        at::TensorOptions().dtype(at::kByte).device(tensor_a.options().device()));
    if (batchIdx >= 0 && batchIdx < MAX_CAPTURE_NUM) {
        aclrtMemcpy(global_tiling_data.data_ptr<uint8_t>() + (tilingSize * batchIdx), tilingSize,
                    &matmulTilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
     } else {
        // Handle the case where batchIdx is out of range
        TORCH_CHECK(false, "batchIdx is out of range: ", batchIdx);
    }
    at::Tensor tiling_tensor = at::from_blob(global_tiling_data.data_ptr<uint8_t>() + (tilingSize * batchIdx),
                                                tilingSize, at::kByte);

    EXEC_KERNEL_CMD(batch_matmul_transpose, block_dim, tensor_a, tensor_b, tensor_c, tiling_tensor);
}

} // namespace npu_kernel

} // namespace sglang
