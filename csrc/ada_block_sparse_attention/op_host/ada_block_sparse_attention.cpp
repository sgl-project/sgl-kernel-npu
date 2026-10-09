/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "ada_block_sparse_attention_tiling.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
// Load the SDK's GE/GERT definitions before torch_npu, which also ships GE
// headers with the same include guards. Mixing the two sets is not supported.
#include "graph/types.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"

#include "aclrtlaunch_ada_block_sparse_attention.h"
#include "defines.h"

namespace sglang::npu_kernel {
namespace {

inline int32_t checked_int(int64_t value, const char *name)
{
    TORCH_CHECK(value > 0 && value <= std::numeric_limits<int32_t>::max(), name, " is out of range");
    return static_cast<int32_t>(value);
}

struct InputShape {
    int64_t batch, heads, sequence, dim;
};

inline InputShape check_input(const at::Tensor &x, const std::string &layout, int64_t heads)
{
    TORCH_CHECK(x.device().type() == DEVICE_TYPE, "sparse attention inputs must be NPU tensors");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "sparse attention supports FP16 and BF16");
    checked_int(heads, "head count");
    for (auto size : x.sizes())
        checked_int(size, "input dimension");
    if (layout == "BNSD" || layout == "BSND") {
        TORCH_CHECK(x.dim() == 4, "BNSD and BSND inputs must have rank 4");
        TORCH_CHECK(x.size(layout == "BNSD" ? 1 : 2) == heads, "head count does not match input shape");
        return {x.size(0), heads, x.size(layout == "BNSD" ? 2 : 1), x.size(3)};
    }
    TORCH_CHECK(layout == "BSH" && x.dim() == 3, "supported layouts are BNSD, BSND and BSH");
    TORCH_CHECK(x.size(2) % heads == 0, "hidden size must be divisible by head count");
    return {x.size(0), heads, x.size(1), x.size(2) / heads};
}

inline void check_pair(const at::Tensor &q, const at::Tensor &k, const InputShape &qs, const InputShape &ks)
{
    TORCH_CHECK(q.device() == k.device() && q.scalar_type() == k.scalar_type(),
                "inputs must have the same device and dtype");
    int device = 0;
    c10_npu::GetDevice(&device);
    TORCH_CHECK(q.get_device() == device, "inputs must be on the current NPU device");
    TORCH_CHECK(qs.batch == ks.batch && qs.dim == ks.dim && qs.heads % ks.heads == 0,
                "incompatible query and key/value shapes");
}

inline at::Tensor normalize(const at::Tensor &x, const std::string &layout)
{
    auto result = x.contiguous();
    return layout == "BSND" ? result.view({x.size(0), x.size(1), x.size(2) * x.size(3)}) : result;
}

inline gert::StorageShape storage_shape(at::IntArrayRef sizes)
{
    gert::StorageShape shape;
    for (auto size : sizes) {
        shape.MutableOriginShape().AppendDim(size);
        shape.MutableStorageShape().AppendDim(size);
    }
    return shape;
}

inline at::IntArrayRef check_lengths(c10::OptionalIntArrayRef lengths, const InputShape &shape)
{
    auto values = lengths.value_or(at::IntArrayRef{});
    TORCH_CHECK(values.empty() || values.size() == 1 || values.size() == static_cast<size_t>(shape.batch),
                "actual sequence lengths must have one element or one per batch");
    for (auto value : values) {
        TORCH_CHECK(value > 0 && value <= shape.sequence, "actual sequence length is out of range");
    }
    return values;
}

inline gert::Tensor length_tensor(at::IntArrayRef values)
{
    const int64_t size = static_cast<int64_t>(values.size());
    gert::Tensor tensor(gert::StorageShape({size}, {size}), gert::StorageFormat(), ge::DT_INT64);
    tensor.SetData(gert::TensorData{const_cast<int64_t *>(values.data())});
    return tensor;
}

inline at::Tensor cached_device_copy(const at::Tensor &host)
{
    // Keep the host-to-device copies out of repeated graph captures. Include
    // the stream so that a cached copy is never consumed before it completes.
    static std::mutex mutex;
    static std::unordered_map<int, std::unordered_map<std::string, at::Tensor>> caches;
    int device = 0;
    c10_npu::GetDevice(&device);
    const auto stream = reinterpret_cast<uintptr_t>(c10_npu::getCurrentNPUStream().stream(false));
    std::string key(reinterpret_cast<const char *>(host.data_ptr()), host.nbytes());
    key.append(reinterpret_cast<const char *>(&stream), sizeof(stream));
    const auto dtype = host.scalar_type();
    key.append(reinterpret_cast<const char *>(&dtype), sizeof(dtype));
    std::lock_guard<std::mutex> lock(mutex);
    auto &cache = caches[device];
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    auto result = TorchNpuHelper::CopyTensorHostToDevice(host);
    if (cache.size() < 1024) cache.emplace(std::move(key), result);
    return result;
}

inline at::Tensor device_lengths(at::IntArrayRef values)
{
    if (values.empty()) return at::Tensor();
    auto host = at::tensor(values, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    return cached_device_copy(host);
}

inline void *data_or_null(const at::Tensor &tensor)
{
    return tensor.defined() ? tensor.data_ptr() : nullptr;
}

template <typename Tiling>
inline at::Tensor device_tiling(Tiling &tiling)
{
    const int64_t size = (tiling.GetDataSize() + 31) / 32 * 32;
    auto host = at::zeros({size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    tiling.SaveToBuffer(host.data_ptr(), size);
    return cached_device_copy(host);
}

inline platform_ascendc::PlatformAscendC *get_platform()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get the AscendC platform");
    const auto soc = platform->GetSocVersion();
    TORCH_CHECK(soc == platform_ascendc::SocVersion::ASCEND910B || soc == platform_ascendc::SocVersion::ASCEND910_93,
                "sparse attention is supported only on Ascend A2/A3");
    TORCH_CHECK(platform->GetCoreNumAic() > 0 && platform->GetCoreNumAiv() > 0 && platform->GetCoreNumAiv() <= 64,
                "unsupported sparse attention core count");
    return platform;
}

}  // namespace

HOST_API at::Tensor ada_block_sparse_attention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                               const at::Tensor &sparse_mask, const at::Tensor &sparse_count_table,
                                               std::string input_layout, int64_t sparse_size, int64_t num_heads,
                                               int64_t num_key_value_heads, double scale_value, bool causal,
                                               int64_t inner_precise, int64_t pre_tokens, int64_t next_tokens,
                                               c10::OptionalIntArrayRef actual_seq_lengths,
                                               c10::OptionalIntArrayRef actual_seq_lengths_kv)
{
    auto platform = get_platform();
    auto qs = check_input(query, input_layout, num_heads);
    auto ks = check_input(key, input_layout, num_key_value_heads);
    auto vs = check_input(value, input_layout, num_key_value_heads);
    check_pair(query, key, qs, ks);
    check_pair(query, value, qs, vs);
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
    TORCH_CHECK(std::isfinite(scale_value), "scale_value must be finite");
    int32_t sparseSize = checked_int(sparse_size, "sparse_size");
    TORCH_CHECK(sparseSize >= 128 && sparseSize <= 512 && sparseSize % 128 == 0,
                "sparse_size must be a multiple of 128 in [128, 512]");
    TORCH_CHECK(inner_precise >= 0 && inner_precise <= 3, "inner_precise must be in [0, 3]");
    TORCH_CHECK(pre_tokens >= 0 && pre_tokens <= INT32_MAX && next_tokens >= 0 && next_tokens <= INT32_MAX,
                "pre_tokens and next_tokens must fit in a nonnegative int32");
    const int64_t rows = (qs.sequence + sparse_size - 1) / sparse_size;
    const int64_t columns = ((ks.sequence + sparse_size - 1) / sparse_size + 31) / 32 * 32;
    TORCH_CHECK(sparse_mask.device() == query.device() && sparse_mask.scalar_type() == at::kChar &&
                    sparse_mask.sizes() == at::IntArrayRef({qs.batch, qs.heads, rows, columns}),
                "sparse_mask must be int8 [B, Nq, ceil(Sq/sparse_size), align32(ceil(Skv/sparse_size))]");
    TORCH_CHECK(sparse_count_table.device() == query.device() && sparse_count_table.scalar_type() == at::kInt &&
                    sparse_count_table.sizes() == at::IntArrayRef({qs.batch, qs.heads, rows}),
                "sparse_count_table must be int32 [B, Nq, ceil(Sq/sparse_size)]");
    auto qLengths = check_lengths(actual_seq_lengths, qs);
    auto kvLengths = check_lengths(actual_seq_lengths_kv, ks);
    auto qLengthTensor = length_tensor(qLengths);
    auto kvLengthTensor = length_tensor(kvLengths);
    auto q = normalize(query, input_layout);
    auto k = normalize(key, input_layout);
    auto v = normalize(value, input_layout);
    auto mask = sparse_mask.contiguous();
    auto count = sparse_count_table.contiguous();
    auto output = at::empty_like(q);
    auto qShape = storage_shape(q.sizes());
    auto kShape = storage_shape(k.sizes());
    auto vShape = storage_shape(v.sizes());
    auto maskShape = storage_shape(mask.sizes());
    if (input_layout == "BSND") input_layout = "BSH";

    optiling::AdaBlockSparseAttentionCompileInfo info{};
    info.aicNum = platform->GetCoreNumAic();
    info.aivNum = platform->GetCoreNumAiv();
    info.socShortName = platform->GetSocVersion();
    info.defaultSysWorkspaceSize = platform->GetLibApiWorkSpaceSize();
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, info.l0ASize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, info.l0BSize);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0CSize);
    int32_t heads = checked_int(num_heads, "num_heads");
    int32_t kvHeads = checked_int(num_key_value_heads, "num_key_value_heads");
    int32_t sparseMode = 0;
    uint8_t causalValue = causal;
    float scale = static_cast<float>(scale_value);
    size_t workspaceSize = 0;
    optiling::ContextParamsForBSATiling params{};
    params.opName = "AdaBlockSparseAttention";
    params.queryInputShape = params.outputShape = &qShape;
    params.keyInputShape = &kShape;
    params.valueInputShape = &vShape;
    params.sparseMaskShape = &maskShape;
    params.inputDataType = query.scalar_type() == at::kHalf ? ge::DT_FLOAT16 : ge::DT_BF16;
    params.kDataType = params.vDataType = params.outputDataType = params.inputDataType;
    params.pseShiftDataType = params.maskDataType = params.inputDataType;
    params.deqScaleType = params.deqScale2Type = params.inputDataType;
    params.quantScale2Type = params.quantOffset2Type = ge::DT_FLOAT;
    params.actualSeqenceLengthQ = qLengths.empty() ? nullptr : &qLengthTensor;
    params.actualSeqenceLengthKV = kvLengths.empty() ? nullptr : &kvLengthTensor;
    params.headsNumber = &heads;
    params.numKeyValueHeads = &kvHeads;
    params.sparseSize = &sparseSize;
    params.causal = &causalValue;
    params.sparseMode = &sparseMode;
    params.innerPrecisePtr = &inner_precise;
    params.preToken = &pre_tokens;
    params.nextToken = &next_tokens;
    params.scaleValue = &scale;
    params.layout = input_layout.c_str();
    params.workspaceSize = &workspaceSize;
    params.compileInfoPtr = &info;
    optiling::AdaBlockSparseAttentionTilingData tiling;
    tiling.promptAttentionBaseParams.set_causal(causalValue);
    tiling.promptAttentionBaseParams.set_sparseSize(sparseSize);
    tiling.promptAttentionBaseParams.set_sparseMaskS1(rows);
    tiling.promptAttentionBaseParams.set_sparseMaskS2(columns);
    optiling::AdaBlockSparseAttentionTiling tiler(nullptr);
    uint64_t tilingKey = 7;
    uint32_t blockDim = 0;
    TORCH_CHECK(tiler.RunBigKernelTilingWithParams(params, tilingKey, blockDim, tiling) == ge::GRAPH_SUCCESS,
                "AdaBlockSparseAttention tiling failed");
    tilingKey += 1000000000000000000ULL;
    switch (tilingKey) {
        case 1000000000000101012ULL:
        case 1000000000002101012ULL:
        case 1000000000000001012ULL:
        case 1000000000002001012ULL:
        case 1000000000000111112ULL:
        case 1000000000002111112ULL:
        case 1000000000000011112ULL:
        case 1000000000002011112ULL:
            break;
        default:
            TORCH_CHECK(false, "unsupported AdaBlockSparseAttention tiling key: ", tilingKey);
    }
    TORCH_CHECK(blockDim > 0 && workspaceSize <= static_cast<size_t>(INT64_MAX), "invalid launch configuration");
    auto tilingTensor = device_tiling(tiling);
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));
    auto qLengthsDevice = device_lengths(qLengths);
    auto kvLengthsDevice = device_lengths(kvLengths);
    void *qLengthsPtr = data_or_null(qLengthsDevice);
    void *kvLengthsPtr = data_or_null(kvLengthsDevice);
    EXEC_KERNEL_CMD(ada_block_sparse_attention, blockDim, q, k, v, qLengthsPtr, kvLengthsPtr, mask, count, output,
                    workspace, tilingTensor, tilingKey);
    return output.view(query.sizes());
}

}  // namespace sglang::npu_kernel
