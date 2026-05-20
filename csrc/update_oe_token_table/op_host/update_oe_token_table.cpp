#include "update_oe_token_table_tiling.h"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include "aclrtlaunch_update_oe_token_table.h"
#include "common.h"
#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang::npu_kernel {
namespace {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024;
constexpr uint32_t MAX_CAPTURE_NUM = 1024U;

uint32_t actualCaptureNum = 0U;
static std::unordered_map<uint64_t, uint32_t> captureMap;
static at::Tensor globalTilingBuffer;
static c10::DeviceIndex cachedDeviceIndex = -1;

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }
    return x;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type CeilAlign(T x, T align)
{
    return CeilDiv(x, align) * align;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type FloorAlign(T x, T align)
{
    return align == 0 ? 0 : x / align * align;
}

void CheckUpdateOeTokenTableInputs(const at::Tensor &tokens, const at::Tensor &req_lens,
                                   const at::Tensor &row_indices, const at::Tensor &column_starts,
                                   const at::Tensor &ignore_tokens, const at::Tensor &oe_token_table,
                                   int64_t batch_size, int64_t max_context_len)
{
    TORCH_CHECK(tokens.scalar_type() == at::kInt, "tokens must be int32");
    TORCH_CHECK(req_lens.scalar_type() == at::kInt, "req_lens must be int32");
    TORCH_CHECK(row_indices.scalar_type() == at::kLong, "row_indices must be int64");
    TORCH_CHECK(column_starts.scalar_type() == at::kInt, "column_starts must be int32");
    TORCH_CHECK(ignore_tokens.scalar_type() == at::kInt, "ignore_tokens must be int32");
    TORCH_CHECK(oe_token_table.scalar_type() == at::kInt, "oe_token_table must be int32");

    TORCH_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
    TORCH_CHECK(req_lens.is_contiguous(), "req_lens must be contiguous");
    TORCH_CHECK(row_indices.is_contiguous(), "row_indices must be contiguous");
    TORCH_CHECK(column_starts.is_contiguous(), "column_starts must be contiguous");
    TORCH_CHECK(ignore_tokens.is_contiguous(), "ignore_tokens must be contiguous");
    TORCH_CHECK(oe_token_table.is_contiguous(), "oe_token_table must be contiguous");

    TORCH_CHECK(tokens.dim() == 1, "tokens must be a 1D tensor");
    TORCH_CHECK(req_lens.dim() == 1, "req_lens must be a 1D tensor");
    TORCH_CHECK(row_indices.dim() == 1, "row_indices must be a 1D tensor");
    TORCH_CHECK(column_starts.dim() == 1, "column_starts must be a 1D tensor");
    TORCH_CHECK(ignore_tokens.dim() == 1, "ignore_tokens must be a 1D tensor");
    TORCH_CHECK(oe_token_table.dim() == 2, "oe_token_table must be a 2D tensor");

    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(max_context_len > 0, "max_context_len must be positive");
    TORCH_CHECK(req_lens.numel() == batch_size, "req_lens length must match batch_size");
    TORCH_CHECK(row_indices.numel() == batch_size, "row_indices length must match batch_size");
    TORCH_CHECK(column_starts.numel() == batch_size, "column_starts length must match batch_size");
    TORCH_CHECK(oe_token_table.size(1) >= max_context_len,
                "oe_token_table second dimension must be >= max_context_len");
    TORCH_CHECK(tokens.device() == req_lens.device(), "tokens and req_lens must be on the same device");
    TORCH_CHECK(tokens.device() == row_indices.device(), "tokens and row_indices must be on the same device");
    TORCH_CHECK(tokens.device() == column_starts.device(),
                "tokens and column_starts must be on the same device");
    TORCH_CHECK(tokens.device() == ignore_tokens.device(), "tokens and ignore_tokens must be on the same device");
    TORCH_CHECK(tokens.device() == oe_token_table.device(),
                "tokens and oe_token_table must be on the same device");
}

UpdateOeTokenTableTilingData BuildTilingData(uint32_t &block_dim, int64_t batch_size, int64_t max_context_len,
                                             int64_t ignore_token_num)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aiv_num = static_cast<uint32_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ub_size = 0;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    uint32_t block_factor = CeilDiv(static_cast<uint32_t>(batch_size), aiv_num);
    uint32_t used_core_num = CeilDiv(static_cast<uint32_t>(batch_size), block_factor);
    block_factor = CeilDiv(static_cast<uint32_t>(batch_size), used_core_num);
    uint32_t tail_block_factor =
        batch_size % used_core_num == 0 ? block_factor : static_cast<uint32_t>(batch_size % block_factor);

    int64_t ignore_ub_size = CeilAlign(ignore_token_num * static_cast<int64_t>(sizeof(int32_t)), BLOCK_SIZE);
    int64_t req_len_ub_size = CeilAlign(batch_size * static_cast<int64_t>(sizeof(int32_t)), BLOCK_SIZE);
    int64_t reserve_ub_size = static_cast<int64_t>(ub_size) - ignore_ub_size - req_len_ub_size;
    TORCH_CHECK(reserve_ub_size > 0, "insufficient UB size for update_oe_token_table");
    int32_t ub_factor =
        static_cast<int32_t>(FloorAlign(reserve_ub_size / (DOUBLE_BUFFER * 4), BLOCK_SIZE) / sizeof(int32_t));
    TORCH_CHECK(ub_factor > 0, "computed ubFactor must be positive");

    block_dim = used_core_num;

    UpdateOeTokenTableTilingData tiling_data;
    tiling_data.usedCoreNum = used_core_num;
    tiling_data.blockFactor = block_factor;
    tiling_data.tailBlockFactor = tail_block_factor;
    tiling_data.ubFactor = static_cast<uint32_t>(ub_factor);
    tiling_data.batchSize = static_cast<uint32_t>(batch_size);
    tiling_data.maxContextLen = static_cast<uint32_t>(max_context_len);
    tiling_data.ignoreTokenNum = static_cast<uint32_t>(ignore_token_num);

    return tiling_data;
}

at::Tensor GetOrCreateCachedTilingTensor(const UpdateOeTokenTableTilingData &tiling_data,
                                         const at::Tensor &device_anchor)
{
    int32_t tiling_size = static_cast<int32_t>(
        ((sizeof(UpdateOeTokenTableTilingData) + PADDING_BYTE - 1) / PADDING_BYTE) * PADDING_BYTE);
    auto hash_key = std::make_tuple(tiling_data.usedCoreNum, tiling_data.blockFactor, tiling_data.tailBlockFactor,
                                    tiling_data.ubFactor, tiling_data.batchSize, tiling_data.maxContextLen,
                                    tiling_data.ignoreTokenNum);
    uint64_t hash_value = host_utils::TupleHasher::Hash(hash_key);
    c10::DeviceIndex device_index = device_anchor.get_device();

    if (!globalTilingBuffer.defined() || cachedDeviceIndex != device_index) {
        globalTilingBuffer =
            at::empty({static_cast<int64_t>(tiling_size) * MAX_CAPTURE_NUM},
                      at::TensorOptions().dtype(at::kByte).device(device_anchor.device()));
        captureMap.clear();
        actualCaptureNum = 0U;
        cachedDeviceIndex = device_index;
    }

    auto it = captureMap.find(hash_value);
    if (it != captureMap.end()) {
        return at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + static_cast<int64_t>(tiling_size) * it->second,
                             tiling_size, at::kByte);
    }

    auto tiling_cpu = at::zeros({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &tiling_data, sizeof(UpdateOeTokenTableTilingData));
    auto device_tiling = TorchNpuHelper::CopyTensorHostToDevice(tiling_cpu).view({static_cast<int64_t>(tiling_size)});

    if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        return device_tiling;
    }

    uint32_t slot = actualCaptureNum++;
    captureMap.emplace(hash_value, slot);
    globalTilingBuffer
        .slice(0, static_cast<int64_t>(slot) * tiling_size, static_cast<int64_t>(slot + 1) * tiling_size)
        .copy_(device_tiling);
    return at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + static_cast<int64_t>(tiling_size) * slot,
                         tiling_size, at::kByte);
}

}  // namespace

HOST_API at::Tensor update_oe_token_table(const at::Tensor &tokens, const at::Tensor &req_lens,
                                          const at::Tensor &row_indices, const at::Tensor &column_starts,
                                          const at::Tensor &ignore_tokens, int64_t batch_size,
                                          int64_t max_context_len, const at::Tensor &oe_token_table)
{
    CheckUpdateOeTokenTableInputs(tokens, req_lens, row_indices, column_starts, ignore_tokens,
                                  oe_token_table, batch_size, max_context_len);

    uint32_t block_dim = 1;
    auto tiling_data = BuildTilingData(block_dim, batch_size, max_context_len, ignore_tokens.numel());
    auto tiling_tensor = GetOrCreateCachedTilingTensor(tiling_data, oe_token_table);
    auto workspace_tensor =
        at::empty({WORKSPACE_SIZE}, at::TensorOptions().dtype(at::kByte).device(oe_token_table.device()));

    EXEC_KERNEL_CMD(update_oe_token_table, block_dim, tokens, req_lens, row_indices, column_starts,
                    ignore_tokens, oe_token_table, workspace_tensor, tiling_tensor);
    return oe_token_table;
}

}  // namespace sglang::npu_kernel
