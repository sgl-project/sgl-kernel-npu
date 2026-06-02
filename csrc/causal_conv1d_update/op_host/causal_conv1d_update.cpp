#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <functional>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/causal_conv1d_update_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "common_tiling.h"
#include "common.h"
#include "stub/aclrtlaunch_causal_conv1d_update_bfloat16_t.h"
#include "stub/aclrtlaunch_causal_conv1d_update_half.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;  // 对齐 lightning_indexer

// Graph mode tiling cache
uint32_t actualCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> captureMap;

// Helper struct for hashing tiling parameters
struct CausalConv1dUpdateTilingKey {
    int64_t batch;
    int64_t seqLen;
    int64_t dim;
    int64_t width;
    int64_t stateLen;
    int64_t hasIndices;
    int64_t hasBias;
    int64_t hasNumAccept;
    int64_t hasQueryLoc;
    int64_t activationMode;
    int64_t padSlotId;

    bool operator==(const CausalConv1dUpdateTilingKey &other) const
    {
        return batch == other.batch && seqLen == other.seqLen && dim == other.dim && width == other.width &&
               stateLen == other.stateLen && hasIndices == other.hasIndices && hasBias == other.hasBias &&
               hasNumAccept == other.hasNumAccept && hasQueryLoc == other.hasQueryLoc &&
               activationMode == other.activationMode && padSlotId == other.padSlotId;
    }
};

// Hash function for CausalConv1dUpdateTilingKey
struct CausalConv1dUpdateTilingKeyHash {
    std::size_t operator()(const CausalConv1dUpdateTilingKey &k) const
    {
        std::size_t h1 = std::hash<int64_t>{}(k.batch);
        std::size_t h2 = std::hash<int64_t>{}(k.seqLen);
        std::size_t h3 = std::hash<int64_t>{}(k.dim);
        std::size_t h4 = std::hash<int64_t>{}(k.width);
        std::size_t h5 = std::hash<int64_t>{}(k.stateLen);
        std::size_t h6 = std::hash<int64_t>{}(k.hasIndices);
        std::size_t h7 = std::hash<int64_t>{}(k.hasBias);
        std::size_t h8 = std::hash<int64_t>{}(k.hasNumAccept);
        std::size_t h9 = std::hash<int64_t>{}(k.hasQueryLoc);
        std::size_t h10 = std::hash<int64_t>{}(k.activationMode);
        std::size_t h11 = std::hash<int64_t>{}(k.padSlotId);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6) ^ (h8 << 7) ^ (h9 << 8) ^
               (h10 << 9) ^ (h11 << 10);
    }
};

HOST_API at::Tensor causal_conv1d_update_impl(const at::Tensor &x, const at::Tensor &weight,
                                              const at::Tensor &conv_state, const at::Tensor &conv_state_indices,
                                              const at::Tensor &bias, const at::Tensor &num_accepted_tokens,
                                              const at::Tensor &query_start_loc, bool activation_mode,
                                              int64_t pad_slot_id)
{
    // Input validation
    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor [batch, seq_len, dim], got shape ", x.sizes());
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D tensor [width, dim], got shape ", weight.sizes());
    TORCH_CHECK(conv_state.dim() == 3, "conv_state must be 3D tensor [cache_len, width-1, dim], got shape ",
                conv_state.sizes());

    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported, got ", dtype);
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_state.scalar_type() == dtype, "conv_state dtype must match x dtype");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous before entering the NPU kernel. Fix this in Python.");
    TORCH_CHECK(weight.is_contiguous(),
                "weight must be contiguous. Transposed weights are NOT allowed. Fix this in Python.");
    TORCH_CHECK(conv_state.is_contiguous(), "conv_state must be contiguous. Fix this in Python.");

    const int64_t batch = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t dim = x.size(2);
    const int64_t width = weight.size(0);
    const int64_t state_len = conv_state.size(1);

    // Check optional tensors
    const bool has_indices = conv_state_indices.numel() > 0;
    const bool has_bias = bias.numel() > 0;
    const bool has_num_accept = num_accepted_tokens.numel() > 0;
    const bool has_query_loc = query_start_loc.numel() > 0;

    // Create output tensor
    at::Tensor y = at::empty_like(x);

    // Pre-shift conv_state to maintain the rolling-buffer convention used by the
    // vLLM/SGLang reference. The kernel only writes positions [VAL .. state_len-1]
    // (where VAL = state_len - seq_len), leaving the older "scratch" prefix stale.
    // We shift state[0..VAL-1] := state[seq_len..seq_len+VAL-1] beforehand so that
    // after the kernel completes, the full conv_state matches the rolling window
    // [old[seq_len:state_len], x[0:seq_len]].
    const int64_t val_shift = state_len - seq_len;
    if (val_shift > 0) {
        if (has_indices) {
            // Map pad_slot_id (e.g. -1) to index 0 so that index_select stays
            // in-bounds.  Shifting a pad slot is harmless because its data is
            // meaningless, and duplicate writes to index 0 are all identical
            // (same source row, same shift), so the result is correct.
            auto cs_indices_long = conv_state_indices.to(at::kLong);
            auto safe_indices = cs_indices_long.clamp_min(0);
            auto cs_view = conv_state.index_select(0, safe_indices);  // [B, state_len, dim]
            auto src = cs_view.slice(1, seq_len, seq_len + val_shift).clone();
            cs_view.slice(1, 0, val_shift).copy_(src);
            conv_state.index_copy_(0, safe_indices, cs_view);
        } else {
            auto src = conv_state.slice(1, seq_len, seq_len + val_shift).clone();
            conv_state.slice(1, 0, val_shift).copy_(src);
        }
    }

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_aiv_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    int32_t block_dim = std::min(max_aiv_core, static_cast<int32_t>(batch));
    if (block_dim == 0) {
        block_dim = 1;
    }
    int32_t workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    // Per-tile launch helper. Captures everything needed except the tensors and the
    // current tile_dim, so we can call it once for the whole tensor or once per tile
    // when dim exceeds the kernel's UB budget.
    auto launch_for_tile = [&](const at::Tensor &x_t, const at::Tensor &weight_t, const at::Tensor &conv_state_t,
                               const at::Tensor &bias_t, at::Tensor &y_t, int64_t tile_dim) {
        // 1. Prepare Tiling Data Struct
        CausalConv1dUpdateTilingData tiling_data;
        SGLang::CausalConv1dUpdate::ComputeTilingData(batch, seq_len, tile_dim, width, state_len, has_indices, has_bias,
                                                      has_num_accept, has_query_loc, activation_mode, pad_slot_id,
                                                      block_dim, tiling_data);

        int32_t tilingSize = (sizeof(CausalConv1dUpdateTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
        at::Tensor tilingTensor;

        // 2. Hash computation
        CausalConv1dUpdateTilingKey key{.batch = batch,
                                        .seqLen = seq_len,
                                        .dim = tile_dim,
                                        .width = width,
                                        .stateLen = state_len,
                                        .hasIndices = has_indices ? 1 : 0,
                                        .hasBias = has_bias ? 1 : 0,
                                        .hasNumAccept = has_num_accept ? 1 : 0,
                                        .hasQueryLoc = has_query_loc ? 1 : 0,
                                        .activationMode = activation_mode ? 1 : 0,
                                        .padSlotId = pad_slot_id};
        uint64_t hashValue = CausalConv1dUpdateTilingKeyHash{}(key);

        // Helper: wrap tiling_data in a CPU tensor and copy to device via
        // torch_npu dispatch so the H2D transfer is properly ordered on the
        // same task queue as the preceding pre-shift ops and the subsequent
        // kernel launch.  Using raw aclrtMemcpy here would bypass OpCommand
        // and break stream ordering in graph / non-blocking mode.
        auto copyTilingToDevice = [&]() {
            auto cpuTiling = at::empty({tilingSize}, at::kByte);
            std::memcpy(cpuTiling.data_ptr(), &tiling_data, sizeof(CausalConv1dUpdateTilingData));
            return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
        };

        // 3. cache management
        static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                                   at::TensorOptions().dtype(at::kByte).device(x_t.options().device()));

        if (captureMap.find(hashValue) != captureMap.end()) {
            tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                         tilingSize, at::kByte);
        } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
            // Overflow: no room in cache, create a one-shot device copy.
            tilingTensor = copyTilingToDevice();
        } else {
            // New capture: copy to device and cache in globalTilingBuffer.
            captureMap[hashValue] = actualCaptureNum;
            auto deviceTiling = copyTilingToDevice();
            // D2D copy into the pre-allocated cache buffer (ordered via dispatch).
            globalTilingBuffer.slice(0, actualCaptureNum * tilingSize, actualCaptureNum * tilingSize + tilingSize)
                .copy_(deviceTiling);
            actualCaptureNum++;
            tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                         tilingSize, at::kByte);
        }

        // 4. Create workspace
        auto workspace_tensor =
            at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(x_t.options().device()));

        // 5. Launch kernel
        if (dtype == at::kBFloat16) {
            EXEC_KERNEL_CMD(causal_conv1d_update_bfloat16_t, block_dim, x_t, weight_t, conv_state_t,
                            has_indices ? conv_state_indices : at::empty(0, x_t.options()),
                            has_bias ? bias_t : at::empty(0, x_t.options()),
                            has_num_accept ? num_accepted_tokens : at::empty(0, x_t.options()),
                            has_query_loc ? query_start_loc : at::empty(0, x_t.options()), y_t, workspace_tensor,
                            tilingTensor);
        } else {
            EXEC_KERNEL_CMD(causal_conv1d_update_half, block_dim, x_t, weight_t, conv_state_t,
                            has_indices ? conv_state_indices : at::empty(0, x_t.options()),
                            has_bias ? bias_t : at::empty(0, x_t.options()),
                            has_num_accept ? num_accepted_tokens : at::empty(0, x_t.options()),
                            has_query_loc ? query_start_loc : at::empty(0, x_t.options()), y_t, workspace_tensor,
                            tilingTensor);
        }
    };

    // The kernel allocates ~26*dim bytes of UB (width*dim weight + several dim*sizeof
    // operand/accum buffers). Atlas A3 has 192KB UB per AIV core, so dim much beyond
    // ~7500 overflows ("ub address out of bounds" / vector core exception). Models
    // like Qwen3-Next have a much larger conv_dim, so we tile along the dim axis
    // here at the integration layer and call the kernel per tile.
    constexpr int64_t kMaxDimTile = 4096;
    if (dim <= kMaxDimTile) {
        launch_for_tile(x, weight, conv_state, bias, y, dim);
        return y;
    }

    const int64_t num_tiles = (dim + kMaxDimTile - 1) / kMaxDimTile;
    for (int64_t t = 0; t < num_tiles; ++t) {
        const int64_t tile_start = t * kMaxDimTile;
        const int64_t tile_end = std::min(tile_start + kMaxDimTile, dim);
        const int64_t tile_dim = tile_end - tile_start;

        // Slice along the dim axis. Slicing yields non-contiguous views; the kernel
        // requires contiguous inputs, so we materialize tile copies and write the
        // results back after the kernel returns.
        auto x_tile = x.slice(2, tile_start, tile_end).contiguous();
        auto weight_tile = weight.slice(1, tile_start, tile_end).contiguous();
        auto conv_state_tile = conv_state.slice(2, tile_start, tile_end).contiguous();
        auto bias_tile = has_bias ? bias.slice(0, tile_start, tile_end).contiguous() : bias;
        auto y_tile = at::empty_like(x_tile);

        launch_for_tile(x_tile, weight_tile, conv_state_tile, bias_tile, y_tile, tile_dim);

        // Stitch the tile outputs back into the full tensors. conv_state is updated
        // in-place by the kernel on the tile copy, so we must copy it back.
        y.slice(2, tile_start, tile_end).copy_(y_tile);
        conv_state.slice(2, tile_start, tile_end).copy_(conv_state_tile);
    }

    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
