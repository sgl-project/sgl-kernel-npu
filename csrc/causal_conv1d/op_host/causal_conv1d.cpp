#include "causal_conv1d.h"

#include <algorithm>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
// AscendC generates one launch stub (aclrtlaunch_<entry>.h) per __global__ kernel
// entry, and EXEC_KERNEL_CMD(entry, ...) needs the matching declaration. We compile
// a dedicated conv + writeback entry per (width, dtype), so there is one include per
// entry: widths {2,3,4,5,8,16,32,64} (4 = the bare name) x {half, bf16} x {conv, wb}.
#include "aclrtlaunch_causal_conv1d_half.h"
#include "aclrtlaunch_causal_conv1d_bf16.h"
#include "aclrtlaunch_causal_conv1d_k2_half.h"
#include "aclrtlaunch_causal_conv1d_k2_bf16.h"
#include "aclrtlaunch_causal_conv1d_k3_half.h"
#include "aclrtlaunch_causal_conv1d_k3_bf16.h"
#include "aclrtlaunch_causal_conv1d_k5_half.h"
#include "aclrtlaunch_causal_conv1d_k5_bf16.h"
#include "aclrtlaunch_causal_conv1d_k8_half.h"
#include "aclrtlaunch_causal_conv1d_k8_bf16.h"
#include "aclrtlaunch_causal_conv1d_k16_half.h"
#include "aclrtlaunch_causal_conv1d_k16_bf16.h"
#include "aclrtlaunch_causal_conv1d_k32_half.h"
#include "aclrtlaunch_causal_conv1d_k32_bf16.h"
#include "aclrtlaunch_causal_conv1d_k64_half.h"
#include "aclrtlaunch_causal_conv1d_k64_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_half.h"
#include "aclrtlaunch_causal_conv1d_wb_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_k2_half.h"
#include "aclrtlaunch_causal_conv1d_wb_k2_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_k3_half.h"
#include "aclrtlaunch_causal_conv1d_wb_k3_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_k5_half.h"
#include "aclrtlaunch_causal_conv1d_wb_k5_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_k8_half.h"
#include "aclrtlaunch_causal_conv1d_wb_k8_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_k16_half.h"
#include "aclrtlaunch_causal_conv1d_wb_k16_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_k32_half.h"
#include "aclrtlaunch_causal_conv1d_wb_k32_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_k64_half.h"
#include "aclrtlaunch_causal_conv1d_wb_k64_bf16.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {
namespace {  // file-local helpers (internal linkage; avoids clashes with other ops)

// Minimum rows per sequence chunk when splitting the L axis to fill cores (smaller
// chunks expose more parallelism but replay more causal-halo rows per chunk).
constexpr uint32_t LC_MIN = 32;

// Per-tile channel width for filter width K -- MUST match the (K, MAX_W) the
// kernel's DEF_CONV/DEF_WB instantiations are compiled with. A larger K needs a
// smaller tile to fit the 192 KiB UB (the accumulator ring grows with K).
constexpr uint32_t maxWForK(uint32_t k)
{
    switch (k) {
        case 2:
        case 3:
        case 4:
            return 3072u;
        case 5:
            return 2048u;
        case 8:
            return 1536u;
        case 16:
            return 896u;
        case 32:
            return 384u;
        default:
            return 128u;  // k == 64
    }
}

// Filter widths the kernel is compiled for (each has a dedicated entry).
constexpr bool isSupportedWidth(uint32_t k)
{
    return k == 2 || k == 3 || k == 4 || k == 5 || k == 8 || k == 16 || k == 32 || k == 64;
}

constexpr uint32_t ceil_div(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}
constexpr uint32_t round_up_128(uint32_t x)
{
    return ((x + 127u) / 128u) * 128u;
}

}  // namespace

HOST_API at::Tensor causal_conv1d_impl(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &conv_states,
                                       const at::Tensor &query_start_loc, const at::Tensor &cache_indices,
                                       const at::Tensor &has_initial_state, const at::Tensor &bias,
                                       bool activation_mode, int64_t pad_slot_id)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D [cu_seqlen, dim] or 3D [batch, seq_len, dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [width, dim], got shape ", weight.sizes());
    const uint32_t width = static_cast<uint32_t>(weight.size(0));
    TORCH_CHECK(isSupportedWidth(width), "Only widths 2, 3, 4, 5, 8, 16, 32, 64 are supported, got ", weight.size(0));
    TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D [num_cache_lines, state_len, dim]");
    const at::ScalarType dt = x.scalar_type();
    TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16, "Only BF16 and FP16 are supported, got ", dt);
    TORCH_CHECK(weight.scalar_type() == dt, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dt, "conv_states dtype must match x dtype");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc dtype must be int32");
    TORCH_CHECK(cache_indices.scalar_type() == at::kInt, "cache_indices dtype must be int32");
    TORCH_CHECK(has_initial_state.scalar_type() == at::kBool, "has_initial_state dtype must be bool");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && conv_states.is_contiguous(),
                "inputs must be contiguous");

    const bool has_bias = bias.numel() > 0;
    uint32_t inputMode, batch, seqLen, dim;
    if (x.dim() == 2) {
        inputMode = 0;
        dim = static_cast<uint32_t>(x.size(1));
        seqLen = 0;
        batch = static_cast<uint32_t>(query_start_loc.size(0) - 1);
    } else {
        inputMode = 1;
        batch = static_cast<uint32_t>(x.size(0));
        seqLen = static_cast<uint32_t>(x.size(1));
        dim = static_cast<uint32_t>(x.size(2));
    }
    TORCH_CHECK(batch > 0 && dim > 0, "bad batch/dim");
    TORCH_CHECK(dim % 16 == 0, "dim must be multiple of 16 for fp16/bf16 alignment, but got ", dim);
    TORCH_CHECK(weight.size(1) == static_cast<int64_t>(dim), "weight.shape[1] must equal dim");
    TORCH_CHECK(conv_states.size(2) == static_cast<int64_t>(dim), "conv_states.shape[2] must equal dim");
    const uint32_t stateLen = static_cast<uint32_t>(conv_states.size(1));
    TORCH_CHECK(stateLen >= width - 1, "state_len must be >= width-1");

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(plat != nullptr, "no AscendC platform");
    const uint32_t core_num = static_cast<uint32_t>(plat->GetCoreNumAiv());
    TORCH_CHECK(core_num > 0, "bad core_num");

    // ---- launch grid: (channel tile) x (sequence chunk) work units, sized to
    // fill all AIV cores. Batch parallelism comes first; if batch alone can't fill
    // the cores we split the channel axis into tiles and the L axis into chunks. ----
    const uint32_t avgSeqLen =
        (inputMode == 1) ? seqLen : std::max<uint32_t>(1u, static_cast<uint32_t>(x.size(0)) / batch);
    const uint32_t maxChannelsPerTile = maxWForK(width);  // UB-bound tile width for this K
    const uint32_t targetTilesPerSeq = std::max<uint32_t>(1u, ceil_div(core_num, batch));
    const uint32_t maxSeqChunks = std::max<uint32_t>(1u, avgSeqLen / LC_MIN);

    uint32_t channelTiles =
        std::max<uint32_t>(ceil_div(dim, maxChannelsPerTile), ceil_div(targetTilesPerSeq, maxSeqChunks));
    channelTiles = std::min<uint32_t>(channelTiles, ceil_div(dim, 128u));  // >= 128 channels (one lane) per tile
    channelTiles = std::max<uint32_t>(1u, channelTiles);

    uint32_t channelsPerTile = round_up_128(ceil_div(dim, channelTiles));
    channelsPerTile = std::min<uint32_t>(std::max<uint32_t>(channelsPerTile, 128u), maxChannelsPerTile);
    channelsPerTile = std::min<uint32_t>(channelsPerTile, dim);
    channelTiles = ceil_div(dim, channelsPerTile);
    const uint32_t seqChunks =
        std::min<uint32_t>(std::max<uint32_t>(1u, ceil_div(targetTilesPerSeq, channelTiles)), maxSeqChunks);

    // weight/bias are tiny ([width, dim]/[dim]); cast once to fp32 (the kernel accumulates in fp32).
    auto weight_f32 = weight.to(at::kFloat).contiguous();
    auto bias_f32 = has_bias ? bias.to(at::kFloat).contiguous() : at::empty({0}, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty_like(x);

    // conv has one task per (batch, channel tile, seq chunk); the writeback only
    // touches the sequence tail, so it drops the seq-chunk axis. Cap block dim at the cores.
    const uint32_t blockDimConv = std::min<uint32_t>(batch * channelTiles * seqChunks, core_num);
    const uint32_t blockDimWb = std::min<uint32_t>(batch * channelTiles, core_num);
    const uint32_t actFlag = activation_mode ? 1u : 0u;
    const uint32_t biasFlag = has_bias ? 1u : 0u;
    const int32_t padSlot = static_cast<int32_t>(pad_slot_id);

    // One dedicated entry per (width, dtype), selected by width. launch(suffix) fires
    // the conv + writeback pair for entry `causal_conv1d_<suffix>` (e.g. k5_half; width
    // 4 is the bare `half`/`bf16`). dispatch(dt) pastes the dtype so it's named once.
#define launch(suffix)                                                                                                 \
    do {                                                                                                               \
        EXEC_KERNEL_CMD(causal_conv1d_##suffix, blockDimConv, x, weight_f32, bias_f32, conv_states, query_start_loc,   \
                        cache_indices, has_initial_state, y, dim, batch, inputMode, seqLen, stateLen, channelsPerTile, \
                        channelTiles, seqChunks, actFlag, biasFlag, padSlot);                                          \
        EXEC_KERNEL_CMD(causal_conv1d_wb_##suffix, blockDimWb, x, conv_states, query_start_loc, cache_indices,         \
                        has_initial_state, dim, batch, inputMode, seqLen, stateLen, channelsPerTile, channelTiles,     \
                        padSlot);                                                                                      \
    } while (0)
#define dispatch(dt)                 \
    switch (width) {                 \
        case 2:                      \
            launch(k2_##dt);         \
            break;                   \
        case 3:                      \
            launch(k3_##dt);         \
            break;                   \
        case 4:                      \
            launch(dt);              \
            break;                   \
        case 5:                      \
            launch(k5_##dt);         \
            break;                   \
        case 8:                      \
            launch(k8_##dt);         \
            break;                   \
        case 16:                     \
            launch(k16_##dt);        \
            break;                   \
        case 32:                     \
            launch(k32_##dt);        \
            break;                   \
        default:                     \
            launch(k64_##dt);        \
            break; /* width == 64 */ \
    }
    if (dt == at::kHalf) {
        dispatch(half);
    } else {
        dispatch(bf16);
    }
#undef dispatch
#undef launch
    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
