#include "causal_conv1d.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"

// Ring sizes the kernel is compiled for -- must match FOR_EACH_RING_SIZE in
// op_kernel/causal_conv1d.cpp. Each row is (ringSize, maxTileWidth); this one list
// drives the launch-stub declarations, the per-ring tile-width lookup, and the dispatch.
#define FOR_EACH_RING_SIZE(DO) DO(2, 4096) DO(4, 3072) DO(8, 1536) DO(16, 896) DO(32, 384) DO(64, 128)

// AscendC emits one host launch stub (aclrtlaunch_<entry>) per kernel entry. A macro
// can't generate #include directives, so rather than pull in 24 generated headers we
// forward-declare the stubs from the ring-size list (their definitions come from the
// linked causal_conv1d_kernel lib; same approach as causal_conv1d_update).

// EXEC_KERNEL_CMD launches via ACLRT_LAUNCH_KERNEL(name) -> the aclrtlaunch_<name> symbol.
#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif
// clang-format off
// Launch-stub arg types = (blockDim, stream, then the kernel params with GM_ADDR ->
// void*); must mirror CONV_PARAMS / WB_PARAMS in op_kernel/causal_conv1d.cpp.
#define CONV_STUB_PARAMS                                                                                       \
    uint32_t, aclrtStream, void *, void *, void *, void *, void *, void *, void *, void *, uint32_t, uint32_t, \
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t
#define WRITEBACK_STUB_PARAMS                                                                                  \
    uint32_t, aclrtStream, void *, void *, void *, void *, void *, uint32_t, uint32_t, uint32_t, uint32_t,     \
        uint32_t, uint32_t, uint32_t, uint32_t, int32_t
// Worker macro: one ring size -> the conv + writeback stub, each for half and bf16.
#define DECLARE_LAUNCH_STUBS(ringSize, maxTileWidth)                                             \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_rs##ringSize##_half(CONV_STUB_PARAMS);         \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_rs##ringSize##_bf16(CONV_STUB_PARAMS);         \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_wb_rs##ringSize##_half(WRITEBACK_STUB_PARAMS); \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_wb_rs##ringSize##_bf16(WRITEBACK_STUB_PARAMS);
FOR_EACH_RING_SIZE(DECLARE_LAUNCH_STUBS)  // expand the list -> all 24 launch-stub declarations
#undef DECLARE_LAUNCH_STUBS
#undef WRITEBACK_STUB_PARAMS
#undef CONV_STUB_PARAMS
// clang-format on

namespace sglang {
namespace npu_kernel {
namespace {  // file-local helpers (internal linkage; avoids clashes with other ops)

// Accumulator-ring size = smallest power of two >= width. The kernel templates on the
// ring size (compile-time) and takes the width at runtime; the host computes the ring
// size here and launches the matching variant, so any width <= ring size reuses it.
constexpr uint32_t roundUpToPow2(uint32_t width)
{
    uint32_t n = (width != 0u) ? width - 1u : 0u;
    n |= n >> 1u;
    n |= n >> 2u;
    n |= n >> 4u;
    n |= n >> 8u;
    n |= n >> 16u;
    return n + 1u;
}

// Per-ring channel-tile width -- MUST match the (ringSize, maxTileWidth) the kernel is
// compiled with (both derive from FOR_EACH_RING_SIZE). A larger ring needs a smaller
// tile to fit the 192 KiB UB.
#define MAX_WIDTH_CASE(ringSize, maxTileWidth) \
    case ringSize:                             \
        return maxTileWidth##u;
constexpr uint32_t maxTileWidthForRing(uint32_t ringSize)
{
    switch (ringSize) {
        FOR_EACH_RING_SIZE(MAX_WIDTH_CASE)
        default:
            return 0u;
    }
}
#undef MAX_WIDTH_CASE

// Supported filter widths: any width in [2, 64], routed to the roundUpToPow2(width)
// ring variant. width > 64 would need ring 128, which does not fit UB.
constexpr bool isSupportedWidth(uint32_t width)
{
    return width >= 2u && width <= 64u;
}

template <class T>
constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}
constexpr uint32_t round_up_128(uint32_t x)
{
    return ((x + 127u) / 128u) * 128u;
}

std::pair<uint32_t, uint32_t> tiling_causal_conv1d(uint64_t numCores, uint64_t batch, const uint64_t dim,
                                                   const uint64_t seqLength, const uint64_t width,
                                                   const uint64_t maxChannels)
{
    // 128 = Number of Vector Lanes in FP16/BF16
    constexpr uint64_t minChannels = 128;  // Must divide maxChannels

    uint64_t gcdCoreBatch = std::gcd(numCores, batch);
    numCores /= gcdCoreBatch;
    batch /= gcdCoreBatch;

    uint64_t numChannels = ceil_div(dim, maxChannels);
    uint64_t channelsPerTile = ceil_div(ceil_div(dim, numChannels), minChannels) * minChannels;

    uint64_t seqChunks = 1u;
    double bestScore = std::numeric_limits<double>::infinity();

    numChannels = ceil_div(dim, channelsPerTile);
    const uint64_t depthNumerator = batch * numChannels;

    const uint64_t uppBnd = numCores / std::gcd(numCores, numChannels);
    for (uint64_t numChunks = 1u; numChunks <= uppBnd; ++numChunks) {
        uint64_t depth = ceil_div(depthNumerator * numChunks, numCores);
        uint64_t tokens = ceil_div(seqLength, numChunks);
        uint64_t work = tokens + width;
        double score = static_cast<double>(depth) * static_cast<double>(work);
        if (score < bestScore) {
            bestScore = score;
            seqChunks = numChunks;
        }
    }

    return {static_cast<uint32_t>(channelsPerTile), static_cast<uint32_t>(seqChunks)};
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
    TORCH_CHECK(isSupportedWidth(width), "Only filter widths 2..64 are supported, got ", weight.size(0));
    TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D [num_cache_lines, state_len, dim]");
    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "Only BF16 and FP16 are supported, got ", dtype);
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dtype, "conv_states dtype must match x dtype");
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
        // Guard the size(0)-1 below: an empty/too-short qsl would underflow batch to ~4e9.
        TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) >= 2,
                    "query_start_loc must be 1D and have at least 2 elements");
        batch = static_cast<uint32_t>(query_start_loc.size(0) - 1);
    } else {
        inputMode = 1;
        batch = static_cast<uint32_t>(x.size(0));
        seqLen = static_cast<uint32_t>(x.size(1));
        dim = static_cast<uint32_t>(x.size(2));
    }
    TORCH_CHECK(batch > 0 && dim > 0, "bad batch/dim");
    // cache_indices[seq] and has_initial_state[seq] are read per sequence in both layouts.
    TORCH_CHECK(cache_indices.dim() == 1 && cache_indices.size(0) >= static_cast<int64_t>(batch),
                "cache_indices must be 1D and have size >= batch");
    TORCH_CHECK(has_initial_state.dim() == 1 && has_initial_state.size(0) >= static_cast<int64_t>(batch),
                "has_initial_state must be 1D and have size >= batch");
    TORCH_CHECK(dim % 16 == 0, "dim must be multiple of 16 for fp16/bf16 alignment, but got ", dim);
    TORCH_CHECK(weight.size(1) == static_cast<int64_t>(dim), "weight.shape[1] must equal dim");
    TORCH_CHECK(conv_states.size(2) == static_cast<int64_t>(dim), "conv_states.shape[2] must equal dim");
    if (has_bias) {  // bias is read in the I/O dtype and cast to fp32 in the kernel
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == static_cast<int64_t>(dim), "bias must be 1D [dim]");
        TORCH_CHECK(bias.scalar_type() == dtype, "bias dtype must match x dtype");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
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
    const uint32_t ringSize = roundUpToPow2(width);                     // compile-time ring variant to launch
    const uint32_t maxChannelsPerTile = maxTileWidthForRing(ringSize);  // UB-bound tile width for this ring

    const auto [channelsPerTile, seqChunks] =
        tiling_causal_conv1d(core_num, batch, dim, avgSeqLen, width, maxChannelsPerTile);
    const uint32_t channelTiles = ceil_div(dim, channelsPerTile);

    // weight/bias enter in the I/O dtype (fp16/bf16) and are cast to fp32 inside the
    // kernel; pass a native empty placeholder when there is no bias.
    const at::Tensor biasArg = has_bias ? bias : at::empty({0}, x.options());
    at::Tensor y = at::empty_like(x);

    // conv has one task per (batch, channel tile, seq chunk); the writeback only
    // touches the sequence tail, so it drops the seq-chunk axis. Cap block dim at the cores.
    const uint32_t blockDimConv = std::min<uint32_t>(batch * channelTiles * seqChunks, core_num);
    const uint32_t blockDimWb = std::min<uint32_t>(batch * channelTiles, core_num);
    const uint32_t actFlag = activation_mode ? 1u : 0u;
    const uint32_t biasFlag = has_bias ? 1u : 0u;
    const int32_t padSlot = static_cast<int32_t>(pad_slot_id);

    // Launch the ring = roundUpToPow2(width) variant (entry suffix rs<ring>) and pass
    // the actual width as the runtime K. launch(suffix) fires the conv + writeback pair;
    // the per-dtype switch over the ring-size list selects the entry.
#define launch(suffix)                                                                                          \
    do {                                                                                                        \
        EXEC_KERNEL_CMD(causal_conv1d_##suffix, blockDimConv, x, weight, biasArg, conv_states, query_start_loc, \
                        cache_indices, has_initial_state, y, dim, batch, inputMode, seqLen, stateLen, width,    \
                        channelsPerTile, channelTiles, seqChunks, actFlag, biasFlag, padSlot);                  \
        EXEC_KERNEL_CMD(causal_conv1d_wb_##suffix, blockDimWb, x, conv_states, query_start_loc, cache_indices,  \
                        has_initial_state, dim, batch, inputMode, seqLen, stateLen, width, channelsPerTile,     \
                        channelTiles, padSlot);                                                                 \
    } while (0)
#define DISPATCH_HALF(ringSize, maxTileWidth) \
    case ringSize:                            \
        launch(rs##ringSize##_half);          \
        break;
#define DISPATCH_BF16(ringSize, maxTileWidth) \
    case ringSize:                            \
        launch(rs##ringSize##_bf16);          \
        break;
    if (dtype == at::kHalf) {
        switch (ringSize) {
            FOR_EACH_RING_SIZE(DISPATCH_HALF)
            default:
                break;
        }
    } else {
        switch (ringSize) {
            FOR_EACH_RING_SIZE(DISPATCH_BF16)
            default:
                break;
        }
    }
#undef DISPATCH_BF16
#undef DISPATCH_HALF
#undef launch
    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
