/*!
 * \file causal_conv1d.cpp
 * \brief PTO-ISA causal_conv1d host launcher (drop-in alternative to the AscendC one).
 *
 * Same semantics/signature as causal_conv1d_impl. Casts the tiny weight/bias to fp32
 * once on host, chooses a (channel-tile x L-chunk) grid that fills all cores even at
 * small batch, then launches the PTO conv kernel followed by a state-writeback kernel.
 */
#include "causal_conv1d.h"

#include <algorithm>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_causal_conv1d_half.h"
#include "aclrtlaunch_causal_conv1d_bf16.h"
#include "aclrtlaunch_causal_conv1d_wb_half.h"
#include "aclrtlaunch_causal_conv1d_wb_bf16.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {
namespace {

constexpr int64_t WIDTH = 4;
constexpr uint32_t MAX_W = 3072;
constexpr uint32_t LC_MIN = 32;

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
inline uint32_t round_up_128(uint32_t x) { return ((x + 127u) / 128u) * 128u; }

}  // namespace

HOST_API at::Tensor causal_conv1d_impl(const at::Tensor &x, const at::Tensor &weight,
                                           const at::Tensor &conv_states, const at::Tensor &query_start_loc,
                                           const at::Tensor &cache_indices, const at::Tensor &has_initial_state,
                                           const at::Tensor &bias, bool activation_mode, int64_t pad_slot_id)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D [cu_seqlen, dim] or 3D [batch, seq_len, dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [width, dim], got shape ", weight.sizes());
    TORCH_CHECK(weight.size(0) == WIDTH, "Only width == 4 is supported, got ", weight.size(0));
    TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D [num_cache_lines, state_len, dim]");
    const at::ScalarType dt = x.scalar_type();
    TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16, "Only BF16 and FP16 are supported, got ", dt);
    TORCH_CHECK(weight.scalar_type() == dt, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dt, "conv_states dtype must match x dtype");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc dtype must be int32");
    TORCH_CHECK(cache_indices.scalar_type() == at::kInt, "cache_indices dtype must be int32");
    TORCH_CHECK(has_initial_state.scalar_type() == at::kBool, "has_initial_state dtype must be bool");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && conv_states.is_contiguous(), "inputs must be contiguous");

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
    TORCH_CHECK(stateLen >= WIDTH - 1, "state_len must be >= width-1");

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(plat != nullptr, "no AscendC platform");
    const uint32_t core_num = static_cast<uint32_t>(plat->GetCoreNumAiv());
    TORCH_CHECK(core_num > 0, "bad core_num");

    // ---- grid: channel tiles (col_w) x L-chunks, filling cores; batch first ----
    uint32_t avg_len;
    if (inputMode == 1) {
        avg_len = seqLen;
    } else {
        avg_len = std::max<uint32_t>(1u, static_cast<uint32_t>(x.size(0)) / batch);
    }
    const uint32_t target = std::max<uint32_t>(1u, ceil_div(core_num, batch));
    const uint32_t max_chunks = std::max<uint32_t>(1u, avg_len / LC_MIN);
    uint32_t num_wt = std::max<uint32_t>(ceil_div(dim, MAX_W), ceil_div(target, max_chunks));
    num_wt = std::min<uint32_t>(num_wt, ceil_div(dim, 128u));
    num_wt = std::max<uint32_t>(1u, num_wt);
    uint32_t col_w = round_up_128(ceil_div(dim, num_wt));
    col_w = std::min<uint32_t>(std::max<uint32_t>(col_w, 128u), MAX_W);
    col_w = std::min<uint32_t>(col_w, dim);
    num_wt = ceil_div(dim, col_w);  // = blocksPerSeq
    uint32_t lchunks = std::min<uint32_t>(std::max<uint32_t>(1u, ceil_div(target, num_wt)), max_chunks);

    auto weight_f32 = weight.to(at::kFloat).contiguous();
    auto bias_f32 = has_bias ? bias.to(at::kFloat).contiguous() : at::empty({0}, x.options().dtype(at::kFloat));

    at::Tensor y = at::empty_like(x);

    const uint32_t grid_conv = batch * num_wt * lchunks;
    const uint32_t grid_wb = batch * num_wt;
    const uint32_t bd_conv = std::min<uint32_t>(grid_conv, core_num);
    const uint32_t bd_wb = std::min<uint32_t>(grid_wb, core_num);
    const uint32_t act = activation_mode ? 1u : 0u;
    const uint32_t hb = has_bias ? 1u : 0u;
    const int32_t pad = static_cast<int32_t>(pad_slot_id);

    if (dt == at::kHalf) {
        EXEC_KERNEL_CMD(causal_conv1d_half, bd_conv, x, weight_f32, bias_f32, conv_states, query_start_loc,
                        cache_indices, has_initial_state, y, dim, batch, inputMode, seqLen, stateLen, col_w, num_wt,
                        lchunks, act, hb, pad);
        EXEC_KERNEL_CMD(causal_conv1d_wb_half, bd_wb, x, conv_states, query_start_loc, cache_indices, has_initial_state, dim,
                        batch, inputMode, seqLen, stateLen, col_w, num_wt, pad);
    } else {
        EXEC_KERNEL_CMD(causal_conv1d_bf16, bd_conv, x, weight_f32, bias_f32, conv_states, query_start_loc,
                        cache_indices, has_initial_state, y, dim, batch, inputMode, seqLen, stateLen, col_w, num_wt,
                        lchunks, act, hb, pad);
        EXEC_KERNEL_CMD(causal_conv1d_wb_bf16, bd_wb, x, conv_states, query_start_loc, cache_indices, has_initial_state, dim,
                        batch, inputMode, seqLen, stateLen, col_w, num_wt, pad);
    }
    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
