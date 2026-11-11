// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Framework adaptation of csrc/mega_chunk_gdn; PTO computation is unchanged.

#include "../op_kernel/mega_chunk_gdn_tiling_data.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include <limits>

namespace optiling {
static ge::graphStatus TilingMegaChunkGdn(gert::TilingContext *context)
{
    if (!context) return ge::GRAPH_FAILED;
    const auto *attrs = context->GetAttrs();
    const auto *q_shape = context->GetInputShape(0);
    const auto *v_shape = context->GetInputShape(2);
    if (!attrs || !q_shape || !v_shape) return ge::GRAPH_FAILED;
    const auto &q = q_shape->GetStorageShape();
    const auto &v = v_shape->GetStorageShape();
    const auto *has_initial = attrs->GetAttrPointer<bool>(0);
    const auto *block_dim = attrs->GetAttrPointer<int64_t>(1);
    const auto *batch_size = attrs->GetAttrPointer<int64_t>(2);
    const auto *seq_len = attrs->GetAttrPointer<int64_t>(3);
    const auto *tokens = attrs->GetAttrPointer<int64_t>(4);
    const auto *matrices = attrs->GetAttrPointer<int64_t>(5);
    if (!has_initial || !block_dim || !batch_size || !seq_len || !tokens || !matrices || q.GetDimNum() != 4 ||
        v.GetDimNum() != 4 || q.GetDim(0) != 1 || q.GetDim(3) != 128 || v.GetDim(3) != 128 || q.GetDim(2) <= 0 ||
        v.GetDim(2) % q.GetDim(2) != 0 || *block_dim <= 0 || *block_dim > std::numeric_limits<uint32_t>::max() ||
        *matrices < 0 || *matrices > std::numeric_limits<uint32_t>::max())
        return ge::GRAPH_FAILED;
    sgl_mega_gdn::TilingData data{};
    data.has_initial_state = *has_initial ? 1 : 0;
    data.batch_size = *batch_size;
    data.seq_len = *seq_len;
    data.total_tokens = *tokens;
    data.num_matrices = static_cast<uint32_t>(*matrices);
    data.num_heads = static_cast<uint32_t>(v.GetDim(2));
    data.num_key_heads = static_cast<uint32_t>(q.GetDim(2));
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(data) || !workspace) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    context->SetBlockDim(static_cast<uint32_t>(*block_dim));
    context->SetTilingKey(0);
    // CANN's generated ARCH22 MIX wrapper clears its system workspace.
    // The unchanged PTO scratch buffers are separate tensor arguments.
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus Prepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(SglMegaChunkGdn).Tiling(TilingMegaChunkGdn).TilingParse<CompileInfo>(Prepare);
}  // namespace optiling
