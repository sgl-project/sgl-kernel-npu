/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "aclnn_runtime.h"
#include "aclnn_sgl_compressor.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <limits>

namespace {
at::Tensor compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate, at::Tensor state_cache,
                      const at::Tensor &ape, const at::Tensor &norm_weight, const at::Tensor &rope_sin,
                      const at::Tensor &rope_cos, const c10::optional<at::Tensor> &state_block_table,
                      const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                      const c10::optional<at::Tensor> &start_pos, int64_t rope_head_dim, int64_t cmp_ratio,
                      int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode,
                      int64_t state_cache_stride_dim0)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "compressor: x must be on an NPU");
    const c10::DeviceGuard guard(x.device());
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "compressor: x must be TH or BSH");
    TORCH_CHECK(norm_weight.dim() == 1 && state_cache.dim() == 3, "compressor: invalid weight/cache rank");
    // These checks protect output allocation and the compiled entry. The
    // original tiling validates supported values and tensor-shape relations.
    TORCH_CHECK(cmp_ratio > 0 && cmp_ratio <= std::numeric_limits<int32_t>::max(),
                "compressor: cmp_ratio must be positive and fit int32");
    TORCH_CHECK(rotary_mode == 2, "compressor: only rotary_mode=2 has a compiled kernel");
    if (x.dim() == 2) {
        TORCH_CHECK(cu_seqlens.has_value() && cu_seqlens->dim() == 1 && cu_seqlens->size(0) >= 1,
                    "compressor: TH output allocation requires cu_seqlens [B+1]");
    }
    auto check = [&](const at::Tensor &t, at::ScalarType dtype) {
        TORCH_CHECK(t.device() == x.device(), "compressor: inputs must be on the same NPU");
        TORCH_CHECK(t.is_contiguous(), "compressor: inputs must be contiguous");
        TORCH_CHECK(t.scalar_type() == dtype, "compressor: incorrect input dtype");
    };
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "compressor: x must be bf16 or fp16");
    for (const auto &t : {x, wkv, wgate})
        check(t, x.scalar_type());
    for (const auto &t : {state_cache, ape, norm_weight, rope_sin, rope_cos})
        check(t, at::kFloat);
    for (const auto &t : {state_block_table, cu_seqlens, seqused, start_pos})
        if (t.has_value()) check(*t, at::kInt);
    if (state_cache_stride_dim0 == 0) state_cache_stride_dim0 = state_cache.size(1) * state_cache.size(2);
    // The bridge passes the stride to the unchanged tiling as int32.
    TORCH_CHECK(state_cache_stride_dim0 >= 0 && state_cache_stride_dim0 <= std::numeric_limits<int32_t>::max(),
                "compressor: cache stride must fit nonnegative int32");
    const int64_t dim = norm_weight.size(0);
    at::Tensor output;
    if (x.dim() == 3)
        output = at::empty({x.size(0), (x.size(1) + cmp_ratio - 1) / cmp_ratio, dim}, x.options());
    else
        output = at::empty({std::min(x.size(0), x.size(0) / cmp_ratio + cu_seqlens->size(0) - 1), dim}, x.options());
    // Empty inputs also follow query -> execute. The Kernel handles EMPTY_X;
    // do not bypass CANN validation or leave a queried executor unused.
    const auto dtype = x.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16;
    auto xDesc = sgl_cann::tensor_descriptor(x, dtype);
    auto wkvDesc = sgl_cann::tensor_descriptor(wkv, dtype);
    auto wgateDesc = sgl_cann::tensor_descriptor(wgate, dtype);
    auto stateDesc = sgl_cann::tensor_descriptor(state_cache, ACL_FLOAT);
    auto apeDesc = sgl_cann::tensor_descriptor(ape, ACL_FLOAT);
    auto normDesc = sgl_cann::tensor_descriptor(norm_weight, ACL_FLOAT);
    auto sinDesc = sgl_cann::tensor_descriptor(rope_sin, ACL_FLOAT);
    auto cosDesc = sgl_cann::tensor_descriptor(rope_cos, ACL_FLOAT);
    auto optionalDesc = [](const c10::optional<at::Tensor> &t) -> std::shared_ptr<aclTensor> {
        return t.has_value() ? sgl_cann::tensor_descriptor(*t, ACL_INT32) : nullptr;
    };
    auto tableDesc = optionalDesc(state_block_table);
    auto cuDesc = optionalDesc(cu_seqlens);
    auto usedDesc = optionalDesc(seqused);
    auto posDesc = optionalDesc(start_pos);
    auto outDesc = sgl_cann::tensor_descriptor(output, dtype);
    static auto query = reinterpret_cast<decltype(&aclnnSglCompressorGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglCompressorGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglCompressor)>(sgl_cann::op_api_symbol("aclnnSglCompressor"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(xDesc.get(), wkvDesc.get(), wgateDesc.get(), stateDesc.get(), apeDesc.get(), normDesc.get(),
                        sinDesc.get(), cosDesc.get(), tableDesc.get(), cuDesc.get(), usedDesc.get(), posDesc.get(),
                        rope_head_dim, cmp_ratio, coff, static_cast<float>(norm_eps), rotary_mode, cache_mode,
                        state_cache_stride_dim0, outDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "compressor workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "compressor workspace exceeds Tensor allocation limit");
    // arch35's legacy wrapper zeroes readGen before every launch. ACLNN owns the
    // workspace layout: zero the allocation without assuming its user offset.
    auto workspace = at::zeros({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &t : {x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos, output, workspace})
        t.record_stream(stream);
    for (const auto &t : {state_block_table, cu_seqlens, seqused, start_pos})
        if (t.has_value()) t->record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [=]() -> int {
        // Tensor and descriptor captures retain their storage while the host task queues.
        (void)x;
        (void)wkv;
        (void)wgate;
        (void)state_cache;
        (void)ape;
        (void)norm_weight;
        (void)rope_sin;
        (void)rope_cos;
        (void)state_block_table;
        (void)cu_seqlens;
        (void)seqused;
        (void)start_pos;
        (void)output;
        (void)xDesc;
        (void)wkvDesc;
        (void)wgateDesc;
        (void)stateDesc;
        (void)apeDesc;
        (void)normDesc;
        (void)sinDesc;
        (void)cosDesc;
        (void)tableDesc;
        (void)cuDesc;
        (void)usedDesc;
        (void)posDesc;
        (void)outDesc;
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "compressor ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglCompressor", launch);
    return output;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor! state_cache, Tensor ape, Tensor norm_weight, Tensor "
        "rope_sin, Tensor rope_cos, Tensor? state_block_table=None, Tensor? cu_seqlens=None, Tensor? seqused=None, "
        "Tensor? start_pos=None, int rope_head_dim=64, int cmp_ratio=4, int coff=1, float norm_eps=1e-6, int "
        "rotary_mode=2, int cache_mode=1, int state_cache_stride_dim0=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("compressor", &compressor);
}
