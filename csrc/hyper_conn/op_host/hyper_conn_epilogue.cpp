// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <cstdint>
#include <utility>

#include "defines.h"
#include "torch_helper.h"
#include "hyper_conn_tiling.h"
#include "aclrtlaunch_hyper_conn_grouped_norm_bf16.h"
#include "aclrtlaunch_hyper_conn_scaled_silu_bf16.h"
#include "aclrtlaunch_hyper_conn_branch_mix_bf16.h"
#include "aclrtlaunch_hyper_conn_branch_combine_bf16.h"
#include "tiling/platform/platform_ascendc.h"

namespace sglang {
namespace npu_kernel {

namespace {

using hyper_conn::EpilogueKernel;
using hyper_conn::LaunchBlocks;
using hyper_conn::MAX_BRANCHES;
using hyper_conn::MAX_WIDTH;
using hyper_conn::SelectColumnTile;
using hyper_conn::TunedBlocks;

int64_t VectorCoreNum()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return platform->GetCoreNumAiv();
}

void CheckBf16Npu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE, name, " must be on an NPU device");
}

// The branch count divides the accumulated sum, and the kernel multiplies by
// the reciprocal instead of dividing. A power of two keeps that exact.
void CheckPowerOfTwoBranches(int64_t branches)
{
    TORCH_CHECK(branches > 0 && branches <= static_cast<int64_t>(MAX_BRANCHES) && (branches & (branches - 1)) == 0,
                "branches must be a power of two in [1, ", MAX_BRANCHES, "], got ", branches);
}

void CheckBranchWidth(int64_t width)
{
    TORCH_CHECK(width > 0 && width <= static_cast<int64_t>(MAX_WIDTH) && width % 16 == 0,
                "per-branch width must be a positive multiple of 16 up to ", MAX_WIDTH, ", got ", width);
}

// Reject alias or byte-range overlap between a pre-allocated output tensor and
// any input tensor, while still allowing disjoint slices of one buffer pool.
bool TensorsOverlap(const at::Tensor &a, const at::Tensor &b)
{
    if (!a.defined() || !b.defined() || a.device() != b.device()) {
        return false;
    }
    if (a.unsafeGetTensorImpl() == b.unsafeGetTensorImpl()) {
        return true;
    }
    if (a.numel() == 0 || b.numel() == 0) {
        return false;
    }
    auto byteBounds = [](const at::Tensor &tensor) -> std::pair<uintptr_t, uintptr_t> {
        const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        const uintptr_t elemSize = tensor.element_size();
        if (tensor.is_contiguous()) {
            return {base, base + static_cast<uintptr_t>(tensor.numel()) * elemSize};
        }
        int64_t minElem = 0;
        int64_t maxElem = 0;
        for (int64_t d = 0; d < tensor.dim(); ++d) {
            const int64_t extent = (tensor.size(d) - 1) * tensor.stride(d);
            if (extent >= 0) {
                maxElem += extent;
            } else {
                minElem += extent;
            }
        }
        const uintptr_t begin = base - static_cast<uintptr_t>(-minElem) * elemSize;
        const uintptr_t end = base + static_cast<uintptr_t>(maxElem + 1) * elemSize;
        return {begin, end};
    };
    const auto [aBegin, aEnd] = byteBounds(a);
    const auto [bBegin, bEnd] = byteBounds(b);
    return aBegin < bEnd && bBegin < aEnd;
}

}  // namespace

/* Grouped RMS norm with a unit-offset, cycling weight slice.
 *
 * Groups are the `width`-long spans of `x` in row-major order, so for a
 * [rows, branches * width] input each row contributes `branches` groups. Group
 * g scales by 1 + weight[(g * width) % weight.numel() + j].
 */
HOST_API at::Tensor hyper_conn_grouped_norm(const at::Tensor &x, const at::Tensor &weight,
                                            c10::optional<int64_t> group_size, double eps,
                                            const c10::optional<at::Tensor> &out)
{
    CheckBf16Npu(x, "x");
    CheckBf16Npu(weight, "weight");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions, got ", x.dim());
    TORCH_CHECK(weight.dim() == 1, "weight must be 1-D, got ", weight.dim(), "-D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.device() == x.device(), "weight must be on x's device");

    const int64_t lastDim = x.size(-1);
    const int64_t width = group_size.has_value() ? group_size.value() : lastDim;
    TORCH_CHECK(width > 0 && width <= static_cast<int64_t>(MAX_WIDTH) && width % 8 == 0,
                "group size must be a positive multiple of 8 up to ", MAX_WIDTH, ", got ", width);
    TORCH_CHECK(lastDim % width == 0, "x's last dimension ", lastDim, " must be a multiple of the group size ", width);
    const int64_t weightLen = weight.numel();
    TORCH_CHECK(weightLen > 0 && weightLen % width == 0, "weight length ", weightLen,
                " must be a positive multiple of the group size ", width);
    TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be positive and finite, got ", eps);

    const int64_t groups = x.numel() / width;
    // The weight offset cycles every weightLen / width groups, so a partial
    // period at the end of the tensor would scale its groups by the wrong slice.
    TORCH_CHECK(groups % (weightLen / width) == 0, "the group count ", groups,
                " must be a whole number of weight periods (", weightLen / width, ")");

    at::Tensor result;
    if (out.has_value()) {
        CheckBf16Npu(*out, "out");
        TORCH_CHECK(out->device() == x.device(), "out must be on x's device");
        TORCH_CHECK(out->sizes() == x.sizes(), "out must have x's shape");
        TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
        TORCH_CHECK(!out->requires_grad(), "out must not require gradients");
        TORCH_CHECK(!TensorsOverlap(*out, x) && !TensorsOverlap(*out, weight),
                    "out must not alias or overlap x or weight");
        result = *out;
    } else {
        result = at::empty_like(x);
    }
    if (groups == 0) {
        return result;
    }

    // The sweep is keyed on the leading dimension, which is the token count in
    // the production [rows, branches * width] layout.
    const uint32_t tunedRows = static_cast<uint32_t>(x.size(0));
    const uint32_t blockDim =
        LaunchBlocks(TunedBlocks(EpilogueKernel::GROUPED_NORM, static_cast<uint32_t>(width), tunedRows),
                     static_cast<uint64_t>(groups), VectorCoreNum());

    auto npuStream = c10_npu::getCurrentNPUStream();
    x.record_stream(npuStream);
    weight.record_stream(npuStream);
    if (out.has_value()) {
        result.record_stream(npuStream);
    }

    uint32_t groupsU32 = static_cast<uint32_t>(groups);
    uint32_t widthU32 = static_cast<uint32_t>(width);
    uint32_t weightLenU32 = static_cast<uint32_t>(weightLen);
    float epsF32 = static_cast<float>(eps);
    // The aicore scalar unit cannot convert an unsigned integer to float.
    float invWidth = 1.0f / static_cast<float>(width);
    EXEC_KERNEL_CMD(hyper_conn_grouped_norm_bf16, blockDim, x, weight, result, groupsU32, widthU32, weightLenU32,
                    epsF32, invWidth);
    return result;
}

/* SiLU of x / scale in fp32, rounded to bf16 once when writing the output.
 *
 * `x` may be a column slice of a wider matmul output: only a unit column
 * stride is required, and the row pitch is passed to the kernel.
 */
HOST_API at::Tensor hyper_conn_scaled_silu(const at::Tensor &x, double scale)
{
    CheckBf16Npu(x, "x");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D, got ", x.dim(), "-D");
    TORCH_CHECK(x.stride(1) == 1, "x must have a unit column stride");
    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    TORCH_CHECK(cols > 0 && cols <= static_cast<int64_t>(MAX_WIDTH) && cols % 16 == 0,
                "x's column count must be a positive multiple of 16 up to ", MAX_WIDTH, ", got ", cols);
    TORCH_CHECK(rows == 0 || x.stride(0) >= cols, "x's row pitch ", x.stride(0), " must cover its ", cols, " columns");
    int exponent = 0;
    TORCH_CHECK(std::isfinite(scale) && scale > 0 && std::frexp(scale, &exponent) == 0.5,
                "scale must be a positive power of two, got ", scale);

    at::Tensor out = at::empty({rows, cols}, x.options());
    if (rows == 0) {
        return out;
    }

    // The block count tracks the row count, not the column count, so the
    // lookup uses the first tabulated width rather than this operator's.
    const uint32_t blockDim =
        LaunchBlocks(TunedBlocks(EpilogueKernel::SCALED_SILU, hyper_conn::TUNED_WIDTHS[0], static_cast<uint32_t>(rows)),
                     static_cast<uint64_t>(rows), VectorCoreNum());

    auto npuStream = c10_npu::getCurrentNPUStream();
    x.record_stream(npuStream);

    uint32_t rowsU32 = static_cast<uint32_t>(rows);
    uint32_t colsU32 = static_cast<uint32_t>(cols);
    uint32_t strideU32 = static_cast<uint32_t>(x.stride(0));
    float invScale = static_cast<float>(1.0 / scale);
    EXEC_KERNEL_CMD(hyper_conn_scaled_silu_bf16, blockDim, x, out, rowsU32, colsU32, strideU32, invScale);
    return out;
}

/* Gated reduction over the branch axis of a [rows, branches * width] pair. */
HOST_API at::Tensor hyper_conn_branch_mix(const at::Tensor &normed, const at::Tensor &gate_logits, int64_t branches)
{
    CheckBf16Npu(normed, "normed");
    CheckBf16Npu(gate_logits, "gate_logits");
    TORCH_CHECK(normed.dim() == 2, "normed must be 2-D, got ", normed.dim(), "-D");
    TORCH_CHECK(normed.sizes() == gate_logits.sizes(), "gate_logits must have normed's shape");
    TORCH_CHECK(normed.is_contiguous() && gate_logits.is_contiguous(), "normed and gate_logits must be contiguous");
    TORCH_CHECK(gate_logits.device() == normed.device(), "gate_logits must be on normed's device");
    CheckPowerOfTwoBranches(branches);
    const int64_t rows = normed.size(0);
    TORCH_CHECK(normed.size(1) % branches == 0, "normed's column count ", normed.size(1),
                " must be a multiple of branches ", branches);
    const int64_t width = normed.size(1) / branches;
    CheckBranchWidth(width);

    at::Tensor out = at::empty({rows, width}, normed.options());
    if (rows == 0) {
        return out;
    }
    const uint32_t tile =
        SelectColumnTile(static_cast<uint32_t>(width), static_cast<uint32_t>(branches), static_cast<uint32_t>(rows));
    // Defensive guard: unreachable while CheckBranchWidth enforces width % 16 == 0
    // (candidate = 16 always fits TUNED_UB_BUDGET), kept in case MIN_COLUMN_TILE
    // or alignment rules are relaxed later.
    TORCH_CHECK(tile != 0, "no column tile of width ", width, " fits Unified Buffer for ", branches, " branches");

    const uint64_t tasks = static_cast<uint64_t>(rows) * (width / tile);
    const uint32_t blockDim =
        LaunchBlocks(TunedBlocks(EpilogueKernel::BRANCH_MIX, static_cast<uint32_t>(width), static_cast<uint32_t>(rows)),
                     tasks, VectorCoreNum());

    auto npuStream = c10_npu::getCurrentNPUStream();
    normed.record_stream(npuStream);
    gate_logits.record_stream(npuStream);

    uint32_t rowsU32 = static_cast<uint32_t>(rows);
    uint32_t branchesU32 = static_cast<uint32_t>(branches);
    uint32_t widthU32 = static_cast<uint32_t>(width);
    uint32_t tileU32 = tile;
    float invBranches = 1.0f / static_cast<float>(branches);
    EXEC_KERNEL_CMD(hyper_conn_branch_mix_bf16, blockDim, normed, gate_logits, out, rowsU32, branchesU32, widthU32,
                    tileU32, invBranches);
    return out;
}

/* Gated residual combine: broadcast one block output onto every branch.
 *
 * `gate_logits` may be a column slice of a wider matmul output: only a unit
 * column stride is required, and the row pitch is passed to the kernel.
 */
HOST_API at::Tensor hyper_conn_branch_combine(const at::Tensor &block, const at::Tensor &residual,
                                              const at::Tensor &gate_logits, int64_t branches,
                                              const c10::optional<at::Tensor> &out)
{
    CheckBf16Npu(block, "block");
    CheckBf16Npu(residual, "residual");
    CheckBf16Npu(gate_logits, "gate_logits");
    TORCH_CHECK(residual.dim() == 2, "residual must be 2-D, got ", residual.dim(), "-D");
    TORCH_CHECK(block.dim() == 2, "block must be 2-D, got ", block.dim(), "-D");
    TORCH_CHECK(gate_logits.dim() == 2, "gate_logits must be 2-D, got ", gate_logits.dim(), "-D");
    TORCH_CHECK(block.is_contiguous() && residual.is_contiguous(), "block and residual must be contiguous");
    TORCH_CHECK(gate_logits.stride(1) == 1, "gate_logits must have a unit column stride");
    TORCH_CHECK(block.device() == residual.device() && gate_logits.device() == residual.device(),
                "all inputs must share one device");
    CheckPowerOfTwoBranches(branches);
    const int64_t rows = residual.size(0);
    TORCH_CHECK(residual.size(1) % branches == 0, "residual's column count ", residual.size(1),
                " must be a multiple of branches ", branches);
    const int64_t width = residual.size(1) / branches;
    CheckBranchWidth(width);
    TORCH_CHECK(block.size(0) == rows && block.size(1) == width, "block must be [", rows, ", ", width, "]");
    TORCH_CHECK(gate_logits.size(0) == rows && gate_logits.size(1) == branches, "gate_logits must be [", rows, ", ",
                branches, "]");
    TORCH_CHECK(rows == 0 || gate_logits.stride(0) >= branches, "gate_logits' row pitch ", gate_logits.stride(0),
                " must cover its ", branches, " columns");

    at::Tensor result;
    if (out.has_value()) {
        CheckBf16Npu(*out, "out");
        TORCH_CHECK(out->device() == residual.device(), "out must be on residual's device");
        TORCH_CHECK(out->sizes() == residual.sizes(), "out must have residual's shape");
        TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
        TORCH_CHECK(!out->requires_grad(), "out must not require gradients");
        TORCH_CHECK(
            !TensorsOverlap(*out, block) && !TensorsOverlap(*out, residual) && !TensorsOverlap(*out, gate_logits),
            "out must not alias or overlap any input tensor");
        result = *out;
    } else {
        result = at::empty_like(residual);
    }
    if (rows == 0) {
        return result;
    }
    const uint32_t tile =
        SelectColumnTile(static_cast<uint32_t>(width), static_cast<uint32_t>(branches), static_cast<uint32_t>(rows));
    // Defensive guard: unreachable while CheckBranchWidth enforces width % 16 == 0
    // (candidate = 16 always fits TUNED_UB_BUDGET), kept in case MIN_COLUMN_TILE
    // or alignment rules are relaxed later.
    TORCH_CHECK(tile != 0, "no column tile of width ", width, " fits Unified Buffer for ", branches, " branches");

    const uint64_t tasks = static_cast<uint64_t>(rows) * (width / tile);
    const uint32_t blockDim = LaunchBlocks(
        TunedBlocks(EpilogueKernel::BRANCH_COMBINE, static_cast<uint32_t>(width), static_cast<uint32_t>(rows)), tasks,
        VectorCoreNum());

    auto npuStream = c10_npu::getCurrentNPUStream();
    block.record_stream(npuStream);
    residual.record_stream(npuStream);
    gate_logits.record_stream(npuStream);
    if (out.has_value()) {
        result.record_stream(npuStream);
    }

    uint32_t rowsU32 = static_cast<uint32_t>(rows);
    uint32_t branchesU32 = static_cast<uint32_t>(branches);
    uint32_t widthU32 = static_cast<uint32_t>(width);
    uint32_t tileU32 = tile;
    uint32_t logitsStrideU32 = static_cast<uint32_t>(gate_logits.stride(0));
    float invBranches = 1.0f / static_cast<float>(branches);
    EXEC_KERNEL_CMD(hyper_conn_branch_combine_bf16, blockDim, block, residual, gate_logits, result, rowsU32,
                    branchesU32, widthU32, tileU32, logitsStrideU32, invBranches);
    return result;
}

}  // namespace npu_kernel
}  // namespace sglang
