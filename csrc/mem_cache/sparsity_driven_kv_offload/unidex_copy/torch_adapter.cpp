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

#include "aclnn_runtime.h"
#include "aclnn_sgl_unidex_copy.h"
#include "op_kernel/unidex_copy_tiling.h"
#include <array>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <limits>
#include <torch/library.h>

namespace {

constexpr int64_t kMaxBlockBytes = 32 * 1024;
constexpr uint64_t kUint32Max = std::numeric_limits<uint32_t>::max();

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

void CheckSameDevice(const at::Tensor &tensor, const at::Tensor &reference, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as the reference tensor");
}

void CheckFitsUint32(int64_t value, const char *name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= kUint32Max, name, " exceeds uint32 range: ", value);
}

void unidex_copy(const at::Tensor &src, const at::Tensor &dst, const at::Tensor &src_index, const at::Tensor &dst_index,
                 const at::Tensor &valid_mask, int64_t src_rows, int64_t dst_rows, int64_t block_bytes,
                 int64_t max_copy, int64_t block_dim, c10::optional<int64_t> src_ptr, c10::optional<int64_t> dst_ptr)
{
    const bool useRawSrc = src_ptr.has_value();
    const bool useRawDst = dst_ptr.has_value();

    CheckNpuTensor(src_index, "src_index");
    CheckNpuTensor(dst_index, "dst_index");
    CheckNpuTensor(valid_mask, "valid_mask");
    CheckSameDevice(dst_index, src_index, "dst_index");
    CheckSameDevice(valid_mask, src_index, "valid_mask");
    if (!useRawSrc) {
        CheckNpuTensor(src, "src");
        CheckSameDevice(src, src_index, "src");
    }
    if (!useRawDst) {
        CheckNpuTensor(dst, "dst");
        CheckSameDevice(dst, src_index, "dst");
    }

    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src_index.is_contiguous(), "src_index must be contiguous");
    TORCH_CHECK(dst_index.is_contiguous(), "dst_index must be contiguous");
    TORCH_CHECK(valid_mask.is_contiguous(), "valid_mask must be contiguous");
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "src and dst must have the same dtype, got ", src.scalar_type(),
                " and ", dst.scalar_type());
    TORCH_CHECK(src_index.scalar_type() == at::kLong, "src_index must be int64, got ", src_index.scalar_type());
    TORCH_CHECK(dst_index.scalar_type() == at::kLong, "dst_index must be int64, got ", dst_index.scalar_type());
    TORCH_CHECK(valid_mask.scalar_type() == at::kBool || valid_mask.scalar_type() == at::kByte,
                "valid_mask must be bool or uint8, got ", valid_mask.scalar_type());
    TORCH_CHECK(src_index.dim() == 1, "src_index must be 1-D, got ", src_index.dim());
    TORCH_CHECK(dst_index.dim() == 1, "dst_index must be 1-D, got ", dst_index.dim());
    TORCH_CHECK(valid_mask.dim() == 1, "valid_mask must be 1-D, got ", valid_mask.dim());

    TORCH_CHECK(!useRawSrc || *src_ptr > 0, "src_ptr must be a non-zero address");
    TORCH_CHECK(!useRawDst || *dst_ptr > 0, "dst_ptr must be a non-zero address");
    TORCH_CHECK(src_rows > 0, "src_rows must be positive, got ", src_rows);
    TORCH_CHECK(dst_rows > 0, "dst_rows must be positive, got ", dst_rows);
    TORCH_CHECK(block_bytes > 0, "block_bytes must be positive, got ", block_bytes);
    TORCH_CHECK(block_bytes <= kMaxBlockBytes, "block_bytes exceeds the supported 32 KiB limit: ", block_bytes);
    TORCH_CHECK(max_copy >= 0, "max_copy must be non-negative, got ", max_copy);
    TORCH_CHECK(block_dim > 0, "block_dim must be positive, got ", block_dim);

    CheckFitsUint32(src_rows, "src_rows");
    CheckFitsUint32(dst_rows, "dst_rows");
    CheckFitsUint32(block_bytes, "block_bytes");
    CheckFitsUint32(max_copy, "max_copy");
    CheckFitsUint32(block_dim, "block_dim");

    TORCH_CHECK(src_index.numel() >= max_copy, "src_index has ", src_index.numel(),
                " elements, fewer than max_copy=", max_copy);
    TORCH_CHECK(dst_index.numel() >= max_copy, "dst_index has ", dst_index.numel(),
                " elements, fewer than max_copy=", max_copy);
    TORCH_CHECK(valid_mask.numel() >= max_copy, "valid_mask has ", valid_mask.numel(),
                " elements, fewer than max_copy=", max_copy);

    const uint64_t srcRequiredBytes = static_cast<uint64_t>(src_rows) * static_cast<uint64_t>(block_bytes);
    const uint64_t dstRequiredBytes = static_cast<uint64_t>(dst_rows) * static_cast<uint64_t>(block_bytes);
    TORCH_CHECK(srcRequiredBytes <= kUint32Max, "src_rows * block_bytes exceeds the kernel uint32 address range");
    TORCH_CHECK(dstRequiredBytes <= kUint32Max, "dst_rows * block_bytes exceeds the kernel uint32 address range");
    if (!useRawSrc) {
        const uint64_t srcAvailableBytes =
            static_cast<uint64_t>(src.numel()) * static_cast<uint64_t>(src.element_size());
        TORCH_CHECK(srcRequiredBytes <= srcAvailableBytes, "src storage is too small: need ", srcRequiredBytes,
                    " bytes, got ", srcAvailableBytes);
    }
    if (!useRawDst) {
        const uint64_t dstAvailableBytes =
            static_cast<uint64_t>(dst.numel()) * static_cast<uint64_t>(dst.element_size());
        TORCH_CHECK(dstRequiredBytes <= dstAvailableBytes, "dst storage is too small: need ", dstRequiredBytes,
                    " bytes, got ", dstAvailableBytes);
    }

    if (max_copy == 0) {
        return;
    }

    const c10::DeviceGuard deviceGuard(src_index.device());
    auto stream = c10_npu::getCurrentNPUStream();
    if (!useRawSrc) {
        src.record_stream(stream);
    }
    if (!useRawDst) {
        dst.record_stream(stream);
    }
    src_index.record_stream(stream);
    dst_index.record_stream(stream);
    valid_mask.record_stream(stream);

    auto fields = std::make_shared<std::array<int64_t, optiling::UNIDEX_COPY_ATTR_COUNT>>();
    (*fields)[optiling::SRC_ADDR] = useRawSrc ? *src_ptr : reinterpret_cast<int64_t>(src.data_ptr());
    (*fields)[optiling::DST_ADDR] = useRawDst ? *dst_ptr : reinterpret_cast<int64_t>(dst.data_ptr());
    (*fields)[optiling::SRC_ROWS] = src_rows;
    (*fields)[optiling::DST_ROWS] = dst_rows;
    (*fields)[optiling::BLOCK_BYTES] = block_bytes;
    (*fields)[optiling::MAX_COPY] = max_copy;
    (*fields)[optiling::BLOCK_DIM] = block_dim;
    auto transfer = std::shared_ptr<aclIntArray>(aclCreateIntArray(fields->data(), fields->size()), aclDestroyIntArray);
    TORCH_CHECK(transfer, "aclCreateIntArray failed: ", aclGetRecentErrMsg());
    std::array<at::Tensor, 5> tensors = {src, dst, src_index, dst_index, valid_mask};
    std::array<std::shared_ptr<aclTensor>, 3> descriptors = {
        sgl_cann::tensor_descriptor(src_index, ACL_INT64),
        sgl_cann::tensor_descriptor(dst_index, ACL_INT64),
        sgl_cann::tensor_descriptor(valid_mask, ACL_UINT8),
    };

    static auto query = reinterpret_cast<decltype(&aclnnSglUnidexCopyGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglUnidexCopyGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglUnidexCopy)>(sgl_cann::op_api_symbol("aclnnSglUnidexCopy"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), transfer.get(),
                        &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "unidex_copy workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    at::Tensor workspace;
    if (workspaceSize) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, src_index.options().dtype(at::kByte));
    }
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, fields, transfer, workspace, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "unidex_copy execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglUnidexCopy", launch);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "unidex_copy(Tensor src, Tensor(a!) dst, Tensor src_index, "
        "Tensor dst_index, Tensor valid_mask, int src_rows, int dst_rows, "
        "int block_bytes, int max_copy, int block_dim=8, "
        "int? src_ptr=None, int? dst_ptr=None) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("unidex_copy", &unidex_copy);
}
