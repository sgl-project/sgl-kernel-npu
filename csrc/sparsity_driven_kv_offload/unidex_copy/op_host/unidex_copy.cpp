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

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_unidex_copy.h"

#include <cstdint>
#include <limits>

namespace sglang {
namespace npu_kernel {

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

}  // namespace

/*
 * For every i where valid_mask[i] is true:
 *   dst[dst_index[i]] = src[src_index[i]]
 *
 * src_ptr and dst_ptr optionally override the Tensor storage addresses with
 * device-visible shared-memory addresses. Their lifetime is owned by the
 * caller and must extend through completion on the current NPU stream.
 */
HOST_API void unidex_copy(const at::Tensor &src, at::Tensor &dst, const at::Tensor &src_index,
                          const at::Tensor &dst_index, const at::Tensor &valid_mask, int64_t src_rows, int64_t dst_rows,
                          int64_t block_bytes, int64_t max_copy, int64_t block_dim, c10::optional<int64_t> src_ptr,
                          c10::optional<int64_t> dst_ptr)
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

    auto npuStream = c10_npu::getCurrentNPUStream();
    if (!useRawSrc) {
        src.record_stream(npuStream);
    }
    if (!useRawDst) {
        dst.record_stream(npuStream);
    }
    src_index.record_stream(npuStream);
    dst_index.record_stream(npuStream);
    valid_mask.record_stream(npuStream);

    void *srcAddr =
        useRawSrc ? reinterpret_cast<void *>(static_cast<uintptr_t>(*src_ptr)) : const_cast<void *>(src.data_ptr());
    void *dstAddr = useRawDst ? reinterpret_cast<void *>(static_cast<uintptr_t>(*dst_ptr)) : dst.data_ptr();
    uint32_t srcRowsU32 = static_cast<uint32_t>(src_rows);
    uint32_t dstRowsU32 = static_cast<uint32_t>(dst_rows);
    uint32_t blockBytesU32 = static_cast<uint32_t>(block_bytes);
    uint32_t maxCopyU32 = static_cast<uint32_t>(max_copy);
    uint32_t blockDimU32 = static_cast<uint32_t>(block_dim);

    EXEC_KERNEL_CMD(unidex_copy, blockDimU32, srcAddr, dstAddr, src_index, dst_index, valid_mask, srcRowsU32,
                    dstRowsU32, blockBytesU32, maxCopyU32);
}

}  // namespace npu_kernel
}  // namespace sglang
