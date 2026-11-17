// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

// Torch schema and ACLNN argument conversion for RainFusionAttention. The
// descriptor semantics match the legacy attentions plugin, including its
// storage-based tensor spans and rank-derived formats.

#include "rain_fusion_attention.h"

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr aclDataType kAtenToAclType[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
    ACL_UINT8,      ACL_INT8,         ACL_INT16,        ACL_INT32,        ACL_INT64,
    ACL_FLOAT16,    ACL_FLOAT,        ACL_DOUBLE,       ACL_DT_UNDEFINED, ACL_COMPLEX64,
    ACL_COMPLEX128, ACL_BOOL,         ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
    ACL_BF16,       ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED};

// Legacy ConvertType(at::Tensor): describe the whole storage span and pick the
// format from the rank. Wrapped-number tensors cannot appear among the
// attention arguments, so the legacy device-copy branch is not reproduced.
sglang::attention::AclTensorPtr ToAclTensor(const at::Tensor &tensor)
{
    if (!tensor.defined()) {
        return nullptr;
    }
    const auto scalarType = tensor.scalar_type();
    const aclDataType aclType = kAtenToAclType[static_cast<int64_t>(scalarType)];
    TORCH_CHECK(aclType != ACL_DT_UNDEFINED, c10::toString(scalarType), " has not been supported");
    const auto itemSize = tensor.itemsize();
    TORCH_CHECK(itemSize != 0, "When ConvertType, tensor item size of cannot be zero.");

    std::vector<int64_t> storageDims;
    if (aclType != ACL_STRING) {
        storageDims.push_back(tensor.storage().nbytes() / itemSize);
    }
    aclFormat format = ACL_FORMAT_ND;
    switch (tensor.dim()) {
        case 3:
            format = ACL_FORMAT_NCL;
            break;
        case 4:
            format = ACL_FORMAT_NCHW;
            break;
        case 5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
            break;
    }
    auto *descriptor =
        aclCreateTensor(tensor.sizes().data(), tensor.dim(), aclType, tensor.strides().data(), tensor.storage_offset(),
                        format, storageDims.data(), storageDims.size(), const_cast<void *>(tensor.storage().data()));
    return {descriptor, [](aclTensor *value) {
                if (value != nullptr) {
                    aclDestroyTensor(value);
                }
            }};
}

sglang::attention::AclIntArrayPtr ToAclIntArray(const at::IntArrayRef &values)
{
    auto *array = aclCreateIntArray(values.data(), values.size());
    return {array, [](aclIntArray *value) {
                if (value != nullptr) {
                    aclDestroyIntArray(value);
                }
            }};
}

std::tuple<at::Tensor, at::Tensor> rain_fusion_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &select_idx,
    const at::Tensor &select_num_idx, at::IntArrayRef blockshape, const c10::optional<at::Tensor> &attn_mask,
    c10::OptionalIntArrayRef actual_seq_qlen, c10::OptionalIntArrayRef actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table, std::string q_input_layout, std::string kv_input_layout,
    int64_t head_num, int64_t mask_type, double scale, int64_t inner_precise, int64_t block_size)
{
    TORCH_CHECK(q_input_layout == "TND" || q_input_layout == "BNSD",
                "q_input_layout only supports 'TND' and 'BNSD' now.");
    TORCH_CHECK(kv_input_layout == "TND" || kv_input_layout == "BNSD",
                "kv_input_layout only supports 'TND' and 'BNSD' now.");
    const at::Tensor attenMask = attn_mask.value_or(at::Tensor());
    const at::Tensor blockTable = block_table.value_or(at::Tensor());
    const at::IntArrayRef actualSeqLengths = actual_seq_qlen.value_or(at::IntArrayRef{});
    const at::IntArrayRef actualSeqLengthsKv = actual_seq_kvlen.value_or(at::IntArrayRef{});

    // aclnn takes mutable layout strings; both parameters live for the whole
    // GetWorkspaceSize call.
    char *qlayoutPtr = q_input_layout.data();
    char *kvlayoutPtr = kv_input_layout.data();

    // Both outputs carry the query options and NPU format, in the legacy order.
    at::Tensor attentionOut =
        at_npu::native::empty_with_format(query.sizes(), query.options(), at_npu::native::get_npu_format(query));
    at::Tensor softmaxLse = at_npu::native::empty_with_format({query.sizes()[0], query.sizes()[1], query.sizes()[2]},
                                                              query.options(), at_npu::native::get_npu_format(query));

    const sglang::attention::RainFusionAttentionArgs args{
        ToAclTensor(query),
        ToAclTensor(key),
        ToAclTensor(value),
        ToAclTensor(select_idx),
        ToAclTensor(select_num_idx),
        ToAclIntArray(blockshape),
        ToAclTensor(attenMask),
        ToAclIntArray(actualSeqLengths),
        ToAclIntArray(actualSeqLengthsKv),
        ToAclTensor(blockTable),
        ToAclTensor(attentionOut),
        ToAclTensor(softmaxLse),
        qlayoutPtr,
        kvlayoutPtr,
        head_num,
        mask_type,
        scale,
        inner_precise,
        block_size,
    };
    sglang::attention::launch_rain_fusion_attention(args);
    return std::make_tuple(attentionOut, softmaxLse);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "rainfusionattention(Tensor query, Tensor key, Tensor value, Tensor select_idx, "
        "Tensor select_num_idx, int[] blockshape, Tensor? attn_mask=None, int[]? actual_seq_qlen=None, "
        "int[]? actual_seq_kvlen=None, Tensor? block_table=None, str q_input_layout='TND', "
        "str kv_input_layout='TND', int head_num=1, int mask_type=0, float scale=1.0, "
        "int inner_precise=1, int block_size=0) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("rainfusionattention", &rain_fusion_attention);
}
