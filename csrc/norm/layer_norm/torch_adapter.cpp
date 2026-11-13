// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

// Torch schema and ACLNN argument conversion for LayerNorm. The descriptor
// semantics match the legacy attentions plugin, including its storage-based
// tensor spans and rank-derived formats.

#include "layer_norm.h"

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <torch/library.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <functional>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace {

constexpr aclDataType kAtenToAclType[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
    ACL_UINT8,      ACL_INT8,         ACL_INT16,        ACL_INT32,        ACL_INT64,
    ACL_FLOAT16,    ACL_FLOAT,        ACL_DOUBLE,       ACL_DT_UNDEFINED, ACL_COMPLEX64,
    ACL_COMPLEX128, ACL_BOOL,         ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
    ACL_BF16,       ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED};

// Legacy ConvertType(at::Tensor): describe the whole storage span and pick the
// format from the rank. Wrapped-number tensors cannot appear among the LayerNorm
// arguments, so the legacy device-copy branch is not reproduced.
sglang::norm::AclTensorPtr ToAclTensor(const at::Tensor &tensor)
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

sglang::norm::AclIntArrayPtr ToAclIntArray(const at::IntArrayRef &values)
{
    auto *array = aclCreateIntArray(values.data(), values.size());
    return {array, [](aclIntArray *value) {
                if (value != nullptr) {
                    aclDestroyIntArray(value);
                }
            }};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm(const at::Tensor &input, at::IntArrayRef normalized_shape,
                                                         const c10::optional<at::Tensor> &weight,
                                                         const c10::optional<at::Tensor> &bias, double eps,
                                                         int64_t impl_mode)
{
    const at::Tensor weight_op = weight.value_or(at::Tensor());
    const at::Tensor bias_op = bias.value_or(at::Tensor());

    // Both the output and the statistics follow the legacy allocation: input
    // options and NPU format, with the mean/rstd shape derived from the axes
    // that are not normalized.
    at::Tensor output =
        at_npu::native::empty_with_format(input.sizes(), input.options(), at_npu::native::get_npu_format(input));
    at::Tensor mean_out;
    at::Tensor rstd_out;

    const int64_t normNdim = static_cast<int64_t>(normalized_shape.size());
    const int64_t inputNdim = input.dim();
    TORCH_CHECK(normNdim > 0 && normNdim <= inputNdim,
                "normalized_shape must be non-empty and no longer than input rank");
    const int64_t beginAxis = inputNdim - normNdim;

    const auto input_shape = input.sizes();
    const int64_t m =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + beginAxis, 1LL, std::multiplies<int64_t>());
    // Shape and dtype of mean/rstd depend on the m value and the input dtype.
    if (m <= 0) {
        mean_out = at_npu::native::empty_with_format({m}, input.options(), at_npu::native::get_npu_format(input));
        rstd_out = at_npu::native::empty_with_format({m}, input.options(), at_npu::native::get_npu_format(input));
    } else {
        at::SmallVector<int64_t, 8> mean_shape;  // The operator supports at most 8 dimensions.
        for (int64_t index = 0; index < beginAxis; index++) {
            mean_shape.emplace_back(input.size(index));
        }
        for (int64_t index = beginAxis; index < inputNdim; index++) {
            mean_shape.emplace_back(1);
        }
        mean_out =
            at_npu::native::empty_with_format(mean_shape, input.options(), at_npu::native::get_npu_format(input));
        rstd_out =
            at_npu::native::empty_with_format(mean_shape, input.options(), at_npu::native::get_npu_format(input));
    }

    const sglang::norm::LayerNormArgs args{ToAclTensor(input),
                                           ToAclIntArray(normalized_shape),
                                           ToAclTensor(weight_op),
                                           ToAclTensor(bias_op),
                                           eps,
                                           ToAclTensor(output),
                                           ToAclTensor(mean_out),
                                           ToAclTensor(rstd_out),
                                           impl_mode};
    sglang::norm::launch_layer_norm(args);
    return std::make_tuple(output, mean_out, rstd_out);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "layernorm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, "
        "float eps=1e-05, int impl_mode=0) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("layernorm", &layernorm);
}
