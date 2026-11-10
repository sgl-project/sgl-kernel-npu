// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

// Framework bridge for the migrated MLA preprocess CANN operator. Validation
// mirrors the legacy Host wrapper; all computation remains in the copied kernel.

#include "aclnn_runtime.h"
#include "aclnn_sgl_mla_preprocess.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <initializer_list>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace {

aclDataType ToAclType(const at::Tensor &tensor)
{
    switch (tensor.scalar_type()) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        case at::kChar:
            return ACL_INT8;
        case at::kLong:
            return ACL_INT64;
        default:
            TORCH_CHECK(false, "mla_preprocess does not support dtype ", tensor.scalar_type());
            return ACL_FLOAT;
    }
}

int ParseCacheMode(const c10::optional<c10::string_view> &mode)
{
    const c10::string_view value = mode.value_or("krope_ctkv");
    if (value == "kvcache") {
        return 0;
    }
    if (value == "krope_ctkv") {
        return 1;
    }
    if (value == "int8_nzcache") {
        return 2;
    }
    if (value == "nzcache") {
        return 3;
    }
    TORCH_CHECK(false, "Unsupported cache_mode value: '", value, "'");
    return 1;
}

int ParseQuantMode(const c10::optional<c10::string_view> &mode)
{
    const c10::string_view value = mode.value_or("per_token_quant_symm");
    if (value == "per_tensor_quant_asymm") {
        return 0;
    }
    if (value == "per_token_quant_symm") {
        return 1;
    }
    TORCH_CHECK(false, "Unsupported quant_mode value: '", value, "'");
    return 0;
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> sgl_mla_preprocess(
    const at::Tensor &hidden_state, const at::Tensor &gamma0, const at::Tensor &beta0, const at::Tensor &wdqkv,
    const at::Tensor &descale0, const at::Tensor &gamma1, const at::Tensor &beta1, const at::Tensor &wuq,
    const at::Tensor &descale1, const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const at::Tensor &quant_scale0, const at::Tensor &quant_offset0, const at::Tensor &bias0,
    const at::Tensor &quant_scale1, const at::Tensor &quant_offset1, const at::Tensor &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    const c10::optional<c10::string_view> &cache_mode, const c10::optional<c10::string_view> &quant_mode,
    at::Tensor &q_out0, at::Tensor &kv_cache_out0, at::Tensor &q_out1, at::Tensor &kv_cache_out1)
{
    TORCH_CHECK(hidden_state.device().type() == c10::DeviceType::PrivateUse1, "hidden_state must be on an NPU");
    for (const at::Tensor *tensor : std::initializer_list<const at::Tensor *>{
             &gamma0,      &beta0,         &wdqkv,         &descale0,     &gamma1,       &beta1,         &wuq,
             &descale1,    &gamma2,        &cos,           &sin,          &wuk,          &kv_cache,      &kv_cache_rope,
             &slotmapping, &quant_scale0,  &quant_offset0, &bias0,        &quant_scale1, &quant_offset1, &bias1,
             &q_out0,      &kv_cache_out0, &q_out1,        &kv_cache_out1}) {
        TORCH_CHECK(tensor->device() == hidden_state.device(), "all mla_preprocess tensors must be on the same NPU");
    }

    const int cacheMode = ParseCacheMode(cache_mode);
    const int quantMode = ParseQuantMode(quant_mode);
    const bool isBf16 = hidden_state.scalar_type() == at::kBFloat16;
    TORCH_CHECK(isBf16 || hidden_state.scalar_type() == at::kHalf, "hidden_state must be float16 or bfloat16");

    TORCH_CHECK(hidden_state.dim() == 2, "hidden_state should be [tokenNum, hiddenSize]");
    // The weights arrive either as logical 2D tensors or as the 4D FRACTAL_NZ
    // storage that the Kernel consumes. The Kernel only sees the raw pointer.
    TORCH_CHECK(wdqkv.dim() == 2 || wdqkv.dim() == 4, "wdqkv should be ND [2112, H] or its 4D NZ form");
    TORCH_CHECK(wuq.dim() == 2 || wuq.dim() == 4, "wuq should be ND [headNum*192, 1536] or its 4D NZ form");
    TORCH_CHECK(wuk.dim() == 3, "wuk should be [headNum, 128, 512]");
    TORCH_CHECK(kv_cache.dim() == 4 && kv_cache_rope.dim() == 4,
                "kv_cache and kv_cache_rope should be [blockNum, blockSize, 1, headDim]");
    TORCH_CHECK(slotmapping.dim() == 1 && slotmapping.scalar_type() == at::kInt,
                "slotmapping should be a 1D int32 tensor");
    TORCH_CHECK(hidden_state.size(0) >= 1 && hidden_state.size(0) <= 1024, "tokenNum must be in [1, 1024]");
    TORCH_CHECK(slotmapping.size(0) == hidden_state.size(0), "slotmapping size must equal tokenNum");
    TORCH_CHECK(wuk.size(0) > 0 && wuk.size(2) == 512, "wuk shape must be [headNum, 128, 512]");

    const bool int8Cache = cacheMode == 2;
    if (!isBf16) {
        TORCH_CHECK(cacheMode <= 1, "float16 only supports cache_mode 0 and 1");
        TORCH_CHECK(descale0.scalar_type() == at::kLong && descale1.scalar_type() == at::kLong,
                    "float16 requires int64 descale tensors");
    } else {
        TORCH_CHECK(descale0.scalar_type() == at::kFloat && descale1.scalar_type() == at::kFloat,
                    "bfloat16 requires float32 descale tensors");
    }
    TORCH_CHECK(kv_cache.scalar_type() == (int8Cache ? at::kChar : hidden_state.scalar_type()),
                "kv_cache dtype must match hidden_state, or be int8 for cache_mode=2");
    TORCH_CHECK(kv_cache_rope.scalar_type() == hidden_state.scalar_type(),
                "kv_cache_rope dtype must match hidden_state");
    TORCH_CHECK(q_out1.scalar_type() == hidden_state.scalar_type(), "q_out1 dtype must match hidden_state");
    TORCH_CHECK(kv_cache_out1.scalar_type() == hidden_state.scalar_type(),
                "kv_cache_out1 dtype must match hidden_state");
    if (int8Cache) {
        TORCH_CHECK(q_out0.scalar_type() == at::kChar && kv_cache_out0.scalar_type() == at::kChar,
                    "cache_mode=2 requires int8 q_out0 and kv_cache_out0");
        TORCH_CHECK(ctkv_scale.has_value() && ctkv_scale->defined() && ctkv_scale->numel() > 0,
                    "cache_mode=2 requires ctkv_scale");
        TORCH_CHECK(q_nope_scale.has_value() && q_nope_scale->defined() && q_nope_scale->numel() > 0,
                    "cache_mode=2 requires q_nope_scale");
    } else {
        TORCH_CHECK(q_out0.scalar_type() == hidden_state.scalar_type(), "q_out0 dtype must match hidden_state");
        TORCH_CHECK(kv_cache_out0.scalar_type() == hidden_state.scalar_type(),
                    "kv_cache_out0 dtype must match hidden_state");
    }
    const c10::DeviceGuard deviceGuard(hidden_state.device());

    // Optional scales are unused outside cache_mode=2 and may be omitted.
    const bool hasCtkvScale = ctkv_scale.has_value() && ctkv_scale->defined();
    const bool hasQnopeScale = q_nope_scale.has_value() && q_nope_scale->defined();

    auto hiddenDesc = sgl_cann::tensor_descriptor(hidden_state, ToAclType(hidden_state));
    auto gamma0Desc = sgl_cann::tensor_descriptor(gamma0, ToAclType(gamma0));
    auto beta0Desc = sgl_cann::tensor_descriptor(beta0, ToAclType(beta0));
    auto wdqkvDesc = sgl_cann::tensor_descriptor(wdqkv, ACL_INT8);
    auto descale0Desc = sgl_cann::tensor_descriptor(descale0, ToAclType(descale0));
    auto gamma1Desc = sgl_cann::tensor_descriptor(gamma1, ToAclType(gamma1));
    auto beta1Desc = sgl_cann::tensor_descriptor(beta1, ToAclType(beta1));
    auto wuqDesc = sgl_cann::tensor_descriptor(wuq, ACL_INT8);
    auto descale1Desc = sgl_cann::tensor_descriptor(descale1, ToAclType(descale1));
    auto gamma2Desc = sgl_cann::tensor_descriptor(gamma2, ToAclType(gamma2));
    auto cosDesc = sgl_cann::tensor_descriptor(cos, ToAclType(cos));
    auto sinDesc = sgl_cann::tensor_descriptor(sin, ToAclType(sin));
    auto wukDesc = sgl_cann::tensor_descriptor(wuk, ToAclType(wuk));
    auto kvCacheDesc = sgl_cann::tensor_descriptor(kv_cache, ToAclType(kv_cache));
    auto kvCacheRopeDesc = sgl_cann::tensor_descriptor(kv_cache_rope, ToAclType(kv_cache_rope));
    auto slotMappingDesc = sgl_cann::tensor_descriptor(slotmapping, ACL_INT32);
    auto quantScale0Desc = sgl_cann::tensor_descriptor(quant_scale0, ToAclType(quant_scale0));
    auto quantOffset0Desc = sgl_cann::tensor_descriptor(quant_offset0, ACL_INT8);
    auto bias0Desc = sgl_cann::tensor_descriptor(bias0, ACL_INT32);
    auto quantScale1Desc = sgl_cann::tensor_descriptor(quant_scale1, ToAclType(quant_scale1));
    auto quantOffset1Desc = sgl_cann::tensor_descriptor(quant_offset1, ACL_INT8);
    auto bias1Desc = sgl_cann::tensor_descriptor(bias1, ACL_INT32);
    auto ctkvScaleDesc =
        hasCtkvScale ? sgl_cann::tensor_descriptor(*ctkv_scale, ToAclType(*ctkv_scale)) : std::shared_ptr<aclTensor>{};
    auto qnopeScaleDesc = hasQnopeScale ? sgl_cann::tensor_descriptor(*q_nope_scale, ToAclType(*q_nope_scale))
                                        : std::shared_ptr<aclTensor>{};
    auto qOut0Desc = sgl_cann::tensor_descriptor(q_out0, ToAclType(q_out0));
    auto kvCacheOut0Desc = sgl_cann::tensor_descriptor(kv_cache_out0, ToAclType(kv_cache_out0));
    auto qOut1Desc = sgl_cann::tensor_descriptor(q_out1, ToAclType(q_out1));
    auto kvCacheOut1Desc = sgl_cann::tensor_descriptor(kv_cache_out1, ToAclType(kv_cache_out1));

    static auto query = reinterpret_cast<decltype(&aclnnSglMlaPreprocessGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglMlaPreprocessGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglMlaPreprocess)>(sgl_cann::op_api_symbol("aclnnSglMlaPreprocess"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = query(
        hiddenDesc.get(), gamma0Desc.get(), beta0Desc.get(), wdqkvDesc.get(), descale0Desc.get(), gamma1Desc.get(),
        beta1Desc.get(), wuqDesc.get(), descale1Desc.get(), gamma2Desc.get(), cosDesc.get(), sinDesc.get(),
        wukDesc.get(), kvCacheDesc.get(), kvCacheRopeDesc.get(), slotMappingDesc.get(), quantScale0Desc.get(),
        quantOffset0Desc.get(), bias0Desc.get(), quantScale1Desc.get(), quantOffset1Desc.get(), bias1Desc.get(),
        ctkvScaleDesc.get(), qnopeScaleDesc.get(), static_cast<int64_t>(cacheMode), static_cast<int64_t>(quantMode),
        qOut0Desc.get(), kvCacheOut0Desc.get(), qOut1Desc.get(), kvCacheOut1Desc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "mla_preprocess workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");

    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, hidden_state.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{
        hidden_state,  gamma0,      beta0,         wdqkv,         descale0,      gamma1,       beta1,
        wuq,           descale1,    gamma2,        cos,           sin,           wuk,          kv_cache,
        kv_cache_rope, slotmapping, quant_scale0,  quant_offset0, bias0,         quant_scale1, quant_offset1,
        bias1,         q_out0,      kv_cache_out0, q_out1,        kv_cache_out1, workspace};
    if (hasCtkvScale) tensors.push_back(*ctkv_scale);
    if (hasQnopeScale) tensors.push_back(*q_nope_scale);
    for (const auto &tensor : tensors) {
        tensor.record_stream(stream);
    }
    auto aclStream = stream.stream(false);

    std::vector<std::shared_ptr<aclTensor>> descriptors{
        hiddenDesc, gamma0Desc,      beta0Desc,        wdqkvDesc,       descale0Desc,    gamma1Desc,
        beta1Desc,  wuqDesc,         descale1Desc,     gamma2Desc,      cosDesc,         sinDesc,
        wukDesc,    kvCacheDesc,     kvCacheRopeDesc,  slotMappingDesc, quantScale0Desc, quantOffset0Desc,
        bias0Desc,  quantScale1Desc, quantOffset1Desc, bias1Desc,       ctkvScaleDesc,   qnopeScaleDesc,
        qOut0Desc,  kvCacheOut0Desc, qOut1Desc,        kvCacheOut1Desc};
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "mla_preprocess ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglMlaPreprocess", launch);

    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "mla_preprocess(Tensor hidden_state, Tensor gamma0, Tensor beta0, Tensor wdqkv, Tensor descale0, "
        "Tensor gamma1, Tensor beta1, Tensor wuq, Tensor descale1, Tensor gamma2, Tensor cos, Tensor sin, "
        "Tensor wuk, Tensor kv_cache, Tensor kv_cache_rope, Tensor slotmapping, Tensor quant_scale0, "
        "Tensor quant_offset0, Tensor bias0, Tensor quant_scale1, Tensor quant_offset1, Tensor bias1, "
        "*, Tensor? ctkv_scale=None, Tensor? q_nope_scale=None, str? cache_mode=None, str? quant_mode=None, "
        "Tensor(a!) q_out0, Tensor(b!) kv_cache_out0, Tensor(c!) q_out1, Tensor(d!) kv_cache_out1) "
        "-> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("mla_preprocess", &sgl_mla_preprocess);
}
