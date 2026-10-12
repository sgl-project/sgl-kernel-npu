/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hc_pre_def.h"
#include "hc_pre_tiling.h"
#include "aclrtlaunch_hc_pre.h"
#include "sgl_kenel_npu_ops.h"

namespace sglang::npu_kernel {
constexpr int64_t HC_PRE_HC_LIMIT = 4;
constexpr int64_t HC_PRE_MIX_HC_LIMIT = 24;

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(const at::Tensor &x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<int64_t, 8> y_size;
    at::SmallVector<int64_t, 8> post_size;
    at::SmallVector<int64_t, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.size(0);
        auto size = x.size(1);
        auto d = x.size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3) {
        auto bs = x.size(0);
        auto d = x.size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_pre_output_tensor(const at::Tensor &x, int64_t hc_mult)
{
    at::SmallVector<int64_t, 8> pre_size;
    if (x.dim() == 4) {
        pre_size = {x.size(0), x.size(1), hc_mult};
    } else if (x.dim() == 3) {
        pre_size = {x.size(0), hc_mult};
    }
    return at::empty(pre_size, x.options().dtype(at::kFloat));
}

void check_hc_pre_shape_and_dtype(const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale,
                                  const at::Tensor &hc_base, const c10::optional<at::Tensor> &pre_mix, int64_t hc_mult)
{
    constexpr int64_t HC_SCALE_SIZE = 3;
    auto x_dims = x.dim();
    TORCH_CHECK(x_dims == 3 || x_dims == 4, "Input tensor x's dim num should be 3 or 4, actual ", x_dims, ".");
    for (auto i = 0; i < x_dims; i++) {
        TORCH_CHECK(x.size(i) > 0, "Input tensor x's shape should be positive, but x.shape[", i, "] is ", x.size(i),
                    ".");
    }

    auto hc = x_dims == 4 ? x.size(2) : x.size(1);
    auto d = x_dims == 4 ? x.size(3) : x.size(2);
    TORCH_CHECK(hc_mult == HC_PRE_HC_LIMIT, "hc_mult only supports ", HC_PRE_HC_LIMIT, ", actual ", hc_mult, ".");
    TORCH_CHECK(hc == HC_PRE_HC_LIMIT, "The hc of x only supports ", HC_PRE_HC_LIMIT, ", actual ", hc, ".");
    TORCH_CHECK(hc_fn.dim() == 2, "Input tensor hc_fn's dim num should be 2, actual ", hc_fn.dim(), ".");
    TORCH_CHECK(hc_fn.size(0) == HC_PRE_MIX_HC_LIMIT, "The hc_fn.shape[0] only supports ", HC_PRE_MIX_HC_LIMIT,
                ", actual ", hc_fn.size(0), ".");
    TORCH_CHECK(hc_fn.size(1) == hc * d, "The hc_fn.shape[1] should be hc * d, actual hc_fn.shape[1] is ",
                hc_fn.size(1), ", hc is ", hc, ", d is ", d, ".");
    TORCH_CHECK(hc_scale.dim() == 1, "Input tensor hc_scale's dim num should be 1, actual ", hc_scale.dim(), ".");
    TORCH_CHECK(hc_scale.size(0) == HC_SCALE_SIZE, "Input tensor hc_scale's shape should be [", HC_SCALE_SIZE,
                "], actual [", hc_scale.size(0), "].");
    TORCH_CHECK(hc_base.dim() == 1, "Input tensor hc_base's dim num should be 1, actual ", hc_base.dim(), ".");
    TORCH_CHECK(hc_base.size(0) == HC_PRE_MIX_HC_LIMIT, "The hc_base.shape[0] only supports ", HC_PRE_MIX_HC_LIMIT,
                ", actual ", hc_base.size(0), ".");

    TORCH_CHECK(x.dtype() == at::kBFloat16, "x's dtype should be BFLOAT16.");
    TORCH_CHECK(hc_fn.dtype() == at::kFloat, "hc_fn's dtype should be FLOAT32.");
    TORCH_CHECK(hc_scale.dtype() == at::kFloat, "hc_scale's dtype should be FLOAT32.");
    TORCH_CHECK(hc_base.dtype() == at::kFloat, "hc_base's dtype should be FLOAT32.");
    if (pre_mix.has_value() && pre_mix->defined()) {
        TORCH_CHECK(pre_mix->dtype() == at::kFloat, "pre_mix's dtype should be FLOAT32.");
        TORCH_CHECK(pre_mix->dim() == x_dims - 1, "pre_mix's dim num should be ", x_dims - 1, ", actual ",
                    pre_mix->dim(), ".");
        for (auto i = 0; i < x_dims - 1; i++) {
            TORCH_CHECK(pre_mix->size(i) == x.size(i), "pre_mix.shape[", i, "] should equal x.shape[", i, "], actual ",
                        pre_mix->size(i), " vs ", x.size(i), ".");
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
run_hc_pre_fusion(const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale, const at::Tensor &hc_base,
                  const c10::optional<at::Tensor> &pre_mix, int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
                  double hc_eps)
{
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    at::Tensor pre = construct_hc_pre_pre_output_tensor(x, hc_mult);
    hc_pre::HcPre op("hc_pre");
    op.SetAttrAny("hc_mult", hc_mult);
    op.SetAttrAny("hc_sinkhorn_iters", hc_sinkhorn_iters);
    op.SetAttrAny("hc_eps", static_cast<float>(hc_eps));
    op.SetAttrAny("norm_eps", static_cast<float>(norm_eps));
    auto hcContext = std::make_shared<hc_pre::TilingContext>();
    std::shared_ptr<ge_helper::TilingContext> context = hcContext;
    auto scalarType = x.scalar_type();
    op.SetToContext(context, scalarType);
    context->RegisterTensor(x, true);
    context->RegisterTensor(hc_fn, true);
    context->RegisterTensor(hc_scale, true);
    context->RegisterTensor(hc_base, true);
    context->RegisterTensor(pre_mix, true);
    context->RegisterTensor(y, false);
    context->RegisterTensor(post, false);
    context->RegisterTensor(comb_frag, false);
    context->RegisterTensor(pre, false);
    TORCH_CHECK(optiling::TilingForHcPre(hcContext.get()) == ge::GRAPH_SUCCESS, "hc_pre: tiling failed");
    auto tilingTensor = context->GetTilingTensor(hcContext->serialized);
    // Upstream GetWorkspaceSizes(1)[0] includes its workspace reserve already.
    auto workspace = at::empty({static_cast<int64_t>(*context->GetWorkspaceSizes(1))}, x.options().dtype(at::kByte));
    auto preMixLaunch = pre_mix.value_or(at::Tensor());
    void *preMixPtr = preMixLaunch.defined() ? preMixLaunch.data_ptr() : nullptr;
    auto blockDim = hcContext->blockDim;
    EXEC_KERNEL_CMD(hc_pre, blockDim, x, hc_fn, hc_scale, hc_base, preMixPtr, y, post, comb_frag, pre, workspace,
                    tilingTensor, hcContext->tilingKey);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag, pre);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2_npu(const at::Tensor &x, const at::Tensor &hc_fn,
                                                                 const at::Tensor &hc_scale, const at::Tensor &hc_base,
                                                                 int64_t hc_mult, int64_t hc_sinkhorn_iters,
                                                                 double norm_eps, double hc_eps)
{
    const c10::optional<at::Tensor> pre_mix = c10::nullopt;
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult);
    auto outputs =
        run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
    return {std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_hc_pre_v3_npu(const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale, const at::Tensor &hc_base,
                  const c10::optional<at::Tensor> &pre_mix, int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
                  double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult);
    return run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_meta_output_tensor(const at::Tensor &x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<c10::SymInt, 8> y_size;
    at::SmallVector<c10::SymInt, 8> post_size;
    at::SmallVector<c10::SymInt, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.sym_size(0);
        auto size = x.sym_size(1);
        auto d = x.sym_size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3) {
        auto bs = x.sym_size(0);
        auto d = x.sym_size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty_symint(c10::SymIntArrayRef(y_size), x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty_symint(c10::SymIntArrayRef(post_size), x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty_symint(c10::SymIntArrayRef(comb_frag_size), x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_meta_pre_output_tensor(const at::Tensor &x, int64_t hc_mult)
{
    at::SmallVector<c10::SymInt, 8> pre_size;
    if (x.dim() == 4) {
        pre_size = {x.sym_size(0), x.sym_size(1), hc_mult};
    } else if (x.dim() == 3) {
        pre_size = {x.sym_size(0), hc_mult};
    }
    return at::empty_symint(c10::SymIntArrayRef(pre_size), x.options().dtype(at::kFloat));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_meta(const at::Tensor &x, const at::Tensor &hc_fn,
                                                               const at::Tensor &hc_scale, const at::Tensor &hc_base,
                                                               int64_t hc_mult, int64_t hc_sinkhorn_iters,
                                                               double norm_eps, double hc_eps)
{
    auto output_tensors = construct_hc_pre_meta_output_tensor(x, hc_mult);
    return {std::get<0>(output_tensors), std::get<1>(output_tensors), std::get<2>(output_tensors)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_hc_pre_v3_meta(const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale, const at::Tensor &hc_base,
                   const c10::optional<at::Tensor> &pre_mix, int64_t hc_mult, int64_t hc_sinkhorn_iters,
                   double norm_eps, double hc_eps)
{
    auto output_tensors = construct_hc_pre_meta_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    at::Tensor pre = construct_hc_pre_meta_pre_output_tensor(x, hc_mult);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag, pre);
}

}  // namespace sglang::npu_kernel
