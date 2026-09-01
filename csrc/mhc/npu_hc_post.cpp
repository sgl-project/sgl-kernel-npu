/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "pytorch_npu_helper.h"

namespace sglang::npu_kernel {

constexpr std::string_view HC_POST_NAME = "aclnnHcPost";

const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

// npu tensor max size
const int SIZE = 8;

const int HC_LIMIT = 4;
const int D_LIMIT_3584 = 3584;
const int D_LIMIT_4096 = 4096;
const int D_LIMIT_7168 = 7168;

// step1, 工具函数，推导输出shape
at::Tensor construct_hc_post_output_tensor(const at::Tensor& residual)
{
    auto residualDims = residual.dim();
    // 使用 sym_size/empty_symint 保留动态维，避免动态 shape 下 .sizes() 具象化。
    c10::SymDimVector out_shape;
    for (int64_t i = 0; i < residualDims; i++) {
        out_shape.push_back(residual.sym_size(i));
    }
    at::Tensor out = at::empty_symint(out_shape, residual.options().dtype(residual.dtype()));
    return out;
}

// 工具函数，检查 x=[bs, d] 场景下 residual/post/com 的 shape
void check_hc_post_shape_2d(
    const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post, const at::Tensor& com)
{
    auto batch_sequence = x.sym_size(DIM_0);
    auto d = x.sym_size(DIM_1);
    TORCH_CHECK(d == D_LIMIT_3584 || d == D_LIMIT_4096 || d == D_LIMIT_7168,
                "The d of x only support ", D_LIMIT_3584, ", ", D_LIMIT_4096, " or ", D_LIMIT_7168,
                ", actual ", d, ".");
    // check residual: [bs, hc, d]
    TORCH_CHECK(residual.dim() == DIM_3,
                "Input tensor residual's dim num should be 3, actual ", residual.dim(), ".");
    TORCH_CHECK(residual.sym_size(DIM_0) == batch_sequence,
                "The residual.shape[0] should be batch_sequence, actual residual.shape[0] is ",
                residual.sym_size(DIM_0), ", batch_sequence is ", batch_sequence, ".");
    auto hc = residual.sym_size(DIM_1);
    TORCH_CHECK(hc == HC_LIMIT, "The hc of residual only support ", HC_LIMIT, ", actual ", hc, ".");
    TORCH_CHECK(residual.sym_size(DIM_2) == d,
                "The residual.shape[2] should be d, actual residual.shape[2] is ",
                residual.sym_size(DIM_2), ", d is ", d, ".");
    // check post: [bs, hc]
    TORCH_CHECK(post.dim() == DIM_2, "Input tensor post's dim num should be 2, actual ", post.dim(), ".");
    TORCH_CHECK(post.sym_size(DIM_0) == batch_sequence,
                "The post.shape[0] should be batch_sequence, actual post.shape[0] is ",
                post.sym_size(DIM_0), ", batch_sequence is ", batch_sequence, ".");
    TORCH_CHECK(post.sym_size(DIM_1) == hc,
                "The post.shape[1] should be hc, actual post.shape[1] is ",
                post.sym_size(DIM_1), ", hc is ", hc, ".");
    // check com: [bs, hc, hc]
    TORCH_CHECK(com.dim() == DIM_3, "Input tensor com's dim num should be 3, actual ", com.dim(), ".");
    TORCH_CHECK(com.sym_size(DIM_0) == batch_sequence,
                "The com.shape[0] should be batch_sequence, actual com.shape[0] is ",
                com.sym_size(DIM_0), ", batch_sequence is ", batch_sequence, ".");
    TORCH_CHECK(com.sym_size(DIM_1) == hc,
                "The com.shape[1] should be hc, actual com.shape[1] is ",
                com.sym_size(DIM_1), ", hc is ", hc, ".");
    TORCH_CHECK(com.sym_size(DIM_2) == hc,
                "The com.shape[2] should be hc, actual com.shape[2] is ",
                com.sym_size(DIM_2), ", hc is ", hc, ".");
}

// 工具函数，检查 x=[b, s, d] 场景下 residual/post/com 的 shape
void check_hc_post_shape_3d(
    const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post, const at::Tensor& com)
{
    auto batch = x.sym_size(DIM_0);
    auto sequence = x.sym_size(DIM_1);
    auto d = x.sym_size(DIM_2);
    TORCH_CHECK(d == D_LIMIT_3584 || d == D_LIMIT_4096 || d == D_LIMIT_7168,
                "The d of x only support ", D_LIMIT_3584, ", ", D_LIMIT_4096, " or ", D_LIMIT_7168,
                ", actual ", d, ".");
    // check residual: [b, s, hc, d]
    TORCH_CHECK(residual.dim() == DIM_4,
                "Input tensor residual's dim num should be 4, actual ", residual.dim(), ".");
    auto hc = residual.sym_size(DIM_2);
    TORCH_CHECK(residual.sym_size(DIM_0) == batch,
                "The residual.shape[0] should be batch, actual residual.shape[0] is ",
                residual.sym_size(DIM_0), ", batch is ", batch, ".");
    TORCH_CHECK(residual.sym_size(DIM_1) == sequence,
                "The residual.shape[1] should be sequence, actual residual.shape[1] is ",
                residual.sym_size(DIM_1), ", sequence is ", sequence, ".");
    TORCH_CHECK(hc == HC_LIMIT, "The hc of residual only support ", HC_LIMIT, ", actual ", hc, ".");
    TORCH_CHECK(residual.sym_size(DIM_3) == d,
                "The residual.shape[3] should be d, actual residual.shape[3] is ",
                residual.sym_size(DIM_3), ", d is ", d, ".");
    // check post [b, s, hc]
    TORCH_CHECK(post.dim() == DIM_3, "Input tensor post's dim num should be 3, actual ", post.dim(), ".");
    TORCH_CHECK(post.sym_size(DIM_0) == batch,
                "The post.shape[0] should be batch, actual post.shape[0] is ",
                post.sym_size(DIM_0), ", batch is ", batch, ".");
    TORCH_CHECK(post.sym_size(DIM_1) == sequence,
                "The post.shape[1] should be sequence, actual post.shape[1] is ",
                post.sym_size(DIM_1), ", sequence is ", sequence, ".");
    TORCH_CHECK(post.sym_size(DIM_2) == hc,
                "The post.shape[2] should be hc, actual post.shape[2] is ",
                post.sym_size(DIM_2), ", hc is ", hc, ".");
    // check com: [b, s, hc, hc]
    TORCH_CHECK(com.dim() == DIM_4, "Input tensor com's dim num should be 4, actual ", com.dim(), ".");
    TORCH_CHECK(com.sym_size(DIM_0) == batch,
                "The com.shape[0] should be batch, actual com.shape[0] is ",
                com.sym_size(DIM_0), ", batch is ", batch, ".");
    TORCH_CHECK(com.sym_size(DIM_1) == sequence,
                "The com.shape[1] should be sequence, actual com.shape[1] is ",
                com.sym_size(DIM_1), ", sequence is ", sequence, ".");
    TORCH_CHECK(com.sym_size(DIM_2) == hc,
                "The com.shape[2] should be hc, actual com.shape[2] is ",
                com.sym_size(DIM_2), ", hc is ", hc, ".");
    TORCH_CHECK(com.sym_size(DIM_3) == hc,
                "The com.shape[3] should be hc, actual com.shape[3] is ",
                com.sym_size(DIM_3), ", hc is ", hc, ".");
}

// step1，工具函数，检查输入shape
void check_hc_post_shape_and_dtype(
    const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post, const at::Tensor& com)
{
    // check x shape: [bs, d] or [b, s, d]
    TORCH_CHECK(x.dim() == DIM_2 || x.dim() == DIM_3,
                "Input tensor x's dim num should be 2 or 3, actual ", x.dim(), ".");
    for (int64_t i = 0; i < x.dim(); i++) {
        TORCH_CHECK(x.sym_size(i) > DIM_0,
                    "Input tensor x's shape should be positive, but x.shape[", i, "] is :", x.sym_size(i), ".");
    }
    if (x.dim() == DIM_2) {
        check_hc_post_shape_2d(x, residual, post, com);
    } else {
        check_hc_post_shape_3d(x, residual, post, com);
    }
    // check dtype
    TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or FLOAT32.");
    TORCH_CHECK(residual.dtype() == x.dtype(), "x's dtype should be equal to residual's dtype.");
    TORCH_CHECK(post.dtype() == at::kFloat || post.dtype() == at::kHalf || post.dtype() == at::kBFloat16,
                "post should be FLOAT16, BFLOAT16, or FLOAT32.");
    TORCH_CHECK(com.dtype() == post.dtype(), "com's dtype should be equal to post's dtype.");
}

// step2, 为NPU设备实现前向接口
at::Tensor npu_hc_post_npu(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    check_hc_post_shape_and_dtype(x, residual, post, comb);
    // construct the output tensor
    at::Tensor out = construct_hc_post_output_tensor(residual);
    EXEC_NPU_CMD<HC_POST_NAME>(x, residual, post, comb, out);
    return out;
}

// step3, 为META设备实现前向接口
at::Tensor npu_hc_post_meta(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    check_hc_post_shape_and_dtype(x, residual, post, comb);
    // construct the output tensor
    at::Tensor outputs = construct_hc_post_output_tensor(residual);
    return outputs;
}

}  // namespace sglang::npu_kernel

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("hc_post", &sglang::npu_kernel::npu_hc_post_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("hc_post", &sglang::npu_kernel::npu_hc_post_meta);
}
