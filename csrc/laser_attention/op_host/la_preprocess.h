/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef LA_PREPROCESS_MINDIE_SD_IMPL_H
#define LA_PREPROCESS_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

namespace sglang {
namespace npu_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor>la_preprocess(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    int64_t align_len = 256);

}
}

#endif // LA_PREPROCESS_MINDIE_SD_IMPL_H
