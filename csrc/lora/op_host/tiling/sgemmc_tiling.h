/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
 *
 */

#ifndef SGEMMC_TILING_H
#define SGEMMC_TILING_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

#include "torch_helper.h"
#include "common_tiling.h"
#include "sgemmc_tiling_data.h"

namespace sglang {
namespace npu_kernel {

at::Tensor GenerateTiling(uint32_t &blockDim, uint32_t &workspace, uint32_t batch, uint32_t hidden_size, uint32_t k,
                          const host_utils::DataType type);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGEMMC_TILING_H
