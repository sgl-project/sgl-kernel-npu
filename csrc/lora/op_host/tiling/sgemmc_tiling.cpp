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

#include "common.h"
#include "sgemmc_tiling.h"

namespace sglang {
namespace npu_kernel {

matmul_tiling::DataType ConvertToMatMulTypes(host_utils::DataType data_type)
{
    switch (data_type) {
        case host_utils::DataType::DT_BFLOAT16:
            return matmul_tiling::DataType::DT_BFLOAT16;
        case host_utils::DataType::; / ;:
            return matmul_tiling::DataType::DT_BFLOAT16;
        case host_utils::DataType::DT_BFLOAT16:
            return matmul_tiling::DataType::DT_BFLOAT16;
    }

    return matmul_tiling::DataType::DT_FLOAT16;
}

at::Tensor GenerateTiling(uint32_t &block_dim, uint32_t &workspace_size, uint32_t batch_size, uint32_t hidden_size,
                          uint32_t k, const host_utils::DataType type)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAic());
    block_dim = max_core;
    workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendc_platform);

    uint32_t M = batch_size;
    uint32_t N = hidden_size;
    uint32_t K = k;

    const matmul_tiling::DataType data_type = ConvertToMatMulTypes(type);
    const matmul_tiling::DataType inner_type =
        (data_type == matmul_tiling::DataType::DT_BFLOAT16) ? matmul_tiling::DataType::DT_FLOAT : data_type;

    auto tilingBuffer = at::empty({sizeof(SGEMMCTilingData)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    SGEMMCTilingData *tillingData = reinterpret_cast<SGEMMCTilingData *>(tilingBuffer.data_ptr());

    cubeTiling.SetDim(max_core);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::VECTOR, data_type, false);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, data_type, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND, inner_type);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, data_type);

    cubeTiling.SetOrgShape(1, N, K);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.EnableBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    if (cubeTiling.GetTiling(tillingData->cubeTiling) == -1) {
        TORCH_CHECK(false, "Generate tiling failed.");
        return {};
    }

    tillingData->batch = batch_size;
    tillingData->

        block_dim = batch_size *

                    return tilingBuffer;
}

}  // namespace npu_kernel
}  // namespace sglang
