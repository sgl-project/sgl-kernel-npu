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
        case host_utils::DataType::DT_FLOAT:
            return matmul_tiling::DataType::DT_FLOAT;
        case host_utils::DataType::DT_FLOAT16:
            return matmul_tiling::DataType::DT_FLOAT16;
    }

    return matmul_tiling::DataType::DT_FLOAT16;
}

at::Tensor GenerateTiling(uint32_t &block_dim, uint32_t &workspace_size, uint32_t batch_size, uint32_t inner_size,
                          uint32_t output_size, const host_utils::DataType type)
{
    auto ascendc_platform = *platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    uint32_t aic_num = ascendc_platform.GetCoreNumAic();
    workspace_size = ascendc_platform.GetLibApiWorkSpaceSize();

    auto tilingBuffer = at::empty({sizeof(SGEMMCTilingData)}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    SGEMMCTilingData *tiling_data = reinterpret_cast<SGEMMCTilingData *>(tilingBuffer.data_ptr());

    matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendc_platform);

    const matmul_tiling::DataType data_type = ConvertToMatMulTypes(type);

    cubeTiling.EnableBias(false);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::VECTOR, data_type, false);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, data_type, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, data_type);
    cubeTiling.EnableMultiCoreSplitK(false);
    cubeTiling.SetDim(aic_num);

    cubeTiling.SetOrgShape(1, inner_size, output_size);
    cubeTiling.SetShape(1, inner_size, output_size);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    if (cubeTiling.GetTiling(tiling_data->cubeTiling) == -1) {
        TORCH_CHECK(false, "Generate tiling failed.");
        return {};
    }

    tiling_data->batch = batch_size;
    tiling_data->dataType = (type == host_utils::DataType::DT_BFLOAT16);

    block_dim = batch_size * tiling_data->cubeTiling.usedCoreNum;

    return tilingBuffer;
}

}  // namespace npu_kernel
}  // namespace sglang
