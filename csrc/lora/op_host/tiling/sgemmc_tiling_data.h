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

#ifndef SGEMMC_TILING_DATA_H
#define SGEMMC_TILING_DATA_H

#include <cstdint>

namespace AscendC {
namespace tiling {

struct TCubeTiling;

}  // namespace tiling
}  // namespace AscendC

namespace sglang {
namespace npu_kernel {

#pragma pack(push, 1)
struct SGEMMCTilingData {
    uint32_t dataType;
    uint32_t batch;
    uint32_t hidden;
    uint32_t k;
    uint32_t slices;
    AscendC::tiling::TCubeTiling cubeTiling;
};
#pragma pack(pop)

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGEMMC_TILING_DATA_H
