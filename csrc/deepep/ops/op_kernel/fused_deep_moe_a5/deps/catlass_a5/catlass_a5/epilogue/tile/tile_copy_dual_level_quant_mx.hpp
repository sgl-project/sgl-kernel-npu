/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_COPY_DUAL_LEVEL_QUANT_MX_HPP
#define CATLASS_EPILOGUE_TILE_TILE_COPY_DUAL_LEVEL_QUANT_MX_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"
#include "catlass_a5/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass_a5/epilogue/tile/copy_ub_to_gm.hpp"

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

namespace Catlass::Epilogue::Tile {

/**
 * @brief TileCopy for the Dual-Level Quantization + MX Format block.
 *
 * Block 不区分 A/B,只看 "一个矩阵的一个 tile",所以这里只需要 4 组拷贝类型:
 *   - Input  (GM -> UB)
 *   - Output (UB -> GM)   FP4 packed bytes
 *   - Scale1 (UB -> GM)   float
 *   - Scale2 (UB -> GM)   fp8_e8m0
 *
 * A 与 B 的差异由 kernel 在传入 GM tensor + layout 时解决,本结构不感知。
 */
template <
    class ArchTag,
    class InputType,
    class OutputType,
    class Scale1Type,
    class Scale2Type
>
struct TileCopyDualLevelQuantMx {
    using ElementInput  = typename InputType::Element;
    using ElementOutput = typename OutputType::Element;
    using ElementScale1 = typename Scale1Type::Element;
    using ElementScale2 = typename Scale2Type::Element;

    using CopyGmToUbInput  = CopyGm2Ub<ArchTag, InputType>;
    using CopyUbToGmOutput = CopyUb2Gm<ArchTag, OutputType>;
    using CopyUbToGmScale1 = CopyUb2Gm<ArchTag, Scale1Type>;
    using CopyUbToGmScale2 = CopyUb2Gm<ArchTag, Scale2Type>;
};

} // namespace Catlass::Epilogue::Tile

#endif // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

#endif // CATLASS_EPILOGUE_TILE_TILE_COPY_DUAL_LEVEL_QUANT_MX_HPP
