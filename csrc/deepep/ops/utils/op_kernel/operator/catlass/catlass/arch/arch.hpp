/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_ARCH_ARCH_HPP
#define CATLASS_ARCH_ARCH_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

template <AscendC::TPosition POS>
using PositionType = std::integral_constant<AscendC::TPosition, POS>;

using PositionGM = PositionType<AscendC::TPosition::GM>;
using PositionL1 = PositionType<AscendC::TPosition::A1>;
using PositionL0A = PositionType<AscendC::TPosition::A2>;
using PositionL0B = PositionType<AscendC::TPosition::B2>;
using PositionL0C = PositionType<AscendC::TPosition::CO1>;
using PositionUB = PositionType<AscendC::TPosition::VECCALC>;

} // namespace Catlass::Arch

#endif // CATLASS_ARCH_ARCH_HPP