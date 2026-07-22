/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_DISPATCH_POLICY_HPP
#define CATLASS_EPILOGUE_DISPATCH_POLICY_HPP

#include "catlass_a5/catlass.hpp"
#include "catlass_a5/arch/arch.hpp"

namespace Catlass::Epilogue {

// For AtlasA2, an element wise epilogue of the form D = C + X, where X is an additional source
struct EpilogueAtlasA2ElemWiseOneSource {
    using ArchTag = Arch::AtlasA2;
    // Number of operands. Including C, X, and D 3 operands
    static constexpr uint32_t OPERANDS_NUM = 3;
};

struct EpilogueAtlasA2ElemWiseNoSource {
    using ArchTag = Arch::AtlasA2;
    // Number of operands. Including C, D 2 operands
    static constexpr uint32_t OPERANDS_NUM = 2;
};

// For AtlasA2, FA Softmax
struct EpilogueAtlasA2FASoftmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, FA RescaleO
struct EpilogueAtlasA2FARescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLA Softmax
struct EpilogueAtlasA2MLASoftmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, FA Infer online Softmax no mask
struct EpilogueAtlasA2OnlineSoftmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, FA Infer RescaleO no split row
struct EpilogueAtlasA2RescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For Ascend950, FA Infer online Softmax
template <bool ATTENTION_MASK_FLAG_ = false, bool ENABLE_P_SCALE_ = false>
struct EpilogueAscend950FASoftmax {
    using ArchTag = Arch::Ascend950;
    static constexpr bool ATTENTION_MASK_FLAG = ATTENTION_MASK_FLAG_;
    static constexpr bool ENABLE_P_SCALE = ENABLE_P_SCALE_;
};

// For Ascend950, FA Infer RescaleO
struct EpilogueAscend950FARescaleO {
    using ArchTag = Arch::Ascend950;
};

// For AtlasA2, MLA RescaleO
struct EpilogueAtlasA2MLARescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLA FD RescaleO
template <uint32_t COMPUTE_ELE_NUM_>
struct EpilogueAtlasA2MLAFDRescaleO {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t KV_SPLIT_MAX = 64;
    static constexpr uint32_t HEADS_PROCESS_MAX = 16;
    static constexpr uint32_t COMPUTE_ELE_NUM = COMPUTE_ELE_NUM_;
};

// For AtlasA2, MLA TP1 Softmax
struct EpilogueAtlasA2MLATP1Softmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLA TP1 RescaleO
struct EpilogueAtlasA2MLATP1RescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, AMLA TP1 Softmax
struct EpilogueAtlasA2AMLATP1Softmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, AMLA TP1 RescaleO
struct EpilogueAtlasA2AMLATP1RescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, per token dequant
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// For AtlasA2, per token dequant tla version
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantTla {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// For Ascend950, per token dequant
template <uint32_t UB_STAGES_>
struct EpilogueAscend950PerTokenDequantTla {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// For Ascend950, perGroup + perBlock dequant
struct BlockEpiloguePertile {
    using ArchTag = Arch::Ascend950;
};

// For AtlasA2, W4A4 epilogue process
struct EpilogueAtlasA2W4A4PerTokenPerChannelDequant {
    using ArchTag = Arch::AtlasA2;
};
////////////////////////////
/// new add
// For AtlasA2, GEMM
struct EpilogueAtlasA2Gemm {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, GEMV
struct EpilogueAtlasA2Gemv {
    using ArchTag = Arch::AtlasA2;
};
///////////////////////////
// For Ascend950, fixpipe-opti
template <bool SPLIT_M_>
struct EpilogueAscend950Fixpipe {
    using ArchTag = Arch::Ascend950;
    static constexpr bool SPLIT_M = SPLIT_M_;
};
// For Ascend950, full dequant
struct BlockEpilogueDequant {
    using ArchTag = Arch::Ascend950;
};

// For Ascend950, dual-level quantization + MX format
template <uint32_t UB_STAGES_>
struct EpilogueAscend950DualLevelQuantMx {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t OPERANDS_NUM = 8;
};

// For Ascend950, per block quant tla
template <uint32_t UB_STAGES_>
struct EpilogueAscend950PerBlockQuantTla {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// For Epilogue with Visitor(EVG)
// 模板参数 USE_UB_WORKSPACE_ 控制是否使用 UB 作为 MMAD 的 workspace
// - false: 使用 GM workspace
// - true:  使用 UB workspace
template <bool USE_UB_WORKSPACE_ = false>
struct EpilogueVisitor {
    static constexpr bool USE_UB_WORKSPACE = USE_UB_WORKSPACE_;
};

// For Ascend950, SwiGLU activation + MX quant output
struct BlockEpilogueSwigluMxQuant {
    using ArchTag = Arch::Ascend950;
};

// For Ascend950, finalize routing epilogue (AIV post-processing)
// template <uint32_t UB_STAGES_>
template <uint32_t UB_STAGES_>
struct EpilogueAscend950FinalizeRouting {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// For Ascend950, flash_attention_chunk_prefill softmax
struct EpilogueFAOnlineSoftmax {
    using ArchTag = Arch::Ascend950;
};

// For Ascend950, flash_attention_chunk_prefill rescaleO
struct EpilogueFARescaleO {
    using ArchTag = Arch::Ascend950;
};
}  // namespace Catlass::Epilogue

#endif  // CATLASS_EPILOGUE_DISPATCH_POLICY_HPP
