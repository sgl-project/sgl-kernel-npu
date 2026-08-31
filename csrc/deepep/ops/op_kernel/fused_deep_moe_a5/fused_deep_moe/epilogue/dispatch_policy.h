#pragma once
#include "catlass/epilogue/dispatch_policy.hpp"

namespace Catlass::Epilogue {

constexpr uint32_t ACTIVATION_SILU = 0U;
constexpr uint32_t ACTIVATION_SITU = 1U;

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5SiluHalf {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// SiTU (Kimi K3 soft-saturation gated) activation; the BlockEpilogue
// specialization is hosted in epilogue/block/block_epilogue_silu_half.h.
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5SituHalf {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5SiluSituHalf {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t EXEC_FLAG_>
struct EpilogueAtlasA5CastCombine {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t UB_STAGES = 1;
    static constexpr uint32_t EXEC_FLAG = EXEC_FLAG_;
};

}  // namespace Catlass::Epilogue
