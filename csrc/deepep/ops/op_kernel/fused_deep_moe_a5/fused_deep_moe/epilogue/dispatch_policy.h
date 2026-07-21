#pragma once
#include "catlass_a5/epilogue/dispatch_policy.hpp"

namespace Catlass::Epilogue {

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5SiluHalf {
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
