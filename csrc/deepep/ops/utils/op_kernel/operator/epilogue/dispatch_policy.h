#pragma once
#include "../catlass/act/epilogue/dispatch_policy.hpp"

namespace Act::Epilogue {

template <uint32_t UB_STAGES_, uint32_t EXEC_FLAG_>
struct EpilogueAtlasA2PerTokenDequantSwiglu {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t EXEC_FLAG = EXEC_FLAG_;
};

}  // namespace Act::Epilogue
