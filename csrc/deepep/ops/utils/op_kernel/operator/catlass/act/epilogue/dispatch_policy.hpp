#ifndef ACT_EPILOGUE_DISPATCH_POLICY_HPP
#define ACT_EPILOGUE_DISPATCH_POLICY_HPP

#include "../../act/arch/arch.hpp"

namespace Act::Epilogue {

// For AtlasA2, an element wise epilogue of the form D = C + X, where X is an
// additional source
struct EpilogueAtlasA2ElemWiseOneSource {
    using ArchTag = Arch::AtlasA2;
    // Number of operands. Including C, X, and D 3 operands
    static constexpr uint32_t OPERANDS_NUM = 3;
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

// For AtlasA2, per token dequant
template <uint32_t UB_STAGES_, uint32_t EXEC_FLAG_>
struct EpilogueAtlasA2PerTokenDequant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t EXEC_FLAG = EXEC_FLAG_;
};
}  // namespace Act::Epilogue

#endif  // ACT_EPILOGUE_DISPATCH_POLICY_HPP
