#ifndef DISPATH_POLICY_CUSTOM_HPP
#define DISPATH_POLICY_CUSTOM_HPP

namespace Catlass {
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using DispatchFfnArch = Arch::Ascend950;
#else
using DispatchFfnArch = Arch::AtlasA2;
#endif
}  // namespace Catlass

namespace Catlass::Gemm {
template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2PreloadFixpipeQuant : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadDispatchFfnPreloadAsyncFixpipe
    : public MmadAtlasA2PreloadAsync<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                     ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {
    // Select A5's GM/L1/L0 copies, INT8 MMAD, and per-channel Fixpipe.
    using ArchTag = DispatchFfnArch;
};
}  // namespace Catlass::Gemm

namespace Catlass::Epilogue {

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2UnQuant {
    using ArchTag = DispatchFfnArch;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueDispatchFfnPerTokenDequantSwigluQuant {
    using ArchTag = DispatchFfnArch;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueDispatchFfnPerTokenDequant {
    using ArchTag = DispatchFfnArch;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};
}  // namespace Catlass::Epilogue
#endif  // DISPATH_POLICY_CUSTOM_HPP
