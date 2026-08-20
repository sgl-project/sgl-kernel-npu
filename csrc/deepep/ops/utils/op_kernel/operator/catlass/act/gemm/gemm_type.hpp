#ifndef ACT_GEMM_GEMM_TYPE_HPP
#define ACT_GEMM_GEMM_TYPE_HPP

namespace Act::Gemm {

////////////////////////////////////////////////////////////////////

template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition POSITION = POSITION_;
};

}  // namespace Act::Gemm

#endif  // ACT_GEMM_GEMM_TYPE_HPP
