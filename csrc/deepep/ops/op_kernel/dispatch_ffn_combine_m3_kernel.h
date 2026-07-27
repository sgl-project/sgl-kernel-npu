/*
 * MiniMax-M3 DispatchFFNCombine kernel entry.
 *
 * Keep the M3 ABI and fixed W8A8 layout isolated from the generic operator so
 * decode-specific scheduling can evolve without changing DispatchFFNCombine.
 */
#ifndef DISPATCH_FFN_COMBINE_M3_KERNEL_H
#define DISPATCH_FFN_COMBINE_M3_KERNEL_H

#include "dispatch_ffn_combine.h"

namespace DispatchFFNCombineM3Impl {

template <typename WeightType, typename OutputType>
class DispatchFFNCombineM3 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight1, GM_ADDR weight2, GM_ADDR expertIds,
                                GM_ADDR scale1, GM_ADDR scale2, GM_ADDR probs, GM_ADDR out,
                                GM_ADDR expertTokenNums, GM_ADDR workspace, GM_ADDR tiling)
    {
        impl_.Init(x, weight1, weight2, expertIds, scale1, scale2, probs, out, expertTokenNums, workspace, tiling);
    }

    __aicore__ inline void Process()
    {
        impl_.Process();
    }

private:
    // M3 always uses int8 weights, BF16 output, and its fixed fused-SwiGLU layout.
    DispatchFFNCombineImpl::DispatchFFNCombine<int8_t, WeightType, OutputType, false, true> impl_;
};

}  // namespace DispatchFFNCombineM3Impl

#endif
