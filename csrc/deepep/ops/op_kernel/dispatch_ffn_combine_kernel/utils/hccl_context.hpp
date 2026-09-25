#ifndef DISPATCH_FFN_HCCL_CONTEXT_HPP
#define DISPATCH_FFN_HCCL_CONTEXT_HPP

// A5 uses an HcclCombinOpParam context, not A3's HcclOpResParam ABI.
// Use the same CANN accessors as the A5 FusedDeepMoe kernel.
#if defined(DEEPEP_A5_SYSTEM_MOE_BASE_USE_OP_KERNEL)
#include "op_kernel/moe_distribute_base.h"
#elif defined(DEEPEP_A5_SYSTEM_MOE_BASE_USE_INC_KERNEL)
#include "inc/kernel/moe_distribute_base.h"
#else
#include "moe_distribute_base.h"
#endif

namespace DispatchFfnHccl {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
using Context = Mc2Kernel::HcclOpParam;
__aicore__ inline uint32_t Rank(__gm__ Context *context)
{
    return Mc2Kernel::GetRankId(context);
}
__aicore__ inline uint32_t Size(__gm__ Context *context)
{
    return Mc2Kernel::GetRankDim(context);
}
__aicore__ inline GM_ADDR Window(__gm__ Context *context, int32_t rank)
{
    return Mc2Kernel::GetBaseWindAddrByRankId(context, rank, Rank(context));
}
__aicore__ inline uint64_t WindowSize(__gm__ Context *context)
{
    // The accessor advances each window past CANN's state area and rank offset.
    // Reserve the largest prefix so all peers use the same usable window size.
    uint64_t reserved = Mc2Kernel::A5_MTE_STATE_WIN_SIZE + (Size(context) - 1) * EP_RANK_OFFSET_STEP;
    uint64_t size = Mc2Kernel::GetWinSize(context);
    return size > reserved ? (size - reserved) / 512 * 512 : 0;
}
#else
using Context = HcclOpResParamCustom;
__aicore__ inline uint32_t Rank(__gm__ Context *context)
{
    return context->localUsrRankId;
}
__aicore__ inline uint32_t Size(__gm__ Context *context)
{
    return context->rankSize;
}
__aicore__ inline uint64_t WindowSize(__gm__ Context *context)
{
    return context->winSize;
}
__aicore__ inline GM_ADDR Window(__gm__ Context *context, int32_t rank)
{
    return (GM_ADDR)(rank == Rank(context)
                         ? context->localWindowsIn
                         : ((HcclRankRelationResV2Custom *)(context->remoteRes[rank].nextDevicePtr))->windowsIn);
}
#endif
}  // namespace DispatchFfnHccl
#endif
