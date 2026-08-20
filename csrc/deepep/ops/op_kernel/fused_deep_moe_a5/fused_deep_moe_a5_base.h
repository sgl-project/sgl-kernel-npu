#ifndef FUSED_DEEP_MOE_BASE_H
#define FUSED_DEEP_MOE_BASE_H

#if defined(DEEPEP_A5_SYSTEM_MOE_BASE_USE_OP_KERNEL)
#include "op_kernel/moe_distribute_base.h"
#elif defined(DEEPEP_A5_SYSTEM_MOE_BASE_USE_INC_KERNEL)
#include "inc/kernel/moe_distribute_base.h"
#else
#error "A5 fused requires a system moe_distribute_base.h include mode"
#endif

#define TemplateMC2TypeClass                                                                                     \
    typename ExpandXType, typename WeightType, bool WEIGHT_NZ, typename ExpandIdxType, bool IsNeedReduceScatter, \
        uint32_t EXEC_FLAG
#define TemplateMC2TypeFunc ExpandXType, WeightType, WEIGHT_NZ, ExpandIdxType, IsNeedReduceScatter, EXEC_FLAG

#define TemplateDispatchTypeClass                                                                          \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
        bool IsNeedAllgater, uint32_t EXEC_FLAG
#define TemplateDispatchTypeFunc \
    XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgater, EXEC_FLAG

constexpr int64_t SLEEP_CYCLE = 50;

__aicore__ inline void SPIN_WAIT_CYCLES()
{
    AscendC::Nop<SLEEP_CYCLE>();
}

#endif  // FUSED_DEEP_MOE_BASE_H
