#ifndef FUSED_DEEP_MOE_BASE_H
#define FUSED_DEEP_MOE_BASE_H

#include "moe_distribute_base.h"

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType, bool IsNeedReduceScatter, uint32_t EXEC_FLAG
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType, IsNeedReduceScatter, EXEC_FLAG

#endif  // FUSED_DEEP_MOE_BASE_H
