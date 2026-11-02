/*!
 * \file moe_distribute_dispatch_v2_ccu_tiling.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_CCU_TILING_H
#define MOE_DISTRIBUTE_DISPATCH_V2_CCU_TILING_H

#include "mc2_tiling_utils.h"
#include "moe_distribute_dispatch_tiling_helper.h"

namespace optiling {

ge::graphStatus MoeDistributeDispatchTilingImpl(gert::TilingContext *context);

}  // namespace optiling

#endif  // MOE_DISTRIBUTE_DISPATCH_TILING_ARCH35_H
