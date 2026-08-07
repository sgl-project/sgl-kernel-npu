/*!
 * \file moe_distribute_combine_v2_ccu_tiling.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_CCU_TILING_H
#define MOE_DISTRIBUTE_COMBINE_V2_CCU_TILING_H

#include "mc2_tiling_utils.h"

namespace optiling {

ge::graphStatus MoeDistributeCombineTilingImpl(gert::TilingContext *context);

}  // namespace optiling

#endif  // MOE_DISTRIBUTE_COMBINE_TILING_A5_H
