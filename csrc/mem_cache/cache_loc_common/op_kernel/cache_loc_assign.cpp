// The computation is shared by the cache-location kernels.
#include "cache_loc_kernel.h"

extern "C" __global__ __aicore__ void sgl_cache_loc_assign(GM_ADDR req_indices, GM_ADDR token_pool,
                                                           GM_ADDR start_offset, GM_ADDR end_offset,
                                                           GM_ADDR out_cache_loc, GM_ADDR token_pool_ref,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AssignCacheTillingData);
    GET_TILING_DATA(data, tiling);
    if (data.key == 1) {
        CacheLocAssignKernel<int32_t> op;
        op.Init(req_indices, token_pool_ref, start_offset, end_offset, out_cache_loc, data);
        if ASCEND_IS_AIV {
            op.ProcessForTokenPoolAssign();
        }
    } else if (data.key == 2) {
        CacheLocAssignKernel<int64_t> op;
        op.Init(req_indices, token_pool_ref, start_offset, end_offset, out_cache_loc, data);
        if ASCEND_IS_AIV {
            op.ProcessForTokenPoolAssign();
        }
    }
}
