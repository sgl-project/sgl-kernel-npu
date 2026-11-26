#include "cache_planner_tiling.h"
#include "cache_planner_impl.h"

extern "C" __global__ __aicore__ void sgl_plan_sparse_kv_cache(
    GM_ADDR sourceToSlot, GM_ADDR slotToSource, GM_ADDR reqIds, GM_ADDR topkInput, GM_ADDR topkSrc, GM_ADDR topkDst,
    GM_ADDR topkMiss, GM_ADDR missSrc, GM_ADDR missDst, GM_ADDR missCount, GM_ADDR sourceToSlotOut,
    GM_ADDR slotToSourceOut, GM_ADDR topkSrcOut, GM_ADDR topkDstOut, GM_ADDR topkMissOut, GM_ADDR missSrcOut,
    GM_ADDR missDstOut, GM_ADDR missCountOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::CachePlannerTilingData);
    GET_TILING_DATA(data, tiling);
    KernelPlanSparseKvCache kernel;
    AscendC::TPipe pipe;
    kernel.Init(sourceToSlotOut, slotToSourceOut, reqIds, topkInput, topkSrcOut, topkDstOut, topkMissOut, missSrcOut,
                missDstOut, missCountOut, data.realReqCount, data.maxContextLen, data.sourceStride, data.cacheCapacity,
                data.batchSize, &pipe);
    kernel.Process();
}
