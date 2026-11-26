#include "cache_planner_impl.h"

extern "C" __global__ __aicore__ void plan_sparse_kv_cache(
    GM_ADDR sourceToSlot, GM_ADDR slotToSource, GM_ADDR reqIds, GM_ADDR topkInput, GM_ADDR topkSrcOut,
    GM_ADDR topkDstOut, GM_ADDR topkMissCounts, GM_ADDR missSrcOut, GM_ADDR missDstOut, GM_ADDR missCounts,
    uint32_t realReqCount, uint32_t maxContextLen, uint32_t sourceStride, uint32_t cacheCapacity, uint32_t batchSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelPlanSparseKvCache kernel;
    AscendC::TPipe pipe;
    kernel.Init(sourceToSlot, slotToSource, reqIds, topkInput, topkSrcOut, topkDstOut, topkMissCounts, missSrcOut,
                missDstOut, missCounts, realReqCount, maxContextLen, sourceStride, cacheCapacity, batchSize, &pipe);
    kernel.Process();
}
