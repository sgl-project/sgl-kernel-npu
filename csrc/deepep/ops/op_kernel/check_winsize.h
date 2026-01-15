#ifndef CHECK_WINSIZE_H
#define CHECK_WINSIZE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"

__aicore__ inline void CheckWindowSize(uint64_t tilingWinSizeBytes, uint64_t realWinSizeBytes, AscendC::TPipe *tpipe_,
                                       GM_ADDR exceptionAddr)
{
    if (unlikely(realWinSizeBytes < tilingWinSizeBytes)) {
        constexpr uint64_t DATA_SIZE = 256;  // 定义数据大小为256字节
        AscendC::GlobalTensor<int32_t> exceptionGlobal;
        exceptionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(exceptionAddr), DATA_SIZE);
        AscendC::TBuf<AscendC::TPosition::VECCALC> exceptionBuf;
        tpipe_->InitBuffer(exceptionBuf, 1);  // 初始化一个缓冲区
        AscendC::LocalTensor<int32_t> exceptionLocal = exceptionBuf.Get<int32_t>();
        AscendC::DataCopy(exceptionLocal[1], exceptionGlobal, 1);  // 从全局地址复制数据到本地地址
    }
}
#endif  // CHECK_WINSIZE_H
