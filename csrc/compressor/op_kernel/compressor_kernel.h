/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/*!
 * \file compressor_kernel.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL
#define COMPRESSOR_KERNEL

#include "compressor_comm.h"
#include "compressor_template_tiling_key.h"
#include "compressor_kernel_perf.h"
#include "compressor_tiling_data.h"
#include "compressor_tools.h"

using namespace AscendC;

namespace Compressor {

template <typename COMP>
class CompressorKernel
{
public:
    __aicore__ inline CompressorKernel(TPipe *pipe, const optiling::CompressorTilingData *__restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                __gm__ uint8_t *stateCache, __gm__ uint8_t *ape, __gm__ uint8_t *normWeight,
                                __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                __gm__ uint8_t *cmpKvOut, __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

    // ==============================TilingData&TPipe==============================
    TPipe *pipe_;
    const optiling::CompressorTilingData *__restrict tilingData_;
};

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                                    __gm__ uint8_t *stateCache, __gm__ uint8_t *ape,
                                                    __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin,
                                                    __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                                    __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed,
                                                    __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut,
                                                    __gm__ uint8_t *workspace)
{}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Process()
{}

}  // namespace Compressor

#endif  // COMPRESSOR_KERNEL
