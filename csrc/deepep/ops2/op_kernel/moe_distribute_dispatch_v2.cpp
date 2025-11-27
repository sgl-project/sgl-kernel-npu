/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_dispatch_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_v2.h"
#include "moe_distribute_dispatch_v2_layered_custom.h"
#include "moe_distribute_dispatch_v2_single.h"
#include <cstdio>

using namespace AscendC;
using namespace MoeDistributeDispatchA2Impl;

/*
 2000000000  A2
  100000000  layered
       1000  init
         10  isScales
          2  quantMode
*/
extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
                                                                 GM_ADDR xActiveMask, GM_ADDR elasticInfo,
                                                                 GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                                                 GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
                                                                 GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 2000000000", MoeDistributeDispatchV2TilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(2000001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2100001000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchV2Layered<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else {
            assert(false, "The driver version is too low and does not support layered mode.\n");
        }
    } else if (TILING_KEY_IS(2000011000)) {  // single server
        printf("====enter dispatch single...\n");
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2Single<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(2000001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2000001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(2100001002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchV2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
            op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else {
            assert(false, "The driver version is too low and does not support layered mode.\n");
        }
    } else if (TILING_KEY_IS(2100001012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchV2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, true> op;
            op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, workspaceGM, &pipe, tilingGM, contextGM0);
            op.Process();
        } else {
            assert(false, "The driver version is too low and does not support layered mode.\n");
        }
    } else if (TILING_KEY_IS(2000011002)) {  // single server + quant
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2Single<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    }
#endif
}
