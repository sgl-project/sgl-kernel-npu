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
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2.h"
#include "moe_distribute_combine_v2_layered_custom.h"
#include "moe_distribute_combine_v2_single.h"
#include <cstdio>

using namespace AscendC;
using namespace MoeDistributeCombineA2Impl;

/*
 2000  A2
 3000  A2+layered
  100  quant
*/
extern "C" __global__ __aicore__ void moe_distribute_combine_v2(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, GM_ADDR scales,
    GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale, GM_ADDR groupList,
    GM_ADDR expandScales, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 10000", MoeDistributeCombineV2TilingData);
    TPipe pipe;

#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(2000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
        auto tiling = (__gm__ MoeDistributeCombineV2TilingData *)tilingGM;
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
        MoeDistributeCombineV2<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, xActiveMask, oriX, constExpertAlpha1,
                constExpertAlpha2, constExpertV, XOut, workspaceGM, &pipe, &tilingData, mc2InitTiling, mc2CcTiling);
        op.Process();
    } else if (TILING_KEY_IS(3000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
        auto tiling = (__gm__ MoeDistributeCombineV2TilingData *)tilingGM;
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineV2Layered<DTYPE_EXPAND_X, int32_t, DTYPE_EXPAND_X> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, XOut, workspaceGM, &pipe,
                    &tilingData, mc2InitTiling, mc2CcTiling, contextGM0);
            op.Process();
        } else {
            assert(false, "The driver version is too low and does not support layered mode.\n");
        }
    } else if (TILING_KEY_IS(3100)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
        auto tiling = (__gm__ MoeDistributeCombineV2TilingData *)tilingGM;
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineV2Layered<DTYPE_EXPAND_X, int32_t, int8_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, XOut, workspaceGM, &pipe,
                    &tilingData, mc2InitTiling, mc2CcTiling, contextGM0);
            op.Process();
        } else {
            assert(false, "The driver version is too low. It should not be lower than 25.0.rc1.1.\n");
        }
    } else if (TILING_KEY_IS(5000)) {  // single server
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);

        MoeDistributeCombineV2Single<DTYPE_EXPAND_X, DTYPE_X, int32_t, false, false, false> op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX,
                XOut, workspaceGM, &pipe, tilingGM);
        op.Process();
    }
#endif
}
