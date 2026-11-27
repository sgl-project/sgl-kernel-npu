// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hyper_conn_round.h"
#include "hyper_conn_tiling_data.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

using sglang::npu_kernel::hyper_conn::AlignUp;
using sglang::npu_kernel::hyper_conn::BF16_SIZE;
using sglang::npu_kernel::hyper_conn::F32_SIZE;
using sglang::npu_kernel::hyper_conn::GATE_LANES;
using sglang::npu_kernel::hyper_conn::MAX_BRANCHES;
using sglang::npu_kernel::hyper_conn::SigmoidF32;
using sglang::npu_kernel::hyper_conn::UB_BLOCK;

extern "C" __global__ __aicore__ void sgl_hyper_conn_branch_combine(GM_ADDR block, GM_ADDR residual, GM_ADDR logits,
                                                                    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglHyperConnBranchCombineTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    static_assert(MAX_BRANCHES <= GATE_LANES, "per-row gate buffer holds GATE_LANES lanes");
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t rows = tilingData.rows;
    const uint32_t hc = tilingData.hc;
    const uint32_t width = tilingData.width;
    const uint32_t tile = tilingData.tile;
    const uint32_t logitsStride = tilingData.logitsStride;
    const float invHc = tilingData.invHc;

    const uint32_t cores = AscendC::GetBlockNum();
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t span = hc * tile;
    const uint32_t padBf16 = AlignUp(span * BF16_SIZE, UB_BLOCK);
    const uint32_t padF32 = AlignUp(span * F32_SIZE, UB_BLOCK);
    const uint32_t tileBf16 = AlignUp(tile * BF16_SIZE, UB_BLOCK);
    const uint32_t tileF32 = AlignUp(tile * F32_SIZE, UB_BLOCK);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> blockQue;
    TQue<TPosition::VECIN, 2> residQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<TPosition::VECCALC> blockBuf, residBuf, injBuf;
    TBuf<TPosition::VECCALC> gateF32, gateTmp, gateTwoBuf, gateInBuf;
    pipe.InitBuffer(blockQue, 2, tileBf16);
    pipe.InitBuffer(residQue, 2, padBf16);
    pipe.InitBuffer(outQue, 2, padBf16);
    pipe.InitBuffer(blockBuf, tileF32);
    pipe.InitBuffer(residBuf, padF32);
    pipe.InitBuffer(injBuf, padF32);
    pipe.InitBuffer(gateF32, 2 * UB_BLOCK);
    pipe.InitBuffer(gateTmp, 2 * UB_BLOCK);
    pipe.InitBuffer(gateTwoBuf, 2 * UB_BLOCK);
    pipe.InitBuffer(gateInBuf, UB_BLOCK);

    GlobalTensor<bfloat16_t> gBlock, gResid, gInj, gOut;
    gBlock.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(block));
    gResid.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(residual));
    gInj.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(logits));
    gOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(out));

    LocalTensor<float> blockF = blockBuf.Get<float>();
    LocalTensor<float> residF = residBuf.Get<float>();
    LocalTensor<float> inject = injBuf.Get<float>();
    LocalTensor<float> gateVal = gateF32.Get<float>();
    LocalTensor<float> gateScratch = gateTmp.Get<float>();
    LocalTensor<float> gateTwo = gateTwoBuf.Get<float>();
    LocalTensor<bfloat16_t> gateBf = gateInBuf.Get<bfloat16_t>();
    AscendC::Duplicate(gateTwo, 2.0f, GATE_LANES);

    const AscendC::DataCopyExtParams branchCopy{static_cast<uint16_t>(hc), tile * BF16_SIZE, (width - tile) * BF16_SIZE,
                                                0, 0};
    const AscendC::DataCopyExtParams storeCopy{static_cast<uint16_t>(hc), tile * BF16_SIZE, 0,
                                               (width - tile) * BF16_SIZE, 0};
    const AscendC::DataCopyExtParams tileCopy{1, tile * BF16_SIZE, 0, 0, 0};
    const AscendC::DataCopyExtParams gateCopy{1, hc * BF16_SIZE, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};

    const uint32_t tilesPerRow = width / tile;
    const uint32_t tasks = rows * tilesPerRow;
    uint32_t cachedRow = 0xFFFFFFFFu;
    float gates[MAX_BRANCHES] = {0};

    for (uint32_t task = core; task < tasks; task += cores) {
        const uint32_t row = task / tilesPerRow;
        const uint32_t col = (task % tilesPerRow) * tile;
        const uint64_t base = static_cast<uint64_t>(row) * hc * width + col;

        if (row != cachedRow) {
            cachedRow = row;
            AscendC::DataCopyPad(gateBf, gInj[static_cast<uint64_t>(row) * logitsStride], gateCopy, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::Cast(gateVal, gateBf, RoundMode::CAST_NONE, GATE_LANES);
            AscendC::Muls(gateVal, gateVal, invHc, GATE_LANES);
            SigmoidF32(gateVal, gateVal, gateScratch, gateTwo, GATE_LANES);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            for (uint32_t branch = 0; branch < hc; ++branch) {
                gates[branch] = gateVal.GetValue(branch);
            }
        }

        LocalTensor<bfloat16_t> blockTile = blockQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(blockTile, gBlock[static_cast<uint64_t>(row) * width + col], tileCopy, pad);
        blockQue.EnQue(blockTile);
        LocalTensor<bfloat16_t> residualTile = residQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(residualTile, gResid[base], branchCopy, pad);
        residQue.EnQue(residualTile);

        blockTile = blockQue.DeQue<bfloat16_t>();
        AscendC::Cast(blockF, blockTile, RoundMode::CAST_NONE, tile);
        blockQue.FreeTensor(blockTile);
        residualTile = residQue.DeQue<bfloat16_t>();
        AscendC::Cast(residF, residualTile, RoundMode::CAST_NONE, span);
        residQue.FreeTensor(residualTile);

        for (uint32_t branch = 0; branch < hc; ++branch) {
            AscendC::Muls(inject[branch * tile], blockF, gates[branch], tile);
        }
        AscendC::Add(inject, inject, residF, span);

        LocalTensor<bfloat16_t> outTile = outQue.AllocTensor<bfloat16_t>();
        AscendC::Cast(outTile, inject, RoundMode::CAST_RINT, span);
        outQue.EnQue(outTile);
        outTile = outQue.DeQue<bfloat16_t>();
        AscendC::DataCopyPad(gOut[base], outTile, storeCopy);
        outQue.FreeTensor(outTile);
    }
}
