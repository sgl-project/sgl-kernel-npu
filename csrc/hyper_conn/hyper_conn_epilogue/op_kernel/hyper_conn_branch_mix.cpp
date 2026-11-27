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
using sglang::npu_kernel::hyper_conn::SigmoidF32;
using sglang::npu_kernel::hyper_conn::UB_BLOCK;

extern "C" __global__ __aicore__ void sgl_hyper_conn_branch_mix(GM_ADDR normed, GM_ADDR logits, GM_ADDR mixed,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglHyperConnBranchMixTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t rows = tilingData.rows;
    const uint32_t hc = tilingData.hc;
    const uint32_t width = tilingData.width;
    const uint32_t tile = tilingData.tile;
    const float invHc = tilingData.invHc;

    const uint32_t cores = AscendC::GetBlockNum();
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t span = hc * tile;
    const uint32_t padBf16 = AlignUp(span * BF16_SIZE, UB_BLOCK);
    const uint32_t padF32 = AlignUp(span * F32_SIZE, UB_BLOCK);
    const uint32_t outBf16 = AlignUp(tile * BF16_SIZE, UB_BLOCK);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> gateQue;
    TQue<TPosition::VECIN, 2> normQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<TPosition::VECCALC> gateBuf, normBuf, tmpBuf, oneBuf;
    pipe.InitBuffer(gateQue, 2, padBf16);
    pipe.InitBuffer(normQue, 2, padBf16);
    pipe.InitBuffer(outQue, 2, outBf16);
    pipe.InitBuffer(gateBuf, padF32);
    pipe.InitBuffer(normBuf, padF32);
    pipe.InitBuffer(tmpBuf, padF32);
    pipe.InitBuffer(oneBuf, padF32);

    GlobalTensor<bfloat16_t> gNorm, gGate, gMixed;
    gNorm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(normed));
    gGate.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(logits));
    gMixed.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(mixed));

    LocalTensor<float> gate = gateBuf.Get<float>();
    LocalTensor<float> norm = normBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> ones = oneBuf.Get<float>();
    AscendC::Duplicate(ones, 1.0f, span);

    const AscendC::DataCopyExtParams branchCopy{static_cast<uint16_t>(hc), tile * BF16_SIZE, (width - tile) * BF16_SIZE,
                                                0, 0};
    const AscendC::DataCopyExtParams outCopy{1, tile * BF16_SIZE, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};

    const uint32_t tilesPerRow = width / tile;
    const uint32_t tasks = rows * tilesPerRow;
    for (uint32_t task = core; task < tasks; task += cores) {
        const uint32_t row = task / tilesPerRow;
        const uint32_t col = (task % tilesPerRow) * tile;
        const uint64_t base = static_cast<uint64_t>(row) * hc * width + col;

        LocalTensor<bfloat16_t> gateIn = gateQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(gateIn, gGate[base], branchCopy, pad);
        gateQue.EnQue(gateIn);
        LocalTensor<bfloat16_t> normIn = normQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(normIn, gNorm[base], branchCopy, pad);
        normQue.EnQue(normIn);

        gateIn = gateQue.DeQue<bfloat16_t>();
        AscendC::Cast(gate, gateIn, RoundMode::CAST_NONE, span);
        gateQue.FreeTensor(gateIn);
        normIn = normQue.DeQue<bfloat16_t>();
        AscendC::Cast(norm, normIn, RoundMode::CAST_NONE, span);
        normQue.FreeTensor(normIn);

        SigmoidF32(gate, gate, tmp, ones, span);
        AscendC::Mul(gate, gate, norm, span);
        for (uint32_t branch = 1; branch < hc; ++branch) {
            AscendC::Add(gate, gate, gate[branch * tile], tile);
        }
        AscendC::Muls(gate, gate, invHc, tile);

        LocalTensor<bfloat16_t> out = outQue.AllocTensor<bfloat16_t>();
        AscendC::Cast(out, gate, RoundMode::CAST_RINT, tile);
        outQue.EnQue(out);
        out = outQue.DeQue<bfloat16_t>();
        AscendC::DataCopyPad(gMixed[static_cast<uint64_t>(row) * width + col], out, outCopy);
        outQue.FreeTensor(out);
    }
}
