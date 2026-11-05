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

/* BF16 hyper-connection epilogue kernels.
 *
 * Four vector kernels for the gated-residual (hyper-connection) epilogue of a
 * transformer block: a grouped RMS norm over the branch axis, a scaled SiLU on
 * the low-rank gate activation, a gated reduction over the branch axis, and a
 * gated residual combine that broadcasts one block output back onto every
 * branch. The projections between them stay on the caller's matmul.
 *
 * Numerics. Every kernel evaluates its intermediate arithmetic in fp32
 * throughout and rounds to bf16 once when writing its output tensor to global
 * memory, matching the upstream SGLang CUDA kernels rather than the eager
 * fallback. fp32 -> bf16 uses CAST_RINT, which matches torch's `.bfloat16()`.
 *
 * Synchronization. These kernels build into the no_workspace_kernel target,
 * which does not pass --cce-auto-sync=off, so the compiler inserts automatic
 * synchronization in addition to the explicit SetFlag/WaitFlag pairs below.
 * The explicit pairs are therefore likely redundant. Removing them, or moving
 * these kernels to a dedicated target with auto-sync off, needs an on-device
 * measurement before either is chosen; see the PR discussion.
 *
 * Shape contract, enforced by the host before launch:
 *   - `scale`/`hc` is a power of two, so multiplying by the reciprocal is
 *     exactly the division the reference performs
 *   - `tile` divides `width` and `tile * sizeof(bf16)` is a multiple of 32, so
 *     the strided branch copy lands contiguously in UB and no tile is partial
 */

#include "hyper_conn_round.h"

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

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

/* Grouped RMS norm with a unit-offset weight.
 *
 *   inv    = rsqrt(sum(x^2) / width + eps)                      [fp32]
 *   dst[j] = bf16((x[j] * inv) * (1 + weight[(g * width) % weightLen + j]))
 *
 * `groups` is the number of `width`-long groups in `src`. The cycling weight
 * offset gives each hyper-connection branch its own weight slice when the
 * weight is longer than one group; a weight of exactly one group is shared by
 * all groups and is then loaded once outside the loop.
 *
 * `invWidth` is passed in because the aicore scalar unit cannot convert an
 * unsigned integer to float.
 */
extern "C" __global__ __aicore__ void hyper_conn_grouped_norm_bf16(GM_ADDR src, GM_ADDR weight, GM_ADDR dst,
                                                                   uint32_t groups, uint32_t width, uint32_t weightLen,
                                                                   float eps, float invWidth)
{
    const uint32_t cores = AscendC::GetBlockNum();
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t bytesBf16 = width * BF16_SIZE;
    const uint32_t padBf16 = AlignUp(bytesBf16, UB_BLOCK);
    const uint32_t padF32 = AlignUp(width * F32_SIZE, UB_BLOCK);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<TPosition::VECCALC> weightBuf, valueBuf, scaleBuf, squareBuf, reduceBuf, scalarBuf, oneBuf;
    pipe.InitBuffer(inQue, 2, padBf16);
    pipe.InitBuffer(outQue, 2, padBf16);
    pipe.InitBuffer(weightBuf, padBf16);
    pipe.InitBuffer(valueBuf, padF32);
    pipe.InitBuffer(scaleBuf, padF32);
    pipe.InitBuffer(squareBuf, padF32);
    pipe.InitBuffer(reduceBuf, padF32);
    pipe.InitBuffer(scalarBuf, UB_BLOCK);
    pipe.InitBuffer(oneBuf, UB_BLOCK);

    GlobalTensor<bfloat16_t> gSrc, gWeight, gDst;
    gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));
    gWeight.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weight));
    gDst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dst));

    LocalTensor<bfloat16_t> weightBf = weightBuf.Get<bfloat16_t>();
    LocalTensor<float> value = valueBuf.Get<float>();
    LocalTensor<float> scale = scaleBuf.Get<float>();
    LocalTensor<float> square = squareBuf.Get<float>();
    LocalTensor<float> reduce = reduceBuf.Get<float>();
    LocalTensor<float> scalar = scalarBuf.Get<float>();
    LocalTensor<float> ones = oneBuf.Get<float>();
    AscendC::Duplicate(ones, 1.0f, GATE_LANES);

    const AscendC::DataCopyExtParams copy{1, bytesBf16, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
    const bool sharedWeight = (weightLen == width);
    if (sharedWeight) {
        AscendC::DataCopyPad(weightBf, gWeight, copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(scale, weightBf, RoundMode::CAST_NONE, width);
        AscendC::Adds(scale, scale, 1.0f, width);
    }

    for (uint32_t group = core; group < groups; group += cores) {
        const uint64_t offset = static_cast<uint64_t>(group) * width;
        if (!sharedWeight) {
            const uint32_t base = static_cast<uint32_t>(offset % weightLen);
            AscendC::DataCopyPad(weightBf, gWeight[base], copy, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(scale, weightBf, RoundMode::CAST_NONE, width);
            AscendC::Adds(scale, scale, 1.0f, width);
        }

        LocalTensor<bfloat16_t> in = inQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(in, gSrc[offset], copy, pad);
        inQue.EnQue(in);
        in = inQue.DeQue<bfloat16_t>();
        AscendC::Cast(value, in, RoundMode::CAST_NONE, width);
        inQue.FreeTensor(in);

        AscendC::Mul(square, value, value, width);
        AscendC::ReduceSum<float>(scalar, square, reduce, width);
        AscendC::Muls(scalar, scalar, invWidth, GATE_LANES);
        AscendC::Adds(scalar, scalar, eps, GATE_LANES);
        AscendC::Sqrt(scalar, scalar, GATE_LANES);
        AscendC::Div(scalar, ones, scalar, GATE_LANES);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float inv = scalar.GetValue(0);

        AscendC::Muls(value, value, inv, width);
        AscendC::Mul(value, value, scale, width);
        LocalTensor<bfloat16_t> out = outQue.AllocTensor<bfloat16_t>();
        AscendC::Cast(out, value, RoundMode::CAST_RINT, width);
        outQue.EnQue(out);
        out = outQue.DeQue<bfloat16_t>();
        AscendC::DataCopyPad(gDst[offset], out, copy);
        outQue.FreeTensor(out);
    }
}

/* Scaled SiLU on the low-rank gate activation.
 *
 *   t      = x / scale         [fp32] (scale a power of two, so exact)
 *   dst[i] = bf16(t * sigmoid(t))
 *
 * `srcStride` is the row pitch of `src` in elements, so the kernel can read
 * the leading `cols` columns of a wider matmul output without materializing a
 * contiguous slice. `dst` is compact [rows, cols].
 *
 * UB budget at the maximum `cols = 4096`: two double-buffered bf16 queues
 * (32768 B) plus four fp32 buffers (`valueBuf`, `tmpBuf`, `oneBuf`, `sigBuf`,
 * 65536 B) = 98304 B (96 KB), well below the 192 KB Unified Buffer.
 */
extern "C" __global__ __aicore__ void hyper_conn_scaled_silu_bf16(GM_ADDR src, GM_ADDR dst, uint32_t rows,
                                                                  uint32_t cols, uint32_t srcStride, float invScale)
{
    const uint32_t cores = AscendC::GetBlockNum();
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t padBf16 = AlignUp(cols * BF16_SIZE, UB_BLOCK);
    const uint32_t padF32 = AlignUp(cols * F32_SIZE, UB_BLOCK);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<TPosition::VECCALC> valueBuf, tmpBuf, oneBuf, sigBuf;
    pipe.InitBuffer(inQue, 2, padBf16);
    pipe.InitBuffer(outQue, 2, padBf16);
    pipe.InitBuffer(valueBuf, padF32);
    pipe.InitBuffer(tmpBuf, padF32);
    pipe.InitBuffer(oneBuf, padF32);
    pipe.InitBuffer(sigBuf, padF32);

    GlobalTensor<bfloat16_t> gSrc, gDst;
    gSrc.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));
    gDst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dst));

    LocalTensor<float> value = valueBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> ones = oneBuf.Get<float>();
    LocalTensor<float> sig = sigBuf.Get<float>();
    AscendC::Duplicate(ones, 1.0f, cols);

    const AscendC::DataCopyExtParams copy{1, cols * BF16_SIZE, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
    for (uint32_t row = core; row < rows; row += cores) {
        LocalTensor<bfloat16_t> in = inQue.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(in, gSrc[static_cast<uint64_t>(row) * srcStride], copy, pad);
        inQue.EnQue(in);
        in = inQue.DeQue<bfloat16_t>();
        AscendC::Cast(value, in, RoundMode::CAST_NONE, cols);
        inQue.FreeTensor(in);

        AscendC::Muls(value, value, invScale, cols);
        SigmoidF32(sig, value, tmp, ones, cols);
        AscendC::Mul(value, value, sig, cols);

        LocalTensor<bfloat16_t> out = outQue.AllocTensor<bfloat16_t>();
        AscendC::Cast(out, value, RoundMode::CAST_RINT, cols);
        outQue.EnQue(out);
        out = outQue.DeQue<bfloat16_t>();
        AscendC::DataCopyPad(gDst[static_cast<uint64_t>(row) * cols], out, copy);
        outQue.FreeTensor(out);
    }
}

/* Gated reduction over the hyper-connection branch axis.
 *
 *   mixed[m,j] = bf16( sum_c (sigmoid(logits[m,c,j]) * normed[m,c,j]) / hc )
 *
 * `normed` and `logits` are [rows, hc * width]; `mixed` is [rows, width]. One
 * task is one (row, column tile): all hc branches of a tile arrive in a single
 * strided MTE2 burst, so the elementwise ladder runs once over hc * tile lanes
 * instead of hc times over tile lanes.
 */
extern "C" __global__ __aicore__ void hyper_conn_branch_mix_bf16(GM_ADDR normed, GM_ADDR logits, GM_ADDR mixed,
                                                                 uint32_t rows, uint32_t hc, uint32_t width,
                                                                 uint32_t tile, float invHc)
{
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

    // hc blocks of `tile` elements, stepping one branch (width) apart in GM
    // and landing back to back in UB (tile * 2 is a multiple of 32 by
    // contract).
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
        // Sum over the branch axis in branch order, matching a reduction over
        // the [hc, tile] product tile.
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

/* Gated residual combine.
 *
 *   g[m,c]     = 2 * sigmoid(logits[m,c] / hc)                  [fp32]
 *   out[m,c,j] = bf16(residual[m,c,j] + block[m,j] * g[m,c])
 *
 * `logits` is the [rows, hc] injection-projection output, read with row pitch
 * `logitsStride` so it can be a column slice of a wider matmul output. The
 * per-row gate is computed once and cached across that row's column tiles.
 */
extern "C" __global__ __aicore__ void hyper_conn_branch_combine_bf16(GM_ADDR block, GM_ADDR residual, GM_ADDR logits,
                                                                     GM_ADDR out, uint32_t rows, uint32_t hc,
                                                                     uint32_t width, uint32_t tile,
                                                                     uint32_t logitsStride, float invHc)
{
    static_assert(MAX_BRANCHES <= GATE_LANES, "per-row gate buffer holds GATE_LANES lanes");

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
