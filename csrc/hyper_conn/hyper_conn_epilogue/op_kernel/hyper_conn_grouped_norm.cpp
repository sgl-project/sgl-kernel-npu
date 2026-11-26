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
using sglang::npu_kernel::hyper_conn::UB_BLOCK;

extern "C" __global__ __aicore__ void sgl_hyper_conn_grouped_norm(GM_ADDR src, GM_ADDR weight, GM_ADDR dst,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglHyperConnGroupedNormTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t groups = tilingData.groups;
    const uint32_t width = tilingData.width;
    const uint32_t weightLen = tilingData.weightLen;
    const float eps = tilingData.eps;
    const float invWidth = tilingData.invWidth;

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
