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

extern "C" __global__ __aicore__ void sgl_hyper_conn_scaled_silu(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace,
                                                                 GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglHyperConnScaledSiluTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    const uint32_t rows = tilingData.rows;
    const uint32_t cols = tilingData.cols;
    const uint32_t srcStride = tilingData.srcStride;
    const float invScale = tilingData.invScale;

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
