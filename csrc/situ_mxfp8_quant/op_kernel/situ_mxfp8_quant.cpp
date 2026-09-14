/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"

#if defined(__NPU_ARCH__)

#include "dynamic_mx_quant.h"

namespace SituMxFp8QuantOps {

using namespace AscendC;

constexpr uint32_t INPUT_COLS = 6144;
constexpr uint32_t OUTPUT_COLS = 3072;
constexpr uint32_t SCALE_COLS = OUTPUT_COLS / 32;
constexpr uint32_t FLOAT_OVERFLOW_MODE_CTRL = 60;

template <typename GroupType>
class SituMxFp8Quant
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR group_list, GM_ADDR payload, GM_ADDR scales,
                                uint32_t capacity_rows, uint32_t num_experts,
                                uint32_t group_list_type, float beta, float linear_beta)
    {
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
        group_list_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ GroupType *>(group_list));
        payload_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t *>(payload));
        scales_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(scales));
        capacity_rows_ = capacity_rows;
        num_experts_ = num_experts;
        group_list_type_ = group_list_type;
        beta_ = beta;
        inv_beta_ = 1.0f / beta;
        linear_beta_ = linear_beta;
        inv_linear_beta_ = 1.0f / linear_beta;

        pipe_.InitBuffer(input_buf_, INPUT_COLS * sizeof(bfloat16_t));
        pipe_.InitBuffer(result_buf_, OUTPUT_COLS * sizeof(bfloat16_t));
        pipe_.InitBuffer(payload_buf_, OUTPUT_COLS * sizeof(fp8_e4m3fn_t));
        pipe_.InitBuffer(scales_buf_, SCALE_COLS * sizeof(uint8_t));
        pipe_.InitBuffer(quant_tmp_buf_, 2 * SCALE_COLS * sizeof(uint16_t));
        pipe_.InitBuffer(gate_buf_, OUTPUT_COLS * sizeof(float));
        pipe_.InitBuffer(up_buf_, OUTPUT_COLS * sizeof(float));
        pipe_.InitBuffer(tanh_buf_, OUTPUT_COLS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t valid_rows = GetValidRows();
        if (valid_rows > capacity_rows_) {
            valid_rows = capacity_rows_;
        }
        if (valid_rows == 0) {
            return;
        }

        const uint32_t core_num = GetBlockNum();
        const uint32_t rows_per_core = (valid_rows + core_num - 1) / core_num;
        const uint32_t row_begin = GetBlockIdx() * rows_per_core;
        const uint32_t candidate_end = row_begin + rows_per_core;
        const uint32_t row_end = candidate_end < valid_rows ? candidate_end : valid_rows;
        for (uint32_t row = row_begin; row < row_end; ++row) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline uint32_t GetValidRows()
    {
        if (group_list_type_ == 0) {
            const int64_t value = static_cast<int64_t>(group_list_gm_.GetValue(num_experts_ - 1));
            return value > 0 ? static_cast<uint32_t>(value) : 0U;
        }
        int64_t total = 0;
        for (uint32_t i = 0; i < num_experts_; ++i) {
            total += static_cast<int64_t>(group_list_gm_.GetValue(i));
        }
        return total > 0 ? static_cast<uint32_t>(total) : 0U;
    }

    __aicore__ inline void StableTanh(LocalTensor<float> &dst, float multiplier, float inverse_multiplier)
    {
        // tanh(x) = 2 * sigmoid(2x) - 1. AscendC's sigmoid primitive avoids
        // non-finite intermediates from a hand-written exp/div sequence.
        Muls(dst, dst, 2.0f * inverse_multiplier, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();
        Sigmoid<float, false>(dst, dst, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();
        Muls(dst, dst, 2.0f * multiplier, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();
        Adds(dst, dst, -multiplier, OUTPUT_COLS);
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        auto input = input_buf_.Get<bfloat16_t>();
        auto result = result_buf_.Get<bfloat16_t>();
        auto payload = payload_buf_.Get<fp8_e4m3fn_t>();
        auto scales = scales_buf_.Get<uint8_t>();
        auto quant_tmp = quant_tmp_buf_.Get<uint16_t>();
        auto gate = gate_buf_.Get<float>();
        auto up = up_buf_.Get<float>();
        auto tanh_value = tanh_buf_.Get<float>();

        const uint64_t input_offset = static_cast<uint64_t>(row) * INPUT_COLS;
        DataCopy(input, x_gm_[input_offset], INPUT_COLS);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        Cast(gate, input, RoundMode::CAST_NONE, OUTPUT_COLS);
        Cast(up, input[OUTPUT_COLS], RoundMode::CAST_NONE, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();

        // beta*tanh(gate/beta) * sigmoid(gate)
        Muls(tanh_value, gate, 1.0f, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();
        StableTanh(tanh_value, beta_, inv_beta_);
        PipeBarrier<PIPE_V>();
        Sigmoid<float, false>(gate, gate, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, tanh_value, OUTPUT_COLS);

        // linear_beta*tanh(up/linear_beta)
        StableTanh(up, linear_beta_, inv_linear_beta_);
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, up, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();

        // Match the unfused path's BF16 SiTU output before MXFP8 quantization.
        Cast(result, gate, RoundMode::CAST_RINT, OUTPUT_COLS);
        PipeBarrier<PIPE_V>();

        __ubuf__ bfloat16_t *src = reinterpret_cast<__ubuf__ bfloat16_t *>(result.GetPhyAddr());
        __ubuf__ uint16_t *max_exp = reinterpret_cast<__ubuf__ uint16_t *>(quant_tmp.GetPhyAddr());
        __ubuf__ uint16_t *half_scale =
            reinterpret_cast<__ubuf__ uint16_t *>(quant_tmp[SCALE_COLS].GetPhyAddr());
        __ubuf__ uint16_t *mx_scale = reinterpret_cast<__ubuf__ uint16_t *>(scales.GetPhyAddr());
        __ubuf__ int8_t *out = reinterpret_cast<__ubuf__ int8_t *>(payload.GetPhyAddr());

        quant::ComputeMaxExp(src, max_exp, OUTPUT_COLS);
        quant::ComputeScale<fp8_e4m3fn_t>(max_exp, mx_scale, half_scale, SCALE_COLS);
        quant::ComputeFp8Data<bfloat16_t, fp8_e4m3fn_t, RoundMode::CAST_TRUNC, RoundMode::CAST_RINT, true>(
            src, half_scale, out, OUTPUT_COLS);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        const uint64_t payload_offset = static_cast<uint64_t>(row) * OUTPUT_COLS;
        const uint64_t scale_offset = static_cast<uint64_t>(row) * SCALE_COLS;
        DataCopy(payload_gm_[payload_offset], payload, OUTPUT_COLS);
        DataCopy(scales_gm_[scale_offset], scales, SCALE_COLS);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> input_buf_, result_buf_, payload_buf_, scales_buf_, quant_tmp_buf_;
    TBuf<TPosition::VECCALC> gate_buf_, up_buf_, tanh_buf_;
    GlobalTensor<bfloat16_t> x_gm_;
    GlobalTensor<GroupType> group_list_gm_;
    GlobalTensor<fp8_e4m3fn_t> payload_gm_;
    GlobalTensor<uint8_t> scales_gm_;
    uint32_t capacity_rows_{0};
    uint32_t num_experts_{0};
    uint32_t group_list_type_{1};
    float beta_{4.0f};
    float inv_beta_{0.25f};
    float linear_beta_{25.0f};
    float inv_linear_beta_{0.04f};
};

}  // namespace SituMxFp8QuantOps

extern "C" __global__ __aicore__ void situ_mxfp8_quant(
    GM_ADDR x, GM_ADDR group_list, GM_ADDR payload, GM_ADDR scales,
    uint32_t capacity_rows, uint32_t num_experts, uint32_t group_list_type,
    uint32_t group_dtype, float beta, float linear_beta)
{
    int64_t old_mode = AscendC::GetCtrlSpr<SituMxFp8QuantOps::FLOAT_OVERFLOW_MODE_CTRL,
                                           SituMxFp8QuantOps::FLOAT_OVERFLOW_MODE_CTRL>();
    AscendC::SetCtrlSpr<SituMxFp8QuantOps::FLOAT_OVERFLOW_MODE_CTRL,
                        SituMxFp8QuantOps::FLOAT_OVERFLOW_MODE_CTRL>(0);
    if (group_dtype == 0) {
        SituMxFp8QuantOps::SituMxFp8Quant<int32_t> op;
        op.Init(x, group_list, payload, scales, capacity_rows, num_experts,
                group_list_type, beta, linear_beta);
        op.Process();
    } else {
        SituMxFp8QuantOps::SituMxFp8Quant<int64_t> op;
        op.Init(x, group_list, payload, scales, capacity_rows, num_experts,
                group_list_type, beta, linear_beta);
        op.Process();
    }
    AscendC::SetCtrlSpr<SituMxFp8QuantOps::FLOAT_OVERFLOW_MODE_CTRL,
                        SituMxFp8QuantOps::FLOAT_OVERFLOW_MODE_CTRL>(old_mode);
}

#endif  // defined(__NPU_ARCH__)
