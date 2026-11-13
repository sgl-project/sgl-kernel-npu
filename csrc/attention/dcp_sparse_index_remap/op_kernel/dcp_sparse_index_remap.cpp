#include "kernel_operator.h"

#ifdef SGL_DCP_CANN_PACKAGE
#include "dcp_sparse_index_remap_tiling.h"
#endif

namespace {
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kRowBytes = kTopK * sizeof(int32_t);
constexpr uint32_t kMaskBytes = kTopK / 8;
constexpr uint32_t kInt32PerBlock = 8;

class Remap
{
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t rows)
    {
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(input), rows * kTopK);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output), rows * kTopK);
        pipe_.InitBuffer(in_queue_, 1, kRowBytes);
        pipe_.InitBuffer(out_queue_, 1, kRowBytes);
        pipe_.InitBuffer(page_, kRowBytes);
        pipe_.InitBuffer(local_, kRowBytes);
        pipe_.InitBuffer(temp_, kRowBytes);
        pipe_.InitBuffer(compare_, kRowBytes);
        pipe_.InitBuffer(owner_mask_, kMaskBytes);
        pipe_.InitBuffer(valid_mask_, kMaskBytes);
    }

    __aicore__ inline void Run(uint32_t rows, int32_t page_shift, int32_t dcp_shift, float rank)
    {
        for (uint32_t row = AscendC::GetBlockIdx(); row < rows; row += AscendC::GetBlockNum()) {
            auto input = in_queue_.AllocTensor<int32_t>();
            AscendC::DataCopy(input, input_[row * kTopK], kTopK);
            in_queue_.EnQue(input);
            input = in_queue_.DeQue<int32_t>();
            auto page = page_.Get<int32_t>();
            auto local = local_.Get<int32_t>();
            auto temp = temp_.Get<int32_t>();
            auto compare = compare_.Get<float>();
            auto owner_mask = owner_mask_.Get<uint8_t>();
            auto valid_mask = valid_mask_.Get<uint8_t>();
            auto output = out_queue_.AllocTensor<int32_t>();

            AscendC::ShiftRight(page, input, page_shift, kTopK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftRight(local, page, dcp_shift, kTopK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ShiftLeft(temp, local, dcp_shift, kTopK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(temp, page, temp, kTopK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(compare, temp, AscendC::RoundMode::CAST_NONE, kTopK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(owner_mask, compare, rank, AscendC::CMPMODE::EQ, kTopK);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(compare, input, AscendC::RoundMode::CAST_NONE, kTopK);
            AscendC::Sub(temp, page, local, kTopK);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(valid_mask, compare, 0.0f, AscendC::CMPMODE::GE, kTopK);
            AscendC::ShiftLeft(temp, temp, page_shift, kTopK);
            AscendC::PipeBarrier<PIPE_V>();
            auto owner_bits = owner_mask.ReinterpretCast<uint16_t>();
            AscendC::And(owner_bits, owner_bits, valid_mask.ReinterpretCast<uint16_t>(), kMaskBytes / 2);
            AscendC::Sub(local, input, temp, kTopK);
            AscendC::PipeBarrier<PIPE_V>();

            uint64_t count = 0;
            AscendC::GatherMask(output, local, owner_mask.ReinterpretCast<uint32_t>(), true, kTopK, {1, 1, 0, 0},
                                count);
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t aligned = (count + kInt32PerBlock - 1) / kInt32PerBlock * kInt32PerBlock;
            for (uint32_t i = count; i < aligned; ++i) {
                output.SetValue(i, -1);
            }
            if (aligned < kTopK) {
                AscendC::Duplicate(output[aligned], static_cast<int32_t>(-1), kTopK - aligned);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            in_queue_.FreeTensor(input);
            out_queue_.EnQue(output);
            output = out_queue_.DeQue<int32_t>();
            AscendC::DataCopy(output_[row * kTopK], output, kTopK);
            out_queue_.FreeTensor(output);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_queue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_queue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> page_, local_, temp_, compare_, owner_mask_, valid_mask_;
    AscendC::GlobalTensor<int32_t> input_, output_;
};
}  // namespace

#ifdef SGL_DCP_CANN_PACKAGE
extern "C" __global__ __aicore__ void sgl_dcp_sparse_index_remap(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                                 GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::SglDcpSparseIndexRemapTilingData);
    GET_TILING_DATA(data, tiling);
    Remap op;
    op.Init(input, output, data.rows);
    op.Run(data.rows, data.page_shift, data.dcp_shift, static_cast<float>(data.dcp_rank));
}
#else
extern "C" __global__ __aicore__ void sgl_dcp_sparse_index_remap(GM_ADDR input, GM_ADDR output, uint32_t rows,
                                                                 int32_t page_shift, int32_t dcp_shift,
                                                                 int32_t dcp_rank)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Remap op;
    op.Init(input, output, rows);
    op.Run(rows, page_shift, dcp_shift, static_cast<float>(dcp_rank));
}
#endif
