// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* include file of ascendc */
#include "kernel_operator.h"
#include "../op_host/alloc_extend_tiling.h"
/* tensor num for each queue */
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t byteAlign = 32;
__aicore__ inline uint32_t ceil_div(int64_t a, int64_t b)
{
    if (b == 0)
        return a;
    return (a + b - 1) / b;
}

class KernelAllocExtent {
public:
    __aicore__ inline KernelAllocExtent() {}
    __aicore__ inline void Init(GM_ADDR pre_lens_in, GM_ADDR seq_lens_in, GM_ADDR last_loc_in,
        GM_ADDR free_pages_in, GM_ADDR out_indices_in, GM_ADDR values_in, 
        GM_ADDR workspace_in, GM_ADDR tiling_gm_in) {
        auto tiling_gm = reinterpret_cast<__gm__ sglang::npu_kernel::AllocExtendTilingData *>(tiling_gm_in);
        this->batch_size = tiling_gm->batch_size;
        this->page_size = tiling_gm->page_size;
        this->used_core_num = tiling_gm->used_core_num;
        this->total_extend_tokens = tiling_gm->total_extend_tokens;
        this->core_id = AscendC::GetBlockIdx();
        this->block_num = AscendC::GetBlockNum();
        
        this->pre_lens_gm.SetGlobalBuffer((__gm__ int64_t*)pre_lens_in, this->batch_size);  // 全量数据，有效部分sum
        this->seq_lens_gm.SetGlobalBuffer((__gm__ int64_t*)seq_lens_in, this->batch_size);
        this->last_loc_gm.SetGlobalBuffer((__gm__ int64_t*)last_loc_in, this->batch_size);
        this->free_pages_gm.SetGlobalBuffer((__gm__ int64_t*)free_pages_in);
        this->out_indices_gm.SetGlobalBuffer((__gm__ int64_t*)out_indices_in, this->total_extend_tokens);
        this->values_gm.SetGlobalBuffer((__gm__ int64_t*)values_in);
        
        this->total_size_aligned = ceil_div(this->batch_size * sizeof(int64_t), byteAlign) * byteAlign;
        this->pipe.InitBuffer(this->pre_lens_que, BUFFER_NUM, this->total_size_aligned); // align
        this->pipe.InitBuffer(this->seq_lens_que, BUFFER_NUM, this->total_size_aligned);
        this->pipe.InitBuffer(this->out_indices_que, BUFFER_NUM, this->total_size_aligned);

        int32_t data_size = ceil_div(this->batch_size * sizeof(int32_t), byteAlign) * byteAlign;
        this->pipe.InitBuffer(tmp_pre_lens_que, data_size); // 单位字节
        this->pipe.InitBuffer(tmp_seq_lens_que, data_size);
        this->pipe.InitBuffer(tmp_out_indices_que, data_size);

        AscendC::printf("cur cored_id, block_num, bs, bs_type_align: %d, %d, %d, %d\n", this->core_id, this->block_num, this->batch_size, this->total_size_aligned);
    }
    __aicore__ inline void Process() {
        for (int32_t task_id = this->core_id; task_id < this->batch_size; task_id += this->block_num) {
            CopyIn();
            Compute(task_id);
            CopyOut();
        }
    }
private:
    __aicore__ inline void CopyIn() {
        // pad对齐拷贝
        AscendC::LocalTensor<int64_t> pre_lens_ub = this->pre_lens_que.AllocTensor<int64_t>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->batch_size * sizeof(int64_t)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        AscendC::DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(pre_lens_ub, this->pre_lens_gm, copyParams, padParams); // 从GM->VECIN搬运40Bytes
        this->pre_lens_que.EnQue(pre_lens_ub);
        // AscendC::DumpTensor(pre_lens_ub, 5, this->batch_size);
        
        AscendC::LocalTensor<int64_t> seq_lens_ub = this->seq_lens_que.AllocTensor<int64_t>();
        AscendC::DataCopyPad(seq_lens_ub, this->seq_lens_gm, copyParams, padParams);
        this->seq_lens_que.EnQue(seq_lens_ub);
        // AscendC::DumpTensor(seq_lens_ub, 5, this->batch_size);
        // AscendC::DumpAccChkPoint(seq_lens_ub, 0, 0, 1);
    }
    __aicore__ inline void Compute(int64_t task_id) {
        AscendC::LocalTensor<int64_t> pre_lens_ub = pre_lens_que.DeQue<int64_t>();
        AscendC::LocalTensor<int64_t> seq_lens_ub = seq_lens_que.DeQue<int64_t>();
    
        int32_t extend_lens_sum = 0;
        int32_t num_new_pages_sum = 0;
        int32_t cur_extend_len = 0;
        int32_t cur_need_pages = 0;
        int32_t cur_seq = 0;
        int32_t cur_pre_seq = 0;
        for (int i = 0; i < task_id + 1; i++) {
            cur_seq = seq_lens_ub.GetValue(i);
            cur_pre_seq = pre_lens_ub.GetValue(i);
            cur_extend_len = cur_seq - cur_pre_seq;
            cur_need_pages = (cur_seq + this->page_size - 1) / this->page_size - (cur_pre_seq + this->page_size - 1) / this->page_size;
            extend_lens_sum = extend_lens_sum + cur_extend_len;
            num_new_pages_sum = num_new_pages_sum + cur_need_pages;
        }
        int32_t output_start_loc = extend_lens_sum - cur_extend_len;
        int32_t new_pages_start_loc = num_new_pages_sum - cur_need_pages;
        AscendC::printf("extend_lens_sum, cur_extend_len, num_new_pages_sum, cur_need_pages, cur_seq, cur_pre_seq: %d, %d, %d %d %d %d \n", 
            extend_lens_sum, cur_extend_len, num_new_pages_sum, cur_need_pages, cur_seq, cur_pre_seq);
        // seq_lens, pre_lens // page_size
        if (task_id == this->batch_size -1) {
            int64_t ret_values = (int64_t)((uint64_t)(uint32_t)num_new_pages_sum << 32 | (uint32_t)extend_lens_sum);
            values_gm.SetValue(0, ret_values);
            AscendC::printf("ret_values: %d\n", ret_values);
        }
        int32_t last_loc = this->last_loc_gm.GetValue(task_id);
        int32_t num_part1 = (min(cur_seq, (cur_pre_seq + this->page_size - 1) / this->page_size * this->page_size) - cur_pre_seq);
        AscendC::LocalTensor<int64_t> out_indices_ub = out_indices_que.AllocTensor<int64_t>();
        for (int i=0; i<num_part1; i++) {
            out_indices_gm.SetValue(output_start_loc + i, last_loc + 1 + i);
            AscendC::printf("num_part1, output_start_loc, last_loc: %d %d %d \n", num_part1, output_start_loc + i, last_loc + 1 + i);
        }
        if (cur_pre_seq + num_part1 == cur_seq) {
            out_indices_que.EnQue(out_indices_ub);
            pre_lens_que.FreeTensor(pre_lens_ub);
            seq_lens_que.FreeTensor(seq_lens_ub);
            return;
        }
        int32_t num_part2 = cur_seq / this->page_size * this->page_size - (cur_pre_seq + this->page_size - 1) / this->page_size * this->page_size;
        AscendC::printf("output_start_loc, num_part2: %d %d \n", output_start_loc, num_part2);
        if (num_part2 > 0) {
            int32_t out_offset = output_start_loc + num_part1;
            for (int page_i=0; page_i<num_part2 / this->page_size; page_i++) {
                int32_t page_value = free_pages_gm.GetValue(new_pages_start_loc + page_i);
                AscendC::printf("out_offset, page_i, new_pages_start_loc, page_value: %d %d %d %d\n", out_offset, page_i, new_pages_start_loc, page_value);
                for (int i=0; i<this->page_size; i++) {
                    out_indices_gm.SetValue(out_offset + page_i * this->page_size + i, page_value * this->page_size + i);
                }
            }
        }
        if (cur_pre_seq + num_part1 + num_part2 == cur_seq) {
            out_indices_que.EnQue(out_indices_ub);
            pre_lens_que.FreeTensor(pre_lens_ub);
            seq_lens_que.FreeTensor(seq_lens_ub);
            return;
        }
        int32_t num_part3 = cur_seq - cur_seq / this->page_size * this->page_size;
        int32_t start_page_loc = free_pages_gm.GetValue(new_pages_start_loc + cur_need_pages - 1);
        int32_t out_offset = output_start_loc + num_part1 + num_part2;
        AscendC::printf("out_offset, num_part3, start_page_loc: %d %d %d\n", out_offset, num_part3, start_page_loc);
        for (int i=0; i<num_part3; i++) {
            out_indices_gm.SetValue(out_offset + i, start_page_loc * this->page_size + i);
        }
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(out_indices_gm);
        out_indices_que.EnQue(out_indices_ub);
        pre_lens_que.FreeTensor(pre_lens_ub);
        seq_lens_que.FreeTensor(seq_lens_ub);
    }
    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<int64_t> yLocal = out_indices_que.DeQue<int64_t>();
        uint16_t reducedBytes = static_cast<uint32_t>(this->total_extend_tokens * sizeof(int64_t));
        AscendC::DataCopyExtParams copyParams = {1, reducedBytes, 0, 0, 0};
        // AscendC::DataCopyPad<int64_t>(this->out_indices_gm, yLocal, copyParams);
        out_indices_que.FreeTensor(yLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_pre_lens_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_seq_lens_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_out_indices_que;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> pre_lens_que;  // 1 for que depth
    AscendC::TQue<AscendC::TPosition::VECIN, 1> seq_lens_que;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> out_indices_que;
    AscendC::GlobalTensor<int64_t> pre_lens_gm;
    AscendC::GlobalTensor<int64_t> seq_lens_gm;
    AscendC::GlobalTensor<int64_t> last_loc_gm;
    AscendC::GlobalTensor<int64_t> free_pages_gm;
    AscendC::GlobalTensor<int64_t> out_indices_gm;
    AscendC::GlobalTensor<int64_t> values_gm;

    int32_t core_id;
    int32_t block_num;
    int32_t batch_size;
    int32_t total_size_aligned;
    int32_t page_size;
    int32_t used_core_num;
    int32_t total_extend_tokens;
};


extern "C" __global__ __aicore__ void alloc_extend(GM_ADDR pre_lens_in, GM_ADDR seq_lens_in, GM_ADDR last_loc_in,
    GM_ADDR free_pages_in, GM_ADDR out_indices_in, GM_ADDR values_in, GM_ADDR workspace_in, GM_ADDR tiling_gm_in)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelAllocExtent op;
    op.Init(pre_lens_in, seq_lens_in, last_loc_in, free_pages_in, out_indices_in, values_in, workspace_in, tiling_gm_in);
    op.Process();
}

