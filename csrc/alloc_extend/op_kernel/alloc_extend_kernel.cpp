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
        this->total_block_num = AscendC::GetBlockNum();
        
        this->pre_lens_gm.SetGlobalBuffer((__gm__ int64_t*)pre_lens_in, this->batch_size);  // total data
        this->seq_lens_gm.SetGlobalBuffer((__gm__ int64_t*)seq_lens_in, this->batch_size);
        this->last_loc_gm.SetGlobalBuffer((__gm__ int64_t*)last_loc_in, this->batch_size);
        this->free_pages_gm.SetGlobalBuffer((__gm__ int64_t*)free_pages_in);
        this->out_indices_gm.SetGlobalBuffer((__gm__ int64_t*)out_indices_in, this->total_extend_tokens);
        this->values_gm.SetGlobalBuffer((__gm__ int64_t*)values_in);
        
        this->total_size_aligned = ceil_div(this->batch_size * sizeof(int64_t), byteAlign) * byteAlign;
        this->pipe.InitBuffer(this->input_que, BUFFER_NUM, this->total_size_aligned * 3);
    }
    __aicore__ inline void Process() {
        for (int32_t task_id = this->core_id; task_id < this->batch_size; task_id += this->total_block_num) {
            CopyIn();
            Compute(task_id);
            CopyOut();
        }
    }
private:
    __aicore__ inline void CopyIn() {
        // pad align copy
        AscendC::LocalTensor<int64_t> pre_lens_ub = this->input_que.AllocTensor<int64_t>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->batch_size * sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
        
        AscendC::DataCopyPad(pre_lens_ub, this->pre_lens_gm, copyParams, padParams);
        AscendC::DataCopyPad(pre_lens_ub[this->total_size_aligned], this->seq_lens_gm, copyParams, padParams);
        AscendC::DataCopyPad(pre_lens_ub[this->total_size_aligned * 2], this->last_loc_gm, copyParams, padParams);
        this->input_que.EnQue(pre_lens_ub);
    }
    __aicore__ inline void Compute(int64_t task_id) {
        AscendC::LocalTensor<int64_t> pre_lens_ub = input_que.DeQue<int64_t>();
        AscendC::LocalTensor<int64_t> seq_lens_ub = pre_lens_ub[this->total_size_aligned];
        AscendC::LocalTensor<int64_t> last_loc_ub = pre_lens_ub[this->total_size_aligned * 2];
    
        int64_t extend_lens_sum = 0;
        int64_t num_new_pages_sum = 0;
        int64_t cur_extend_len = 0;
        int64_t cur_need_pages = 0;
        int64_t cur_seq = 0;
        int64_t cur_pre_seq = 0;
        for (int i = 0; i < task_id + 1; i++) {
            cur_seq = seq_lens_ub.GetValue(i);
            cur_pre_seq = pre_lens_ub.GetValue(i);
            cur_extend_len = cur_seq - cur_pre_seq;
            cur_need_pages = (cur_seq + this->page_size - 1) / this->page_size - (cur_pre_seq + this->page_size - 1) / this->page_size;
            extend_lens_sum = extend_lens_sum + cur_extend_len;
            num_new_pages_sum = num_new_pages_sum + cur_need_pages;
        }
        
        int64_t cur_tokens_align = ceil_div(cur_extend_len * sizeof(int64_t), byteAlign) * byteAlign;
        this->cur_extend_len = cur_extend_len;
        this->pipe.InitBuffer(this->out_indices_que, BUFFER_NUM, cur_tokens_align);  // 32 align, maybe >64k
        
        int64_t free_pages_align = ceil_div(num_new_pages_sum * sizeof(int64_t), byteAlign) * byteAlign;
        this->pipe.InitBuffer(this->free_pages_que, BUFFER_NUM, free_pages_align);
        AscendC::LocalTensor<int64_t> free_pages_ub = free_pages_que.AllocTensor<int64_t>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(num_new_pages_sum * sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(free_pages_ub, this->free_pages_gm, copyParams, padParams);
        this->free_pages_que.EnQue(free_pages_ub);
        AscendC::LocalTensor<int64_t> free_pages_ub_read = free_pages_que.DeQue<int64_t>();
        
        int64_t output_start_loc = extend_lens_sum - cur_extend_len;
        this->output_start_loc = output_start_loc;
        int64_t new_pages_start_loc = num_new_pages_sum - cur_need_pages;

        // seq_lens, pre_lens // page_size
        int64_t last_loc = last_loc_ub.GetValue(task_id);
        int64_t num_part1 = (min(cur_seq, (cur_pre_seq + this->page_size - 1) / this->page_size * this->page_size) - cur_pre_seq);
        AscendC::LocalTensor<int64_t> out_indices_ub = out_indices_que.AllocTensor<int64_t>();
        for (int i = 0; i < num_part1; i++) {
            out_indices_ub.SetValue(i, last_loc + 1 + i);
        }
        if (cur_pre_seq + num_part1 == cur_seq) {
            out_indices_que.EnQue(out_indices_ub);
            this->input_que.FreeTensor(pre_lens_ub);
            this->free_pages_que.FreeTensor(free_pages_ub);
            return;
        }
        int64_t num_part2 = cur_seq / this->page_size * this->page_size - (cur_pre_seq + this->page_size - 1) / this->page_size * this->page_size;
        int64_t out_offset = num_part1;
        if (num_part2 > 0) {
            for (int page_i=0; page_i<num_part2 / this->page_size; page_i++) {
                int64_t page_value = free_pages_ub_read.GetValue(new_pages_start_loc + page_i);
                for (int i=0; i<this->page_size; i++) {
                    out_indices_ub.SetValue(out_offset + page_i * this->page_size + i, page_value * this->page_size + i);
                }
            }
        }
        if (cur_pre_seq + num_part1 + num_part2 == cur_seq) {
            out_indices_que.EnQue(out_indices_ub);
            this->input_que.FreeTensor(pre_lens_ub);
            this->free_pages_que.FreeTensor(free_pages_ub);
            return;
        }
        int64_t num_part3 = cur_seq - cur_seq / this->page_size * this->page_size;

        int64_t start_page_loc = free_pages_ub_read.GetValue(new_pages_start_loc + cur_need_pages - 1);
        out_offset = num_part1 + num_part2;

        for (int i = 0; i < num_part3; i++) {
            out_indices_ub.SetValue(out_offset + i, start_page_loc * this->page_size + i);
        }

        if (task_id == this->batch_size -1) {
            values_gm.SetValue(0, num_new_pages_sum);
        }
        out_indices_que.EnQue(out_indices_ub);
        this->input_que.FreeTensor(pre_lens_ub);
        this->free_pages_que.FreeTensor(free_pages_ub);
    }
    __aicore__ inline void CopyOut() {
        AscendC::LocalTensor<int64_t> out_ub = out_indices_que.DeQue<int64_t>();
        uint32_t copy_bytes = static_cast<uint32_t>(this->cur_extend_len) * sizeof(int64_t);  // overflow
        AscendC::DataCopyExtParams copy_params = {1, copy_bytes, 0, 0, 0};
        AscendC::DataCopyPad<int64_t>(this->out_indices_gm[this->output_start_loc], out_ub, copy_params);
        out_indices_que.FreeTensor(out_ub);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> input_que; // 1 for que depth
    AscendC::TQue<AscendC::TPosition::VECIN, 1> out_indices_que;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> free_pages_que;
    AscendC::GlobalTensor<int64_t> pre_lens_gm;
    AscendC::GlobalTensor<int64_t> seq_lens_gm;
    AscendC::GlobalTensor<int64_t> last_loc_gm;
    AscendC::GlobalTensor<int64_t> free_pages_gm;
    AscendC::GlobalTensor<int64_t> out_indices_gm;
    AscendC::GlobalTensor<int64_t> values_gm;

    int32_t core_id;
    int32_t total_block_num;
    int32_t batch_size;
    int32_t total_size_aligned;
    int32_t page_size;
    int32_t used_core_num;
    int64_t total_extend_tokens;
    int64_t output_start_loc;
    int64_t cur_extend_len;
};


extern "C" __global__ __aicore__ void alloc_extend(GM_ADDR pre_lens_in, GM_ADDR seq_lens_in, GM_ADDR last_loc_in,
    GM_ADDR free_pages_in, GM_ADDR out_indices_in, GM_ADDR values_in, GM_ADDR workspace_in, GM_ADDR tiling_gm_in)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelAllocExtent op;
    op.Init(pre_lens_in, seq_lens_in, last_loc_in, free_pages_in, out_indices_in, values_in, workspace_in, tiling_gm_in);
    op.Process();
}

