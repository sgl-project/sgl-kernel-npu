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
#include "../op_host/build_tree_tiling.h"

/* tensor num for each queue */
constexpr int32_t BUFFER_NUM = 1;
constexpr int64_t ALIGN_32B = 32;

class KernalBuildTree
{
public:
    __aicore__ inline KernalBuildTree() {}

    __aicore__ inline void Init(GM_ADDR parent_list,       // [bs, topk * (depth - 1) + 1)]
                                GM_ADDR selected_index,    // [bs, draft_token_num - 1]
                                GM_ADDR verified_seq_len,  // [bs]
                                GM_ADDR tree_mask,
                                GM_ADDR positions,             // [bs * draft_token]
                                GM_ADDR retrive_index,         // [bs, draft_token]
                                GM_ADDR retrive_next_token,    // [bs, draft_token]
                                GM_ADDR retrive_next_sibling,  // [bs, draft_token]
                                GM_ADDR workspace_in, GM_ADDR tiling_gm_in)
    {
        auto tiling = reinterpret_cast<__gm__ sglang::npu_kernel::BuildTreeTilingData *>(tiling_gm_in);
        this->topk = tiling->topk;
        this->depth = tiling->depth;
        this->draft_token_num = static_cast<uint16_t>(tiling->draft_token_num);
        this->tree_mask_mode = tiling->tree_mask_mode;

        this->batch_size = tiling->batch_size;
        this->mask_size = static_cast<uint16_t>(tiling->mask_size);

        this->bs_offset = AscendC::GetBlockIdx() * tiling->big_core_tile_num;
        if (AscendC::GetBlockIdx() < tiling->big_core_num) {
            this->bs_per_core = tiling->big_core_tile_num;
        } else {
            this->bs_per_core = tiling->small_core_tile_num;
            this->bs_offset -= (tiling->big_core_tile_num - tiling->small_core_tile_num) *
                               (AscendC::GetBlockIdx() - tiling->big_core_num);
        }

        // global addr
        this->verified_seq_len_gm.SetGlobalBuffer((__gm__ int64_t *)verified_seq_len, this->batch_size);
        this->tree_mask_gm.SetGlobalBuffer((__gm__ bool *)tree_mask, this->mask_size);

        // tiling offset addr
        auto stride_dim1 = this->topk * (this->depth - 1) + 1;
        auto offset = stride_dim1 * this->bs_offset;
        auto buffer_size = this->bs_per_core * stride_dim1;
        this->parent_list_gm.SetGlobalBuffer((__gm__ int64_t *)parent_list + offset, buffer_size);

        stride_dim1 = this->draft_token_num - 1;
        offset = stride_dim1 * this->bs_offset;
        buffer_size = this->bs_per_core * stride_dim1;
        this->selected_index_gm.SetGlobalBuffer((__gm__ int64_t *)selected_index + offset, buffer_size);

        stride_dim1 = this->draft_token_num;
        offset = stride_dim1 * this->bs_offset;
        buffer_size = this->bs_per_core * stride_dim1;
        this->positions_gm.SetGlobalBuffer((__gm__ int64_t *)positions + offset, buffer_size);
        this->retrive_index_gm.SetGlobalBuffer((__gm__ int64_t *)retrive_index + offset, buffer_size);
        this->retrive_next_token_gm.SetGlobalBuffer((__gm__ int64_t *)retrive_next_token + offset, buffer_size);
        this->retrive_next_sibling_gm.SetGlobalBuffer((__gm__ int64_t *)retrive_next_sibling + offset, buffer_size);

        // init buffer
        auto buf_len = (this->draft_token_num * sizeof(bool) + ALIGN_32B - 1) / ALIGN_32B * ALIGN_32B;
        this->pipe.InitBuffer(this->mask_queue, BUFFER_NUM, buf_len);

        buf_len = (this->draft_token_num * sizeof(int64_t) + ALIGN_32B - 1) / ALIGN_32B * ALIGN_32B;
        this->pipe.InitBuffer(this->positions_queue, BUFFER_NUM, buf_len);
        this->pipe.InitBuffer(this->retrive_index_queue, BUFFER_NUM, buf_len);
        this->pipe.InitBuffer(this->retrive_next_token_queue, BUFFER_NUM, buf_len);
        this->pipe.InitBuffer(this->retrive_next_sibling_queue, BUFFER_NUM, buf_len);
    }

    __aicore__ inline void Process()
    {
        for (int32_t bid = 0; bid < this->bs_per_core; bid++) {  // loop for batch dim
            AscendC::LocalTensor<int64_t> positions_ub = this->positions_queue.AllocTensor<int64_t>();
            AscendC::LocalTensor<int64_t> retrive_index_ub = this->retrive_index_queue.AllocTensor<int64_t>();
            AscendC::LocalTensor<int64_t> retrive_next_token_ub = this->retrive_next_token_queue.AllocTensor<int64_t>();
            AscendC::LocalTensor<int64_t> retrive_next_sibling_ub =
                this->retrive_next_sibling_queue.AllocTensor<int64_t>();
            for (int i = 0; i < this->draft_token_num; i++) {  // initial with -1
                retrive_next_token_ub.SetValue(i, -1);
                retrive_next_sibling_ub.SetValue(i, -1);
            }

            auto bs = bs_offset + bid;
            int64_t seq_tree_idx = draft_token_num * draft_token_num * bs;
            for (int i = 0; i < bs; i++) {
                seq_tree_idx += verified_seq_len_gm.GetValue(i) * draft_token_num;
            }
            int64_t seq_len = verified_seq_len_gm.GetValue(bs);

            for (uint32_t tid = 0; tid < this->draft_token_num; tid++) {  // loop for draft token dim
                Compute(bid, tid, seq_len, seq_tree_idx, positions_ub, retrive_index_ub, retrive_next_token_ub,
                        retrive_next_sibling_ub);
            }

            this->positions_queue.EnQue(positions_ub);
            this->retrive_index_queue.EnQue(retrive_index_ub);
            this->retrive_next_token_queue.EnQue(retrive_next_token_ub);
            this->retrive_next_sibling_queue.EnQue(retrive_next_sibling_ub);

            CopyOut(bid);
        }
    }

private:
    __aicore__ inline void Compute(int32_t bid, uint32_t tid, int64_t seq_len, int64_t seq_tree_idx,
                                   AscendC::LocalTensor<int64_t> &positions_ub,
                                   AscendC::LocalTensor<int64_t> &retrive_index_ub,
                                   AscendC::LocalTensor<int64_t> &retrive_next_token_ub,
                                   AscendC::LocalTensor<int64_t> &retrive_next_sibling_ub)
    {
        auto bs = bs_offset + bid;
        int token_tree_idx;
        if (tree_mask_mode == sglang::npu_kernel::TreeMaskMode::FULL_MASK) {
            // [seq_lens_sum * num_verify_tokens + num_verify_tokens * num_verify_tokens * bs, ]
            token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len;
        } else {
            // [num_verify_tokens * bs * num_verify_tokens, ]
            token_tree_idx = draft_token_num * draft_token_num * bs + draft_token_num * bs;
        }

        AscendC::LocalTensor<bool> tree_mask_ub = mask_queue.AllocTensor<bool>();
        tree_mask_ub.SetValue(0, true);
        for (int i = 1; i < draft_token_num; i++) {
            tree_mask_ub.SetValue(i, false);
        }

        int64_t position = 0;
        if (tid == 0) {
            positions_ub.SetValue(0, seq_len);

            int64_t retrive_index_offset = bs * draft_token_num;
            for (int64_t i = draft_token_num - 1; i > 0; --i) {
                int64_t current_token_idx = retrive_index_offset + i;
                retrive_index_ub.SetValue(i, current_token_idx);
                int64_t parent_tb_idx = selected_index_gm.GetValue(bid * (draft_token_num - 1) + i - 1) / topk;
                int64_t parent_position = 0;
                if (parent_tb_idx > 0) {
                    int64_t parent_token_idx = parent_list_gm.GetValue(bid * (topk * (depth - 1) + 1) + parent_tb_idx);
                    for (; parent_position < draft_token_num; ++parent_position) {
                        if (selected_index_gm.GetValue(bid * (draft_token_num - 1) + parent_position) ==
                            parent_token_idx) {
                            ++parent_position;
                            break;
                        }
                    }
                }
                if (parent_position == draft_token_num) {
                    continue;
                }

                if (retrive_next_token_ub.GetValue(parent_position) == -1) {
                    retrive_next_token_ub.SetValue(parent_position, i);
                } else {
                    int64_t origin_next_token = retrive_next_token_ub.GetValue(parent_position);
                    retrive_next_token_ub.SetValue(parent_position, i);
                    retrive_next_sibling_ub.SetValue(i, origin_next_token);
                }
            }
            retrive_index_ub.SetValue(0, bs * draft_token_num);
        } else {
            int64_t cur_position = tid - 1;
            while (true) {
                position += 1;
                tree_mask_ub.SetValue(1 + cur_position, true);
                int64_t parent_tb_idx = selected_index_gm.GetValue(bid * (draft_token_num - 1) + cur_position) / topk;
                if (parent_tb_idx == 0) {
                    break;
                }

                int64_t token_idx = parent_list_gm.GetValue(bid * (topk * (depth - 1) + 1) + parent_tb_idx);
                for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
                    if (selected_index_gm.GetValue(bid * (draft_token_num - 1) + cur_position) == token_idx) {
                        break;
                    }
                }
            }
            positions_ub.SetValue(tid, position + seq_len);
        }

        this->mask_queue.EnQue(tree_mask_ub);
        AscendC::LocalTensor<bool> mask_ub = mask_queue.DeQue<bool>();
        AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(this->draft_token_num * sizeof(bool)), 0, 0, 0};
        AscendC::DataCopyPad<bool>(this->tree_mask_gm[token_tree_idx], mask_ub, copy_params);
        mask_queue.FreeTensor(mask_ub);
    }

    __aicore__ inline void CopyOut(int32_t bid)
    {
        int offset = bid * this->draft_token_num;

        AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(this->draft_token_num * sizeof(int64_t)), 0, 0,
                                               0};

        AscendC::LocalTensor<int64_t> positions_ub = this->positions_queue.DeQue<int64_t>();
        AscendC::DataCopyPad<int64_t>(this->positions_gm[offset], positions_ub, copy_params);
        this->positions_queue.FreeTensor(positions_ub);

        AscendC::LocalTensor<int64_t> retrive_index_ub = this->retrive_index_queue.DeQue<int64_t>();
        AscendC::DataCopyPad<int64_t>(this->retrive_index_gm[offset], retrive_index_ub, copy_params);
        this->retrive_index_queue.FreeTensor(retrive_index_ub);

        AscendC::LocalTensor<int64_t> retrive_next_token_ub = this->retrive_next_token_queue.DeQue<int64_t>();
        AscendC::DataCopyPad<int64_t>(this->retrive_next_token_gm[offset], retrive_next_token_ub, copy_params);
        this->retrive_next_token_queue.FreeTensor(retrive_next_token_ub);

        AscendC::LocalTensor<int64_t> retrive_next_sibling_ub = this->retrive_next_sibling_queue.DeQue<int64_t>();
        AscendC::DataCopyPad<int64_t>(this->retrive_next_sibling_gm[offset], retrive_next_sibling_ub, copy_params);
        this->retrive_next_sibling_queue.FreeTensor(retrive_next_sibling_ub);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> mask_queue;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> positions_queue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> retrive_index_queue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> retrive_next_token_queue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> retrive_next_sibling_queue;

    AscendC::GlobalTensor<int64_t> parent_list_gm;
    AscendC::GlobalTensor<int64_t> selected_index_gm;
    AscendC::GlobalTensor<int64_t> verified_seq_len_gm;
    AscendC::GlobalTensor<bool> tree_mask_gm;
    AscendC::GlobalTensor<int64_t> positions_gm;
    AscendC::GlobalTensor<int64_t> retrive_index_gm;
    AscendC::GlobalTensor<int64_t> retrive_next_token_gm;
    AscendC::GlobalTensor<int64_t> retrive_next_sibling_gm;

    int64_t topk;
    int64_t depth;
    uint16_t draft_token_num;
    int64_t tree_mask_mode;

    int32_t batch_size;
    uint16_t mask_size;
    int32_t bs_per_core;

    uint32_t bs_offset = 0;
};

extern "C" __global__ __aicore__ void build_tree_efficient(GM_ADDR parent_list, GM_ADDR selected_index,
                                                           GM_ADDR verified_seq_len, GM_ADDR tree_mask,
                                                           GM_ADDR positions, GM_ADDR retrive_index,
                                                           GM_ADDR retrive_next_token, GM_ADDR retrive_next_sibling,
                                                           GM_ADDR workspace_in, GM_ADDR tiling_in)
{
    KernalBuildTree op;
    op.Init(parent_list, selected_index, verified_seq_len, tree_mask, positions, retrive_index, retrive_next_token,
            retrive_next_sibling, workspace_in, tiling_in);
    op.Process();
}
