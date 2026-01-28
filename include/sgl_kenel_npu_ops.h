// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_OPS_H
#define SGL_KERNEL_NPU_OPS_H

namespace sglang {
namespace npu_kernel {
at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y);

at::Tensor cache_loc_assign(const at::Tensor &req_indices,
                            const at::Tensor &token_pool,
                            const at::Tensor &start_offset,
                            const at::Tensor &end_offset,
                            const at::Tensor &out_cache_loc);

at::Tensor cache_loc_update(const at::Tensor &req_indices,
                            const at::Tensor &token_pool,
                            const at::Tensor &start_offset,
                            const at::Tensor &end_offset,
                            const at::Tensor &out_cache_loc);

bool assign_cache_op(at::Tensor &dst_tensor, const at::Tensor &src_tensor,
                     const at::Tensor &dst_start_idx,
                     const at::Tensor &dst_end_idx,
                     const at::Tensor &src_start_idx,
                     const at::Tensor &src_end_idx);

void alloc_extend(const at::Tensor &pre_lens, const at::Tensor &seq_lens,
                  const at::Tensor &last_loc, const at::Tensor &free_pages,
                  int64_t pages_size, at::Tensor &out_indices,
                  at::Tensor &values);

void build_tree_efficient(
    const at::Tensor &parent_list, const at::Tensor &selected_index,
    const at::Tensor &verified_seq_len, const at::Tensor &tree_mask,
    const at::Tensor &positions, const at::Tensor &retrive_index,
    const at::Tensor &retrive_next_token,
    const at::Tensor &retrive_next_sibling, int64_t topk, int64_t depth,
    int64_t draft_token_num, int64_t tree_mask_mode);

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>
mla_preprocess(const at::Tensor &hiddenState, const at::Tensor &gamma0,
               const at::Tensor &beta0, const at::Tensor &wdqkv,
               const at::Tensor &descale0, const at::Tensor &gamma1,
               const at::Tensor &beta1, const at::Tensor &wuq,
               const at::Tensor &descale1, const at::Tensor &gamma2,
               const at::Tensor &cos, const at::Tensor &sin,
               const at::Tensor &wuk, const at::Tensor &kv_cache,
               const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
               const at::Tensor &quant_scale0, const at::Tensor &quant_offset0,
               const at::Tensor &bias0, const at::Tensor &quant_scale1,
               const at::Tensor &quant_offset1, const at::Tensor &bias1,
               const c10::optional<at::Tensor> &ctkv_scale,
               const c10::optional<at::Tensor> &q_nope_scale,
               c10::optional<c10::string_view> cache_mode,
               c10::optional<c10::string_view> quant_mode, at::Tensor &q_out0,
               at::Tensor &kv_cache_out0, at::Tensor &q_out1,
               at::Tensor &kv_cache_out1);

void batch_matmul_transpose(const at::Tensor &tensor_a,
                            const at::Tensor &tensor_b, at::Tensor &tensor_c,
                            c10::optional<c10::string_view> format_mode,
                            c10::optional<c10::string_view> quant_mode);

void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k,
                              at::Tensor &device_v, at::Tensor &host_v,
                              const at::Tensor &device_indices,
                              const at::Tensor &host_indices, int64_t page_size,
                              int64_t direction, int64_t flags);

at::Tensor bgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &indices,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size);

void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices,
                 at::Tensor &y, double scale);

at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight,
                       at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size);

void sgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices,
                 at::Tensor &seq_len, at::Tensor &y, double scale);

at::Tensor sgemmv_expand(at::Tensor &x, at::Tensor &weight,
                         at::Tensor &lora_indices, at::Tensor &seq_len,
                         at::Tensor &lora_ranks, at::Tensor &slice_offsets,
                         at::Tensor &y);

void sgemmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices,
                   at::Tensor &seq_len, at::Tensor &lora_ranks,
                   at::Tensor &lora_scales, at::Tensor &y);

#ifdef BUILD_CATLASS_MODULE
void catlass_matmul_basic(const at::Tensor &tensor_a,
                          const at::Tensor &tensor_b, at::Tensor &tensor_c,
                          c10::optional<c10::string_view> format_mode);
#endif

at::Tensor lightning_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    c10::optional<c10::string_view> layout_query,
    c10::optional<c10::string_view> layout_key,
    c10::optional<int64_t> sparse_count, c10::optional<int64_t> sparse_mode);

/**
 * @brief Triangular inverse of input tensor where last two dimensions represent
 * a matrix.
 *
 * @param [in] tensor_in Tensor of dimensions (..., n, n) where `n` is
 * the matrix size.
 * @return at::Tensor Returns tensor of same shape where each matrix of size n
 * is inversed.
 */
at::Tensor tri_inv_col_sweep(const at::Tensor &tensor_in);
} // namespace npu_kernel

} // namespace sglang

#endif // SGL_KERNEL_NPU_OPS_H
