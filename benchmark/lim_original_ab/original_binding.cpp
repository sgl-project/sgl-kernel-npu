#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/fused_lightning_indexer_manage/fused_lightning_indexer_manage_torch_adpt.h"
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
TORCH_LIBRARY(original_lim, ops) {
  ops.def(
      "run(Tensor index_weights, Tensor query_dequant_scale, Tensor query, "
      "Tensor index_key_dequant_scale, Tensor index_key_cache, Tensor "
      "index_block_table, Tensor actual_seq_lengths_query, Tensor "
      "actual_seq_lengths_key, Tensor offload_seq_lengths_key, Tensor "
      "num_cache_tokens, Tensor request_state, Tensor req_pool_entries, "
      "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, Tensor(c!) "
      "topk_dst_slots, Tensor(d!) topk_miss_counts, Tensor(e!) miss_src_ids, "
      "Tensor(f!) miss_dst_slots, Tensor(g!) miss_counts) -> ()");
  ops.impl("run", torch::kPrivateUse1,
           &vllm_ascend::npu_fused_lightning_indexer_manage);
}
