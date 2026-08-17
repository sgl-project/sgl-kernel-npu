//
// Created by 陈鸿洛 on 2025/11/23.
//

#ifndef NPU_AI_OPERATION_KERNEL_COMPUTE_N_GRAM_IDS_H
#define NPU_AI_OPERATION_KERNEL_COMPUTE_N_GRAM_IDS_H

#define PIPE_SYNC_EVENT(event_type)                                                       \
    do {                                                                                  \
        event_t event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event_type)); \
        AscendC::SetFlag<event_type>(event_id);                                           \
        AscendC::WaitFlag<event_type>(event_id);                                          \
    } while (0)

template <typename __T>
__aicore__ inline void DataCopyG2U(const AscendC::LocalTensor<__T>& dst, const AscendC::GlobalTensor<__T>& src,
                                   const uint32_t calCount) {
  AscendC::DataCopyExtParams param;
  AscendC::DataCopyPadExtParams<__T> ext_param;
  param.blockCount = 1;
  param.blockLen = calCount * sizeof(__T);
  param.srcStride = 0;
  param.dstStride = 0;
  AscendC::DataCopyPad(dst, src, param, ext_param);
}

template <typename __T>
__aicore__ inline void DataCopyU2G(const AscendC::GlobalTensor<__T>& dst, const AscendC::LocalTensor<__T>& src,
                                   const uint32_t calCount) {
  int total_bytes = calCount * sizeof(__T);
  int copy_bytes = (total_bytes / 32) * 32;
  int copyCount = copy_bytes / sizeof(__T);
  int remainCount = calCount - copyCount;
  if (copyCount > 0) {
    AscendC::DataCopy(dst, src, copyCount);
  }
  if (remainCount > 0) {
    AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(remainCount * sizeof(__T)), 0, 0};
    AscendC::DataCopyPad(dst[copyCount], src[copyCount], copyParams);
  }
}

template <typename T>
__aicore__ inline T Min(T a, T b) {
return a > b ? b : a;
}

class ComputeNGramIds {
public:
  static constexpr int32_t BUFFER_NUM = 1;

  __aicore__ inline ComputeNGramIds(){}
  // 初始化函数，完成内存初始化相关操作
  __aicore__ inline void Init(int oe_n, int oe_k, GM_ADDR oe_weights, GM_ADDR oe_mods, GM_ADDR exclusive_oe_embeder_size_sums,
                              GM_ADDR tokens, GM_ADDR exclusive_req_len_sums,
                              GM_ADDR oe_token_table, GM_ADDR row_indices, GM_ADDR column_starts, int max_context_len,
                              GM_ADDR oe_n_gram_ids, int totalTask, int coreNum, int batch_size,
                               AscendC::TPipe* pipe) {
    int coreId = AscendC::GetBlockIdx();
    // AscendC::printf("coreId: %d %d start\n", coreId, max_context_len);
    oeN = oe_n;
    oeK = oe_k;
    oeModsSize = (oe_n-1) * oe_k;
    oeWeightsSize = oeModsSize * oe_n;
    maxContextLen = max_context_len;

    oeWeightsGm.SetGlobalBuffer((__gm__ int32_t*)oe_weights, oeWeightsSize);
    oeModsGm.SetGlobalBuffer((__gm__ int32_t*)oe_mods, oeModsSize);
    exclusiveEmbedderSizeSumsGm.SetGlobalBuffer((__gm__ int32_t*)exclusive_oe_embeder_size_sums, (1 + oeModsSize));

    tokensGm.SetGlobalBuffer((__gm__ int32_t*)tokens); // token_num
    exclusiveReqLenSumsGm.SetGlobalBuffer((__gm__ int32_t*)exclusive_req_len_sums, batch_size);

    oeTokenTableGm.SetGlobalBuffer((__gm__ int32_t*)oe_token_table); // [-1, maxContextLen]
    rowIndicesGm.SetGlobalBuffer((__gm__ int64_t*)row_indices, batch_size);
    columnStartsGm.SetGlobalBuffer((__gm__ int32_t*)column_starts, batch_size);

    oeNGramIdsGm.SetGlobalBuffer((__gm__ int32_t*)oe_n_gram_ids); // [token_num*(oeN-1)*oeK]

    pipe->InitBuffer(oeWeightsBuf, oeWeightsSize * sizeof(int32_t));
    pipe->InitBuffer(oeModsBuf, oeModsSize * sizeof(int32_t));
    pipe->InitBuffer(exclusiveEmbedderSizeSumsBuf, (oeModsSize + 1) * sizeof(int32_t));
    pipe->InitBuffer(exclusiveReqLenSumsBuf, batch_size * sizeof(int32_t));
    pipe->InitBuffer(rowIndicesBuf, batch_size * sizeof(int64_t));
    pipe->InitBuffer(columnStartsBuf, batch_size * sizeof(int32_t));

    pipe->InitBuffer(oeNGramIdsQueue, 1, 1 * sizeof(int32_t));

    // get start index for current core, core parallel
    int taskPerCore = (totalTask + coreNum - 1) / coreNum;
    startTask = coreId * taskPerCore;
    endTask = (coreId + 1) * taskPerCore;
    endTask = min(endTask, totalTask);
    batchSize = batch_size;
    // AscendC::printf("coreId: %d, startTask: %d, endTask: %d oeN: %d, oeK: %d, oeWeightsSize: %d, oeModsSize: %d\n", coreId, startTask, endTask, oeN, oeK, oeWeightsSize, oeModsSize);
  }

  __aicore__ inline void Process() {
//    AscendC::printf("ComputeNGramIds Process");
    // 1. set Local
    auto oeWeightsLocal = oeWeightsBuf.Get<int32_t>(0);
    auto oeModsLocal = oeModsBuf.Get<int32_t>(0);
    auto exclusiveEmbedderSizeSumsLocal = exclusiveEmbedderSizeSumsBuf.Get<int32_t>(0);

    auto exclusiveReqLenSumsLocal = exclusiveReqLenSumsBuf.Get<int32_t>(0);

    auto rowIndicesLocal = rowIndicesBuf.Get<int64_t>(0);
    auto columnStartsLocal = columnStartsBuf.Get<int32_t>(0);

    DataCopyG2U(oeWeightsLocal, oeWeightsGm, oeWeightsSize);
    DataCopyG2U(oeModsLocal, oeModsGm, oeModsSize);
    DataCopyG2U(exclusiveEmbedderSizeSumsLocal, exclusiveEmbedderSizeSumsGm, oeModsSize + 1);
    DataCopyG2U(exclusiveReqLenSumsLocal, exclusiveReqLenSumsGm, batchSize);
    DataCopyG2U(rowIndicesLocal, rowIndicesGm, batchSize);
    DataCopyG2U(columnStartsLocal, columnStartsGm, batchSize);
    PIPE_SYNC_EVENT(AscendC::HardEvent::MTE2_S);

    int coreId = AscendC::GetBlockIdx();
    // 2. compute
    for (int task_id = startTask; task_id < endTask; ++task_id) {
      // 2.1 compute b, n, k
      int req_id = task_id / oeModsSize;
      int other = task_id % oeModsSize;
      int n = other / oeK;
      int k = other % oeK;
      const int oe_weight_base_idx = n * oeK * oeN + k * oeN;
      // 计算当前请求在token table中的起始index,cur index
      const int req_token_table_index = (int)rowIndicesLocal.GetValue(req_id) * maxContextLen;
      const int req_token_table_index_cur = req_token_table_index + columnStartsLocal.GetValue(req_id);

      // 2.2 get reqLenSumsLocal, oeInfoLenSumsLocal
      int32_t start = 0;
      if (req_id >= 1) {
        start = exclusiveReqLenSumsLocal.GetValue(req_id-1);
      }
      int32_t end = exclusiveReqLenSumsLocal.GetValue(req_id);
      uint64_t oe_mod = (uint64_t)oeModsLocal.GetValue(n * oeK + k); // shape: [oeN-1, oeK]
      // AscendC::printf("coreId: %d, task_id: %d, b: %d, n: %d, k: %d start: %d end: %d\n", coreId, task_id, req_id, n, k, start, end);
      for (int i = start; i < end; ++i) {
        uint64_t n_gram_id = 0;
        int current_token_offset = i - start;
        // 计算当前token在token table中的位置
        int current_token_table_index = req_token_table_index_cur + current_token_offset;

        for (int j = 0; j < n + 2; ++j) {
          if (current_token_table_index - j < req_token_table_index) break;

          int token = oeTokenTableGm.GetValue(current_token_table_index - j);
          PIPE_SYNC_EVENT(AscendC::HardEvent::MTE2_S);
          if (token < 0) break; // ignore token

          int weight = oeWeightsLocal.GetValue(oe_weight_base_idx + j);

          uint64_t term = (uint64_t)token * (uint64_t)weight;
          n_gram_id += term % oe_mod;
           // AscendC::printf("coreId: %d, task_id: %d, token: %d, weight: %d, term: %d, n_gram_id: %lld, mod: %d\n", coreId, task_id, token, weight, term, n_gram_id, oe_mod);
        }
        n_gram_id = n_gram_id % oe_mod;
//        add bias
        n_gram_id = n_gram_id + exclusiveEmbedderSizeSumsLocal.GetValue(n * oeK +k);
        int store_idx = i * (oeN - 1) * oeK + n * oeK + k;
		// AscendC::printf("coreId: %d, task_id: %d, store_idx: %d, i: %d, n: %d, k: %d, n_gram_id: %llu\n", coreId, task_id, store_idx, i, n, k, n_gram_id);
        // shape: [seq_len, oeN-1, oeK]
        auto nGramIdLocal = oeNGramIdsQueue.AllocTensor<int32_t>();
        nGramIdLocal.SetValue(0, (int32_t)n_gram_id);
        DataCopyU2G(oeNGramIdsGm[store_idx], nGramIdLocal, 1);
        oeNGramIdsQueue.EnQue(nGramIdLocal);
        nGramIdLocal=oeNGramIdsQueue.DeQue<int32_t>();
        oeNGramIdsQueue.FreeTensor(nGramIdLocal);
      }
    }
  }

private:
  AscendC::GlobalTensor<int32_t> oeWeightsGm;
  AscendC::GlobalTensor<int32_t> oeModsGm;
  AscendC::GlobalTensor<int32_t> exclusiveEmbedderSizeSumsGm;

  AscendC::GlobalTensor<int32_t> tokensGm;
  AscendC::GlobalTensor<int32_t> exclusiveReqLenSumsGm;

  AscendC::GlobalTensor<int32_t> oeTokenTableGm;
  AscendC::GlobalTensor<int64_t> rowIndicesGm;
  AscendC::GlobalTensor<int32_t> columnStartsGm;

  AscendC::GlobalTensor<int32_t> oeNGramIdsGm;

  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> oeNGramIdsQueue;

  // local tensor，global tensor小，直接全部缓存到UB
  AscendC::TBuf<AscendC::TPosition::VECCALC> oeWeightsBuf, oeModsBuf, exclusiveEmbedderSizeSumsBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> exclusiveReqLenSumsBuf, rowIndicesBuf, columnStartsBuf;

  int startTask;
  int endTask;

  int oeN;
  int oeK;
  int oeWeightsSize;
  int oeModsSize;
  int maxContextLen;
  int batchSize;
};


#endif //NPU_AI_OPERATION_KERNEL_COMPUTE_N_GRAM_IDS_H
