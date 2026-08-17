#include "kernel_operator.h"
#include "compute_n_gram_ids.h"
#include "../op_host/compute_n_gram_ids_tiling.h"

extern "C" __global__ __aicore__ void compute_n_gram_ids(GM_ADDR oe_weights,
                                                             GM_ADDR oe_mods,
                                                             GM_ADDR exclusive_oe_embeder_size_sums,
                                                             GM_ADDR tokens,
                                                             GM_ADDR exclusive_req_len_sums,
                                                             GM_ADDR oe_token_table,
                                                             GM_ADDR row_indices,
                                                             GM_ADDR column_starts,
                                                             GM_ADDR oe_n_gram_ids,
                                                             GM_ADDR workspace,
                                                             GM_ADDR tiling) {
    AscendC::TPipe pipe;
    auto tiling_data =
        reinterpret_cast<__gm__ sglang::npu_kernel::ComputeNGramIdsTilingData *>(tiling);
    // 获取参数
    int batchSize = tiling_data->batchSize;
    int oeN = tiling_data->oeN;
    int oeK = tiling_data->oeK;
    int coreNum = tiling_data->coreNum;
    int totalTask = tiling_data->totalTask;
    int maxContextLen = tiling_data->maxContextLen;

    // 实例化算子对象
    ComputeNGramIds op;
    op.Init(oeN, oeK, oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
            tokens, exclusive_req_len_sums,
            oe_token_table, row_indices, column_starts, maxContextLen,
            oe_n_gram_ids, totalTask, coreNum, batchSize, &pipe);
    // 执行主逻辑
    (void)workspace;
    op.Process();
}
