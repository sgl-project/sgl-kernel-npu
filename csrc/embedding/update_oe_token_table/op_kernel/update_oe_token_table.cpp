#include "update_oe_token_table.h"

#define TILING_KEY 10000

extern "C" __global__ __aicore__ void sgl_update_oe_token_table(GM_ADDR tokens, GM_ADDR req_lens, GM_ADDR row_indices,
                                                                GM_ADDR column_starts, GM_ADDR ignore_tokens,
                                                                GM_ADDR oe_token_table, GM_ADDR workspace,
                                                                GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);

    // 实例化算子对象
    UpdateOeTokenTable op(&pipe, &tiling_data);
    op.Init(tokens, req_lens, row_indices, column_starts, ignore_tokens, oe_token_table, workspace);

    // 执行主逻辑
    if (TILING_KEY_IS(TILING_KEY)) {
        op.Process();
    }
}
