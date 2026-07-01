#include "kernel_operator.h"
#include "update_oe_token_table.h"
#include "../op_host/update_oe_token_table_tiling.h"

extern "C" __global__ __aicore__ void update_oe_token_table(GM_ADDR tokens,
                                                            GM_ADDR req_lens,
                                                            GM_ADDR row_indices,
                                                            GM_ADDR column_starts,
                                                            GM_ADDR ignore_tokens,
                                                            GM_ADDR oe_token_table,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    auto tiling_data =
        reinterpret_cast<__gm__ sglang::npu_kernel::UpdateOeTokenTableTilingData *>(tiling);

    UpdateOeTokenTable op(&pipe, tiling_data);
    op.Init(tokens, req_lens, row_indices, column_starts, ignore_tokens, oe_token_table, workspace);
    op.Process();
}
