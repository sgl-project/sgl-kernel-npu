#include "kernel_operator.h"

#ifdef SGL_DCP_CANN_PACKAGE
#include "dcp_verify_page_tables_tiling.h"
#endif

namespace {
constexpr uint32_t kTile = 128;

template <typename Index>
__aicore__ inline void Run(GM_ADDR slots, GM_ADDR requests, GM_ADDR ordinary, GM_ADDR local, uint32_t rows,
                           uint32_t slot_rows, uint32_t slot_cols, uint32_t ordinary_cols, uint32_t local_cols,
                           uint32_t active_cols, uint32_t width, uint32_t page, uint32_t dcp)
{
    using namespace AscendC;
    GlobalTensor<int32_t> slot, full_out, local_out;
    GlobalTensor<Index> req;
    slot.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slots));
    req.SetGlobalBuffer(reinterpret_cast<__gm__ Index *>(requests), rows);
    full_out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ordinary), rows * ordinary_cols);
    local_out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(local), rows * width * local_cols);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> full_queue, local_queue;
    pipe.InitBuffer(input_queue, 1, kTile * 32);
    pipe.InitBuffer(full_queue, 1, kTile * sizeof(int32_t));
    pipe.InitBuffer(local_queue, 1, kTile * sizeof(int32_t));

    uint32_t page_shift = 0;
    for (auto remaining = page; remaining > 1; remaining >>= 1) {
        ++page_shift;
    }
    const uint32_t dcp_shift = dcp == 4 ? 2 : 1;
    const uint32_t all_cols = ordinary_cols > local_cols * dcp ? ordinary_cols : local_cols * dcp;
    const uint32_t tiles = (all_cols + kTile - 1) / kTile;
    for (uint32_t job = GetBlockIdx(); job < rows * tiles; job += GetBlockNum()) {
        const uint32_t row = job / tiles;
        const uint32_t col = (job % tiles) * kTile;
        const uint32_t local_col = col >> dcp_shift;
        const uint32_t full_count =
            col >= ordinary_cols ? 0 : (ordinary_cols - col < kTile ? ordinary_cols - col : kTile);
        const uint32_t local_tile = kTile >> dcp_shift;
        const uint32_t local_count =
            local_col >= local_cols ? 0 : (local_cols - local_col < local_tile ? local_cols - local_col : local_tile);
        const auto request = req.GetValue(row);
        const bool valid_request = request >= 0 && static_cast<uint64_t>(request) < slot_rows;
        const uint32_t valid =
            !valid_request || col >= active_cols ? 0 : (active_cols - col < kTile ? active_cols - col : kTile);
        auto input = input_queue.AllocTensor<int32_t>();
        if (valid > 0) {
            const uint64_t start = static_cast<uint64_t>(request) * slot_cols + col * page;
            // Each 4-byte page start occupies one 32-byte UB block.
            DataCopyExtParams p{static_cast<uint16_t>(valid), 4, (page - 1) * 4, 0, 0};
            DataCopyPad(input, slot[start], p, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        input_queue.EnQue(input);
        input = input_queue.DeQue<int32_t>();
        auto full = full_queue.AllocTensor<int32_t>();
        auto local_tile_out = local_queue.AllocTensor<int32_t>();
        for (uint32_t i = 0; i < full_count; ++i) {
            int32_t id = i < valid ? input.GetValue(i * 8) >> page_shift : 0;
            full.SetValue(i, id);
        }
        for (uint32_t i = 0; i < local_count; ++i) {
            const uint32_t source = i << dcp_shift;
            int32_t id = source < valid ? input.GetValue(source * 8) >> (page_shift + dcp_shift) : 0;
            local_tile_out.SetValue(i, id);
        }
        input_queue.FreeTensor(input);
        full_queue.EnQue(full);
        local_queue.EnQue(local_tile_out);
        full = full_queue.DeQue<int32_t>();
        local_tile_out = local_queue.DeQue<int32_t>();
        if (full_count > 0) {
            DataCopyPad(full_out[row * ordinary_cols + col], full, DataCopyExtParams{1, full_count * 4, 0, 0, 0});
        }
        if (local_count > 0) {
            for (uint32_t q = 0; q < width; ++q) {
                DataCopyPad(local_out[(row * width + q) * local_cols + local_col], local_tile_out,
                            DataCopyExtParams{1, local_count * 4, 0, 0, 0});
            }
        }
        full_queue.FreeTensor(full);
        local_queue.FreeTensor(local_tile_out);
    }
}
}  // namespace

#ifdef SGL_DCP_CANN_PACKAGE
extern "C" __global__ __aicore__ void sgl_dcp_verify_page_tables(GM_ADDR slots, GM_ADDR requests, GM_ADDR ordinary,
                                                                 GM_ADDR local, GM_ADDR ordinary_ref, GM_ADDR local_ref,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::SglDcpVerifyPageTablesTilingData);
    GET_TILING_DATA(data, tiling);
    if (data.index_bytes == sizeof(int64_t)) {
        Run<int64_t>(slots, requests, ordinary_ref, local_ref, data.rows, data.slot_rows, data.slot_cols,
                     data.ordinary_cols, data.local_cols, data.active_cols, data.query_len, data.page_size,
                     data.dcp_size);
    } else {
        Run<int32_t>(slots, requests, ordinary_ref, local_ref, data.rows, data.slot_rows, data.slot_cols,
                     data.ordinary_cols, data.local_cols, data.active_cols, data.query_len, data.page_size,
                     data.dcp_size);
    }
}
#else
extern "C" __global__ __aicore__ void sgl_dcp_verify_page_tables(GM_ADDR slots, GM_ADDR requests, GM_ADDR ordinary,
                                                                 GM_ADDR local, uint32_t rows, uint32_t slot_rows,
                                                                 uint32_t slot_cols, uint32_t ordinary_cols,
                                                                 uint32_t local_cols, uint32_t active_cols,
                                                                 uint32_t width, uint32_t page, uint32_t dcp,
                                                                 uint32_t index_bytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (index_bytes == 8) {
        Run<int64_t>(slots, requests, ordinary, local, rows, slot_rows, slot_cols, ordinary_cols, local_cols,
                     active_cols, width, page, dcp);
    } else {
        Run<int32_t>(slots, requests, ordinary, local, rows, slot_rows, slot_cols, ordinary_cols, local_cols,
                     active_cols, width, page, dcp);
    }
}
#endif
