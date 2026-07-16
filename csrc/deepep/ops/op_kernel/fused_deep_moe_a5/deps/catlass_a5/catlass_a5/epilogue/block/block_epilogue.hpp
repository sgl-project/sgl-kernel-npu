/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP

#include "catlass_a5/catlass.hpp"

namespace Catlass::Epilogue::Block {

template <
    class DispatchPolicy,
    class... Args
>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace Catlass::Epilogue::Block

#include "catlass_a5/epilogue/block/block_epilogue_elemwise_no_source.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_elemwise_one_source.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_fa_softmax.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_fa_rescale_o.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_mla_softmax.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_mla_rescale_o.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_mla_fd_rescale_o.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_per_token_dequant.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_per_token_dequant_tla.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_gemm.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_gemv.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_mla_tp1_softmax.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_mla_tp1_rescale_o.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_amla_tp1_softmax.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_amla_tp1_rescale_o.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_online_softmax_no_mask.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_rescale_o_no_split_row.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_w4a4_per_token_per_channel_dequant.hpp"

#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
#include "catlass_a5/epilogue/block/block_epilogue_fa_softmax_ascend950.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_fa_rescale_o_ascend950.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_fixpipe.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_per_group_per_block.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_dequant.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_dual_level_quant_mx.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_per_block_quant_tla.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_visitor.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_swiglu_mx_quant.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_finalize_routing.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_flash_attention_online_softmax_high_prec.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_flash_attention_online_softmax_low_prec.hpp"
#include "catlass_a5/epilogue/block/block_epilogue_flash_attention_rescale_o.hpp"
#endif

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
