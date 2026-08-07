/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file minimax_indexer_vector.h
 * \brief SetWaitFlag helper for event synchronization (only surviving utility
 *        from the lightning-derived vector helpers; all sort/merge/scale
 *        functions were removed as dead code — MiniMax uses its own
 *        WholeReduceMax-based topk pipeline).
 */
#ifndef MINIMAX_INDEXER_VECTOR_H
#define MINIMAX_INDEXER_VECTOR_H

#include "kernel_operator.h"

namespace sglang::npu_kernel::MIServiceVec {
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

}  // namespace sglang::npu_kernel::MIServiceVec
#endif  // MINIMAX_INDEXER_VECTOR_H
